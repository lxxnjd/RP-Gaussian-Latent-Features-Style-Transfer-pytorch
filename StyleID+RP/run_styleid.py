import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import copy

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import pickle

import math
import warnings
from scipy.sparse.linalg import eigsh
from ldm.modules.diffusionmodules.model import Encoder

torch.cuda.set_device(1)

feat_maps = []

def save_img_from_sample(model, samples_ddim, fname):
    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
    x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(x_sample.astype(np.uint8))
    img.save(fname)

def feat_merge(opt, cnt_feats, sty_feats, start_step=0):
    feat_maps = [{'config': {
                'gamma':opt.gamma,
                'T':opt.T,
                'timestep':_,
                }} for _ in range(50)]

    for i in range(len(feat_maps)):
        if i < (50 - start_step):
            continue
        cnt_feat = cnt_feats[i]
        sty_feat = sty_feats[i]
        ori_keys = sty_feat.keys()

        for ori_key in ori_keys:
            if ori_key[-1] == 'q':
                feat_maps[i][ori_key] = cnt_feat[ori_key]
            if ori_key[-1] == 'k' or ori_key[-1] == 'v':
                feat_maps[i][ori_key] = sty_feat[ori_key]
    return feat_maps


def load_img(path):
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"Loaded input image of size ({x}, {y}) from {path}")
    h = w = 512
    image = transforms.CenterCrop(min(x,y))(image)
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def adain(cnt_feat, sty_feat, model=None, compress_ratio=1, group_nums=1):
    groups_content, groups_style, A_content, content_c, content_h, content_w = sample_split(cnt_feat, sty_feat, compress_ratio, group_nums)
    feat_transfer = cs_AdaIN(groups_content, groups_style, group_nums)
    print('Mean of absolute values of transferred feature tensor: ', torch.mean(torch.abs(feat_transfer)))
    content_reconstruction = ista(torch.as_tensor(feat_transfer, device='cuda', dtype=torch.float),
                      torch.as_tensor(A_content, device='cuda', dtype=torch.float), alpha=0, fast=True,
                      lr='auto', maxiter=200, tol=1e-7, backtrack=False, eta_backtrack=1.5, verbose=False)
    feat = content_reconstruction.reshape(content_c, content_h, content_w).unsqueeze(0)
    print('Mean of absolute values of reconstructed feature tensor: ', torch.mean(torch.abs(feat)))
    
    # save_feat_to_file(feat,'ours_feat.txt')
    
    # if feat.shape[1] > 0:
    #     norm_layer = torch.nn.GroupNorm(
    #         num_groups=min(32, feat.shape[1]),
    #         num_channels=feat.shape[1],
    #         eps=1e-6,
    #         affine=Ture
    #     ).to(feat.device)
        
    #     feat = norm_layer(feat)

    # Mask test: filter feature values with absolute values less than multiple of mean
    # abs_feat = torch.abs(feat)
    # u = torch.mean(abs_feat)
    # threshold = 1 * u
    # feat[abs_feat < threshold] = 0

    return feat

def save_feat_to_file(feat, filename):
    with open(filename, 'w') as f:
        f.write(f"Feature shape: {feat.shape}\n\n")
        
        if feat.dim() == 4:
            for b in range(feat.shape[0]):
                for c in range(feat.shape[1]):
                    f.write(f"---- Batch {b}, Channel {c} ----\n")
                    for h in range(feat.shape[2]):
                        for w in range(feat.shape[3]):
                            f.write(f"{feat[b, c, h, w].item():.6f} ")
                        f.write("\n")
                    f.write("\n")

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def sample_split(content_feat, style_feat, compress_ratio, group_nums):
    content_feat = content_feat.squeeze(0)
    style_feat = style_feat.squeeze(0)
    content_c, content_h, content_w = content_feat.shape
    style_c, style_h, style_w = style_feat.shape

    content_feat_vector = content_feat.view(content_c, content_h*content_w)
    style_feat_vector = style_feat.view(style_c, style_h*style_w)
 
    m_c = int(compress_ratio*content_h*content_w)
    m_s = int(compress_ratio*style_h*style_w)

    # Four types of random matrix:
    # 1. Random Gaussian
    A_content=torch.randn(m_c,content_h*content_w).cuda()
    A_style = torch.randn(m_s, style_h * style_w).cuda()

    # torch.save(A_content, 'table/A_content.pth')
    # torch.save(A_style, 'table/A_style.pth')
    # A_content = torch.load('table/A_content.pth')
    # A_style = torch.load('table/A_style.pth')
    # 2. Random Gaussian: set >0 to 1, <0 to -1
    # A_content = torch.randn(m_c, content_h * content_w).cuda()
    # A_content = torch.sign(A_content)
    # A_style = torch.randn(m_s, style_h * style_w).cuda()
    # A_style = torch.sign(A_style)

    # 3. Random 0,-1,1 matrix
    # A_content = torch.zeros(m_c, content_h * content_w).cuda()
    # num_ones_content = int(round((m_c) ** (1 / 2)))
    # for i in range(content_h * content_w):
    #     indices = torch.randperm(m_c)[:num_ones_content]
    #     values = (torch.randint(0, 2, (num_ones_content,)).float() * 2 - 1).cuda()
    #     A_content[indices, i] = values
    # A_style = torch.zeros(m_s, style_h * style_w).cuda()
    # num_ones_style = int(round((m_s) ** (1 / 2)))
    # for i in range(style_h * style_w):
    #     indices = torch.randperm(m_s)[:num_ones_style]
    #     values = (torch.randint(0, 2, (num_ones_style,)).float() * 2 - 1).cuda()
    #     A_style[indices, i] = values

    # 4. {0,1}: random num_1=sqrt(m) ones per column, others 0
    # A_content = torch.zeros(m_c, content_h * content_w).cuda()
    # num_ones_content = int(round((m_c) ** (1 / 2)))
    # for i in range(content_h * content_w):
    #     indices = torch.randperm(m_c)[:num_ones_content]
    #     A_content[indices, i] = 1
    # A_style = torch.zeros(m_s, style_h * style_w).cuda()
    # num_ones_style = int(round((m_s) ** (1 / 2)))
    # for i in range(style_h * style_w):
    #     indices = torch.randperm(m_s)[:num_ones_style]
    #     A_style[indices, i] = 1

    A_content = A_content/((A_content ** 2).sum(dim=1, keepdim=True)).sqrt()
    A_style = A_style/((A_style ** 2).sum(dim=1, keepdim=True)).sqrt()
    
    # folder_path = 'A_cs0.4'
    # os.makedirs(folder_path, exist_ok=True)
    # file_path_content = os.path.join(folder_path, 'A_content_cs0.4.pt')
    # torch.save(A_content, file_path_content)
    # file_path_style = os.path.join(folder_path, 'A_style_cs0.4.pt')
    # torch.save(A_style, file_path_style)

    # A_content = torch.load('A_cs0.4/A_content_cs0.4.pt')
    # A_style = torch.load('A_cs0.4/A_style_cs0.4.pt')
    
    y_content = torch.matmul(A_content, content_feat_vector.T).T.cuda()
    y_style = torch.matmul(A_style, style_feat_vector.T).T.cuda()
    print(y_content.size(), y_style.size())

    sizes_content = [int(m_c / group_nums)]*(group_nums - 1)
    sizes_content.append(m_c - (group_nums-1)*int(m_c/group_nums))
    groups_content = torch.split(y_content, sizes_content, dim=1)
    sizes_style = [int(m_s / group_nums)]*(group_nums - 1)
    sizes_style.append(m_s-(group_nums-1)*int(m_s/group_nums))
    groups_style = torch.split(y_style, sizes_style, dim=1)

    # Ablation experiment: only group without compression
    # f_c = content_h * content_w
    # f_s = style_h * style_w
    # sizes_content = [int(f_c / group_nums)]*(group_nums - 1)
    # sizes_content.append(f_c - (group_nums-1)*int(f_c/group_nums))
    # groups_content = torch.split(content_feat_vector, sizes_content, dim=1)
    # sizes_style = [int(f_s / group_nums)]*(group_nums - 1)
    # sizes_style.append(f_s-(group_nums-1)*int(f_s/group_nums))
    # groups_style = torch.split(style_feat_vector, sizes_style, dim=1)

    return groups_content, groups_style, A_content, content_c, content_h, content_w

def cs_AdaIN(groups_content, groups_style, group_nums):
    zero_list = [torch.zeros_like(t) for t in groups_content]
    for i in range(group_nums):
        zero_list[i] = adaptive_instance_normalization(groups_content[i], groups_style[i])
    target=torch.cat(zero_list, dim=1)
    return target

def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    # Add noise after projection to increase diversity
    # normalized_feat = (content_feat - content_mean.expand(
    #    size)+torch.normal(0, content_mean/16)) / content_std.expand(size)
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    
    # print('style loss:', criterionMSE(content_std, style_std) + criterionMSE(content_mean, style_mean))

    # normalized_feat_s = (style_feat - style_mean.expand(
    #     style_feat.size())) / style_std.expand(style_feat.size())

    # print('content loss:', criterionMSE(normalized_feat, normalized_feat_s))
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    c = size[0]
    feat_var = feat.contiguous().var(dim=1) + eps
    feat_std = feat_var.sqrt().contiguous().view(c,1)
    feat_mean = feat.contiguous().mean(dim=1).contiguous().view(c,1)
    return feat_mean, feat_std

def _lipschitz_constant(W):
    WtW = torch.matmul(W.t(), W).float()
    eigenvalues = torch.linalg.eigvalsh(WtW)
    L = eigenvalues.max()
    return L

def backtracking(z, x, weight, alpha, lr0, eta=1.5, maxiter=1000, verbose=False):
    if eta <= 1:
        raise ValueError('eta must be > 1.')

    resid_0 = torch.matmul(z, weight.T) - x
    fval_0 = 0.5 * resid_0.pow(2).sum()
    fgrad_0 = torch.matmul(resid_0, weight)

    def calc_F(z_1):
        resid_1 = torch.matmul(z_1, weight.T) - x
        return 0.5 * resid_1.pow(2).sum() + alpha * z_1.abs().sum()

    def calc_Q(z_1, t):
        dz = z_1 - z
        return (fval_0
                + (dz * fgrad_0).sum()
                + (0.5 / t) * dz.pow(2).sum()
                + alpha * z_1.abs().sum())

    lr = lr0
    z_next = None
    for i in range(maxiter):
        z_next = F.softshrink(z - lr * fgrad_0, alpha * lr)
        F_next = calc_F(z_next)
        Q_next = calc_Q(z_next, lr)
        if verbose:
            print('iter: %4d,  t: %0.5f,  F-Q: %0.5f' % (i, lr, F_next-Q_next))
        if F_next <= Q_next:
            break
        lr = lr / eta
    else:
        warnings.warn('backtracking line search failed. Reverting to initial step size')
        lr = lr0
        z_next = F.softshrink(z - lr * fgrad_0, alpha * lr)

    return z_next, lr

def initialize_code(x, weight):
    n_samples = x.size(0)
    n_components = weight.size(1)
    z0 = x.new_zeros(n_samples, n_components)
    return z0

def ista(x, weight, alpha=0, fast=True, lr='auto', maxiter=0,
         tol=1e-5, backtrack=False, eta_backtrack=1.5, verbose=False):
    z0 = initialize_code(x, weight)

    if lr == 'auto':
        L = _lipschitz_constant(weight)
        lr = 1 / L
    tol = z0.numel() * tol

    def loss_fn(z_k):
        resid = torch.matmul(z_k, weight.T) - x
        loss = 0.5 * resid.pow(2).sum() + alpha * z_k.abs().sum()
        return loss / x.size(0)

    def rss_grad(z_k):
        resid = torch.matmul(z_k, weight.T) - x
        return torch.matmul(resid, weight)

    z = z0
    if fast:
        y, t = z0, 1
    for _ in range(maxiter):
        if verbose:
            print('loss: %0.4f' % loss_fn(z))

        z_prev = y if fast else z
        if backtrack:
            z_next, _ = backtracking(z_prev, x, weight, alpha, lr, eta_backtrack)
        else:
            z_next = F.softshrink(z_prev - lr * rss_grad(z_prev), alpha * lr)

        if (z - z_next).abs().sum() <= tol:
            z = z_next
            break

        if fast:
            t_next = (1 + math.sqrt(1 + 4 * t**2)) / 2
            y = z_next + ((t-1)/t_next) * (z_next - z)
            t = t_next
        z = z_next

    return z

def _calc_feat_flatten_mean_std(feat):
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std

def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())

def coral(source, target):
    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnt', default = './data/cnt')
    parser.add_argument('--sty', default = './data/sty')
    parser.add_argument('--ddim_inv_steps', type=int, default=50)
    parser.add_argument('--save_feat_steps', type=int, default=50)
    parser.add_argument('--start_step', type=int, default=49)
    parser.add_argument('--ddim_eta', type=float, default=0.0)
    parser.add_argument('--H', type=int, default=512, help='image height, in pixel space')
    parser.add_argument('--W', type=int, default=512, help='image width, in pixel space')
    parser.add_argument('--C', type=int, default=4, help='latent channels')
    parser.add_argument('--f', type=int, default=8, help='downsampling factor')
    parser.add_argument('--T', type=float, default=1.5, help='attention temperature scaling hyperparameter')
    parser.add_argument('--gamma', type=float, default=0.75, help='query preservation hyperparameter')
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11', help='injection attention feature layers')
    parser.add_argument('--model_config', type=str, default='models/ldm/stable-diffusion-v1/v1-inference.yaml', help='model config')
    parser.add_argument('--precomputed', type=str, default='./precomputed_feats', help='save path for precomputed feature')
    parser.add_argument('--ckpt', type=str, default='models/ldm/stable-diffusion-v1/model.ckpt', help='model checkpoint')
    parser.add_argument('--precision', type=str, default='autocast', help='choices: ["full", "autocast"]')
    parser.add_argument('--output_path', type=str, default='output_ours/0714_fulu')
    parser.add_argument("--without_init_adain", action='store_true')
    parser.add_argument("--without_attn_injection", action='store_true')
    opt = parser.parse_args()

    feat_path_root = opt.precomputed

    seed_everything(22)
    output_path = opt.output_path
    os.makedirs(output_path, exist_ok=True)
    if len(feat_path_root) > 0:
        os.makedirs(feat_path_root, exist_ok=True)
    
    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")

    self_attn_output_block_indices = list(map(int, opt.attn_layer.split(',')))
    ddim_inversion_steps = opt.ddim_inv_steps
    save_feature_timesteps = ddim_steps = opt.save_feat_steps

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False) 
    time_range = np.flip(sampler.ddim_timesteps)
    idx_time_dict = {}
    time_idx_dict = {}
    for i, t in enumerate(time_range):
        idx_time_dict[t] = i
        time_idx_dict[i] = t

    seed = torch.initial_seed()
    opt.seed = seed

    global feat_maps
    feat_maps = [{'config': {
                'gamma':opt.gamma,
                'T':opt.T
                }} for _ in range(50)]

    def ddim_sampler_callback(pred_x0, xt, i):
        save_feature_maps_callback(i)
        save_feature_map(xt, 'z_enc', i)

    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                if block_idx in self_attn_output_block_indices:
                    q = block[1].transformer_blocks[0].attn1.q
                    k = block[1].transformer_blocks[0].attn1.k
                    v = block[1].transformer_blocks[0].attn1.v
                    save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", i)
                    save_feature_map(k, f"{feature_type}_{block_idx}_self_attn_k", i)
                    save_feature_map(v, f"{feature_type}_{block_idx}_self_attn_v", i)
            block_idx += 1

    def save_feature_maps_callback(i):
        save_feature_maps(unet_model.output_blocks , i, "output_block")

    def save_feature_map(feature_map, filename, time):
        global feat_maps
        cur_idx = idx_time_dict[time]
        feat_maps[cur_idx][f"{filename}"] = feature_map

    start_step = opt.start_step
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    uc = model.get_learned_conditioning([""])
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    sty_img_list = sorted(os.listdir(opt.sty))
    cnt_img_list = sorted(os.listdir(opt.cnt))

    begin = time.time()
    for sty_name in sty_img_list:
        sty_name_ = os.path.join(opt.sty, sty_name)
        init_sty = load_img(sty_name_).to(device)
        seed = -1
        sty_feat_name = os.path.join(feat_path_root, os.path.basename(sty_name).split('.')[0] + '_sty.pkl')
        sty_z_enc = None

        if len(feat_path_root) > 0 and os.path.isfile(sty_feat_name):
            print("Precomputed style feature loading: ", sty_feat_name)
            with open(sty_feat_name, 'rb') as h:
                sty_feat = pickle.load(h)
                sty_z_enc = torch.clone(sty_feat[0]['z_enc'])
        else:
            init_sty = model.get_first_stage_encoding(model.encode_first_stage(init_sty))
            sty_z_enc, _ = sampler.encode_ddim(init_sty.clone(), num_steps=ddim_inversion_steps, unconditional_conditioning=uc, \
                                                end_step=time_idx_dict[ddim_inversion_steps-1-start_step], \
                                                callback_ddim_timesteps=save_feature_timesteps,
                                                img_callback=ddim_sampler_callback)
            sty_feat = copy.deepcopy(feat_maps)
            sty_z_enc = feat_maps[0]['z_enc']

        for cnt_name in cnt_img_list:
            cnt_name_ = os.path.join(opt.cnt, cnt_name)
            init_cnt = load_img(cnt_name_).to(device)
            cnt_feat_name = os.path.join(feat_path_root, os.path.basename(cnt_name).split('.')[0] + '_cnt.pkl')
            cnt_feat = None

            if len(feat_path_root) > 0 and os.path.isfile(cnt_feat_name):
                print("Precomputed content feature loading: ", cnt_feat_name)
                with open(cnt_feat_name, 'rb') as h:
                    cnt_feat = pickle.load(h)
                    cnt_z_enc = torch.clone(cnt_feat[0]['z_enc'])
            else:
                init_cnt = model.get_first_stage_encoding(model.encode_first_stage(init_cnt))
                cnt_z_enc, _ = sampler.encode_ddim(init_cnt.clone(), num_steps=ddim_inversion_steps, unconditional_conditioning=uc, \
                                                    end_step=time_idx_dict[ddim_inversion_steps-1-start_step], \
                                                    callback_ddim_timesteps=save_feature_timesteps,
                                                    img_callback=ddim_sampler_callback)
                cnt_feat = copy.deepcopy(feat_maps)
                cnt_z_enc = feat_maps[0]['z_enc']

            with torch.no_grad():
                with precision_scope("cuda"):
                    with model.ema_scope():
                        output_name = f"{os.path.basename(cnt_name).split('.')[0]}_stylized_{os.path.basename(sty_name).split('.')[0]}.png"

                        print(f"Inversion end: {time.time() - begin}")
                        if opt.without_init_adain:
                            adain_z_enc = cnt_z_enc
                        else:
                            adain_z_enc = adain(cnt_z_enc, sty_z_enc, model=model)
                        feat_maps = feat_merge(opt, cnt_feat, sty_feat, start_step=start_step)
                        if opt.without_attn_injection:
                            feat_maps = None

                        samples_ddim, intermediates = sampler.sample(S=ddim_steps,
                                                        batch_size=1,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=adain_z_enc,
                                                        injected_features=feat_maps,
                                                        start_step=start_step,
                                                        )

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                        x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                        x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))

                        img.save(os.path.join(output_path, output_name))
                        if len(feat_path_root) > 0:
                            print("Save features")
                            if not os.path.isfile(cnt_feat_name):
                                with open(cnt_feat_name, 'wb') as h:
                                    pickle.dump(cnt_feat, h)
                            if not os.path.isfile(sty_feat_name):
                                with open(sty_feat_name, 'wb') as h:
                                    pickle.dump(sty_feat, h)

    print(f"Total end: {time.time() - begin}")

if __name__ == "__main__":
    main()