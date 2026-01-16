import math
import warnings

import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from scipy.sparse.linalg import eigsh
from torch.nn import init
from torch.optim import lr_scheduler
# from omp.src.main import run_omp  

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            # lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            lr_l = 0.3 ** max(0, epoch + opt.epoch_count - opt.n_epochs)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=()):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


def calc_mean_std_compress(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    N, C = size[:2]
    feat_var = feat.var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1)
    feat_mean = feat.mean(dim=2).view(N, C, 1)
    return feat_mean, feat_std


def mean_variance_norm_compress(feat):
    size = feat.size()
    mean, std = calc_mean_std_compress(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


class AdaAttN(nn.Module):

    def __init__(self, in_planes, max_sample=256 * 256, key_planes=None):
        super(AdaAttN, self).__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample

    def forward(self, content, style, content_key, style_key, seed=None):
        F = self.f(content_key)
        G = self.g(style_key)
        H = self.h(style)
        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        if w_g * h_g > self.max_sample:
            if seed is not None:
                torch.manual_seed(seed)
            index = torch.randperm(w_g * h_g).to(content.device)[:self.max_sample]
            G = G[:, :, index]
            style_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()
        else:
            style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        S = torch.bmm(F, G)
        # S: b, n_c, n_s
        S = self.sm(S)                                          # 注意力矩阵
        # print('注意力矩阵大小：', S.size())
        # print('风格特征大小：', style_flat.size())
        # mean: b, n_c, c
        mean = torch.bmm(S, style_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))

        # 压缩idea
        encode_start = torch.cuda.Event(enable_timing=True)
        encode_end = torch.cuda.Event(enable_timing=True)

        encode_start.record()  # 开始计时
        N_c, C_c, H_c, W_c = content.size()

        compress_ratio = 1
        group_nums = 256

        mean = mean.transpose(1, 2)
        std = std.transpose(1, 2)
        mean_compress, A_ = random_projection(mean, compress_ratio)
        std_compress, A_ = random_projection(std, compress_ratio, A_)
        # 先压缩再归一化
        content_compress, A_ = random_projection(content.view(N_c, -1, W_c*H_c), compress_ratio, A_)
        # 分组
        m_c = math.ceil(compress_ratio*H_c*W_c)
        print('1211111111111', m_c)
        sizes_content = [int(m_c / group_nums)] * (group_nums - 1)  # 前 n-1 份的大小
        sizes_content.append(m_c - (group_nums - 1) * int(m_c / group_nums))  # 最后一份的大小
        groups_content = torch.split(content_compress, sizes_content, dim=2)
        encode_end.record()  # 结束计时

        torch.cuda.synchronize()
        encode_time = encode_start.elapsed_time(encode_end)
        print(f"压缩分组模块耗时: {encode_time:.2f} ms")


        # encode_start = torch.cuda.Event(enable_timing=True)
        # encode_end = torch.cuda.Event(enable_timing=True)
        # encode_start.record()  # 开始计时
        group_mean = torch.split(mean_compress, sizes_content, dim=2)
        group_std = torch.split(std_compress, sizes_content, dim=2)
        zero_list = [torch.zeros_like(t) for t in groups_content]
        for i in range(group_nums):
            content_compress_norm = mean_variance_norm_compress(groups_content[i])
            zero_list[i] = content_compress_norm * group_std[i] + group_mean[i]
        result_compress = torch.cat(zero_list, dim=2)
        # encode_end.record()  # 结束计时
        # torch.cuda.synchronize()
        # encode_time = encode_start.elapsed_time(encode_end)
        # print(f"风格迁移耗时: {encode_time:.2f} ms")
        A_ = A_[0, :, :]
        
        encode_start = torch.cuda.Event(enable_timing=True)
        encode_end = torch.cuda.Event(enable_timing=True)
        encode_start.record()  # 开始计时
        result = ista(result_compress, A_, alpha=0.05, fast=True,
                                      lr='auto', maxiter=43, tol=1e-7, backtrack=False, eta_backtrack=1.5,
                                      verbose=False).reshape(N_c, C_c, H_c, W_c)
        # x_tensor = result_compress  # 已在GPU上的张量
        # dict_tensor = A_  # 已在GPU上的张量

        # # 调用GPU版本OMP（处理GPU张量）
        # result = run_omp(
        #     X=dict_tensor,            # GPU上的PyTorch张量
        #     y=x_tensor,               # GPU上的PyTorch张量
        #     n_nonzero_coefs=200,
        #     tol=1e-7,
        #     alg='v0'
        # )      


        encode_end.record()  # 结束计时
        torch.cuda.synchronize()
        encode_time = encode_start.elapsed_time(encode_end)
        print(f"重建耗时: {encode_time:.2f} ms")
        return result


class Transformer(nn.Module):

    def __init__(self, in_planes, key_planes=None, shallow_layer=False):
        super(Transformer, self).__init__()
        self.attn_adain_4_1 = AdaAttN(in_planes=in_planes, key_planes=key_planes)
        self.attn_adain_5_1 = AdaAttN(in_planes=in_planes,
                                        key_planes=key_planes + 512 if shallow_layer else key_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1,
                content4_1_key, style4_1_key, content5_1_key, style5_1_key, seed=None):
        return self.merge_conv(self.merge_conv_pad(
            self.attn_adain_4_1(content4_1, style4_1, content4_1_key, style4_1_key, seed=seed) +
            self.upsample5_1(self.attn_adain_5_1(content5_1, style5_1, content5_1_key, style5_1_key, seed=seed))))


class Decoder(nn.Module):

    def __init__(self, skip_connection_3=False):
        super(Decoder, self).__init__()
        self.decoder_layer_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.decoder_layer_2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256 + 256 if skip_connection_3 else 256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3))
        )

    def forward(self, cs, c_adain_3_feat=None):
        cs = self.decoder_layer_1(cs)
        if c_adain_3_feat is None:
            cs = self.decoder_layer_2(cs)
        else:
            cs = self.decoder_layer_2(torch.cat((cs, c_adain_3_feat), dim=1))
        return cs


def random_projection(feat, compress_ratio, A=None):

    N, C, HW = feat.size()
    # print('C,H,W',C,H,W)
    # 计算压缩后的长度
    m_feat = math.ceil(HW*compress_ratio)

    # 四种随机矩阵：
    # 1、随机高斯
    # A_content=torch.randn(m_c,content_h*content_w).cuda()
    # A_style = torch.randn(m_s, style_h * style_w).cuda()
    #
    # 2、随机高斯 大于0的设为1，小于0的设为-1
    # A_content = torch.randn(m_c, content_h * content_w).cuda()
    # A_content = torch.sign(A_content)
    # A_style = torch.randn(m_s, style_h * style_w).cuda()
    # A_style = torch.sign(A_style)

    # 3、随机 0，-1，1矩阵
    # A_content = torch.rand(m_c,content_h*content_w).cuda()
    # A_content[A_content < 1 / 3] = -1
    # A_content[A_content > 2 / 3] = 1
    # mask = (A_content > 1 / 3) & (A_content < 2 / 3)
    # A_content[mask] = 0
    # A_style = torch.rand(m_s, style_h * style_w).cuda()
    # A_style [A_style  < 1 / 3] = -1
    # A_style [A_style  > 2 / 3] = 1
    # mask = (A_style  > 1 / 3) & (A_style  < 2 / 3)
    # A_style[mask] = 0

    # 4、{0,1} 每列随机 num_1=sqrt(m)个元素为1，其余元素为0
    if A is None:
        A = torch.zeros(m_feat, HW).cuda()
        # print('11111')
        num_ones_content = int(round((m_feat) ** (1 / 2)))    # 压缩矩阵中1的个数
        for j in range(HW):
            indices = torch.randperm(m_feat)[:num_ones_content]
            A[indices, j] = 1

        A = A/((A ** 2).sum(dim=1, keepdim=True)).sqrt()  # 矩阵归一化
        A = A.unsqueeze(0).expand(N, -1, -1)              # 扩展batchsize
    if A is not None:
        A = A[0, :, :].unsqueeze(0).expand(N, -1, -1)
    feat_compress = torch.bmm(A, feat.permute(0, 2, 1)).permute(0, 2, 1)
    # print('压缩后的大小', feat_compress.size())

    return feat_compress, A


# 下面是FISTA的代码
def _lipschitz_constant(W):  # 函数计算Lipschitz常数，即矩阵的最大特征值
    # L = torch.linalg.norm(W, ord=2) ** 2
    WtW = torch.matmul(W.t(), W)
    # L = torch.linalg.eigvalsh(WtW)[-1]
    L = eigsh(WtW.detach().cpu().numpy(), k=1, which='LM',
              return_eigenvectors=False).item()
    return L

def backtracking(z, x, weight, alpha, lr0, eta=1.5, maxiter=1000, verbose=False):  # 回溯线搜索算法
    if eta <= 1:
        raise ValueError('eta must be > 1.')

    # store initial values
    resid_0 = torch.matmul(z, weight.T) - x  # 初始残差
    fval_0 = 0.5 * resid_0.pow(2).sum()  # 初始函数值
    fgrad_0 = torch.matmul(resid_0, weight)  # 初始梯度

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
        z_next = F.softshrink(z - lr * fgrad_0, alpha * lr)  # soft shrink软阈值化操作
        F_next = calc_F(z_next)  # 更新参数
        Q_next = calc_Q(z_next, lr)
        if verbose:
            print('iter: %4d,  t: %0.5f,  F-Q: %0.5f' % (i, lr, F_next - Q_next))
        if F_next <= Q_next:  # 若函数目标值小于等于函数近似值，终止迭代
            break
        lr = lr / eta  # 更新学习率
    else:
        warnings.warn('backtracking line search failed. Reverting to initial '
                      'step size')
        lr = lr0
        z_next = F.softshrink(z - lr * fgrad_0, alpha * lr)

    return z_next, lr

def initialize_code(x, weight):
    n_samples = x.size(0)
    n_components = weight.size(1)
    z0 = x.new_zeros(n_samples, n_components)
    return z0

def ista(x, weight, alpha=10.0, fast=True, lr='auto', maxiter=10,
         tol=1e-5, backtrack=False, eta_backtrack=1.5, verbose=False):
    z0 = initialize_code(x, weight)
    if lr == 'auto':
        # set lr based on the maximum eigenvalue of W^T @ W; i.e. the
        # Lipschitz constant of \grad f(z), where f(z) = ||Wz - x||^2
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

    # optimize
    z = z0
    if fast:
        y, t = z0, 1
    for _ in range(maxiter):
        if verbose:
            print('loss: %0.4f' % loss_fn(z))

        # ista update
        z_prev = y if fast else z
        # alpha = 0.5                               # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if backtrack:
            # perform backtracking line search
            z_next, _ = backtracking(z_prev, x, weight, alpha, lr, eta_backtrack)
        else:
            # constant step size
            z_next = F.softshrink(z_prev - lr * rss_grad(z_prev), alpha * lr)

        # check convergence
        if (z - z_next).abs().sum() <= tol:
            z = z_next
            break

        # update variables
        if fast:
            t_next = (1 + math.sqrt(1 + 4 * t ** 2)) / 2
            y = z_next + ((t - 1) / t_next) * (z_next - z)
            t = t_next
        z = z_next

    return z