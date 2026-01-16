import torch
from skimage.exposure import match_histograms
import numpy as np
import math
import numpy as np
import torch.nn.functional as F
from scipy.sparse.linalg import eigsh
import warnings


def sample_split(content_feat, style_feat, compress_ratio, group_nums):
    content_feat = content_feat.squeeze(0)
    style_feat = style_feat.squeeze(0)
    content_c, content_h, content_w = content_feat.shape
    style_c, style_h, style_w = style_feat.shape
    content_feat_vector = content_feat.view(content_c, content_h*content_w)
    style_feat_vector = style_feat.view(style_c, style_h*style_w)

    m_c = int(compress_ratio * content_h * content_w)
    m_s = int(compress_ratio * style_h * style_w)

    ### Random Gaussian matrix
    A_content = torch.randn(m_c, content_h*content_w).cuda()
    A_style = torch.randn(m_s, style_h * style_w).cuda()
    
    ### Random Gaussian (set values >0 to 1, <0 to -1)
    # A_content = torch.randn(m_c, content_h * content_w).cuda()
    # A_content = torch.sign(A_content)
    # A_style = torch.randn(m_s, style_h * style_w).cuda()
    # A_style = torch.sign(A_style)

    ### Random {0, -1, 1} matrix
    # A_content = torch.rand(m_c, content_h*content_w).cuda()
    # A_content[A_content < 1 / 3] = -1
    # A_content[A_content > 2 / 3] = 1
    # mask = (A_content > 1 / 3) & (A_content < 2 / 3)
    # A_content[mask] = 0
    # A_style = torch.rand(m_s, style_h * style_w).cuda()
    # A_style [A_style  < 1 / 3] = -1
    # A_style [A_style  > 2 / 3] = 1
    # mask = (A_style  > 1 / 3) & (A_style  < 2 / 3)
    # A_style[mask] = 0

    ### {0,1} matrix (randomly set num_1=sqrt(m) elements to 1 per column, others to 0)
    # A_content = torch.zeros(m_c, content_h * content_w).cuda()
    # num_ones_content = int(round((m_c) ** (1 / 2)))
    # for i in range(content_h * content_w):
    #     indices = torch.randperm(m_c)[:num_ones_content]
    #     A_content[indices,i] = 1
    # A_style = torch.zeros(m_s, style_h * style_w).cuda()
    # num_ones_style = int(round((m_s) ** (1 / 2)))
    # for i in range(style_h * style_w):
    #     indices = torch.randperm(m_s)[:num_ones_style]
    #     A_style[indices,i] = 1


    # Introduce noise into random matrix
    # non_zero_elements_content = A_content[A_content != 0]
    # indices_content = torch.nonzero(A_content, as_tuple=True)
    # noise_content = torch.normal(mean=0, std=1, size=non_zero_elements_content.size()).cuda()
    # noise_matrix_content = torch.zeros_like(A_content).cuda()
    # noise_matrix_content[A_content != 0] = noise_content
    # A_content = A_content + noise_matrix_content
    #
    # non_zero_elements_style = A_style[A_style != 0]
    # indices_style = torch.nonzero(A_style, as_tuple=True)
    # noise_style = torch.normal(mean=0, std=1, size=non_zero_elements_style.size()).cuda()
    # noise_matrix_style = torch.zeros_like(A_style).cuda()
    # noise_matrix_style[A_style != 0] = noise_style
    # A_style = A_style + noise_matrix_style


    A_content = A_content / ((A_content ** 2).sum(dim=1, keepdim=True)).sqrt()
    A_style = A_style / ((A_style ** 2).sum(dim=1, keepdim=True)).sqrt()
    y_content = torch.matmul(A_content, content_feat_vector.T).T.cuda()
    y_style = torch.matmul(A_style, style_feat_vector.T).T.cuda()

    sizes_content = [int(m_c / group_nums)] * (group_nums - 1)  # Size of first n-1 groups
    sizes_content.append(m_c - (group_nums-1)*int(m_c/group_nums))  # Size of the last group
    groups_content = torch.split(y_content, sizes_content, dim=1)
    sizes_style = [int(m_s / group_nums)] * (group_nums - 1)
    sizes_style.append(m_s - (group_nums-1)*int(m_s/group_nums))
    groups_style = torch.split(y_style, sizes_style, dim=1)
    return groups_content, groups_style, A_content, content_c, content_h, content_w

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()              # size = [C,H*W]
    c = size[0]
    feat_var = feat.contiguous().var(dim=1) + eps
    # make [N,C,H,W] into [N,C,H*W] ; compute var along channel H*W
    # This returns a tensor of shape (N, C), where each element represents the variance of the corresponding channel.
    feat_std = feat_var.sqrt().contiguous().view(c, 1)
    # using feat_var to calculate std ; make [C] into [C,1]
    # The standard deviation tensor has the same batch size and number of channels as the original feature tensor, but both the height and width become 1.
    feat_mean = feat.contiguous().mean(dim=1).contiguous().view(c, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)  # compute style_mean, style_std
    content_mean, content_std = calc_mean_std(content_feat)  # compute content_mean, content_std
    normalized_feat = (content_feat - content_mean.expand(
        # .expand() makes content_mean the same size as content_feature
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

# def adaptive_instance_normalization(content_feat, style_feat):
#     assert (content_feat.size()[:2] == style_feat.size()[:2])
#     size = content_feat.size()
#     style_mean, style_std = calc_mean_std(style_feat)
#     content_mean, content_std = calc_mean_std(content_feat)
#     normalized_feat = (content_feat - content_mean.expand(
#         size)) / content_std.expand(size)
#     return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def exact_feature_distribution_matching(content_feat, style_feat):
    assert (content_feat.size() == style_feat.size())
    C, L = content_feat.size(0), content_feat.size(1)
    value_content, index_content = torch.sort(content_feat.view(C,-1))  # sort conduct a deep copy here.
    value_style, _ = torch.sort(style_feat.view(C,-1))  # sort conduct a deep copy here.
    inverse_index = index_content.argsort(-1)
    new_content = content_feat.view(C,-1) + (value_style.gather(-1, inverse_index) - content_feat.view(C,-1).detach())
    return new_content.view(C,L)

def cs_AdaIN(groups_content, groups_style, group_nums):
    zero_list = [torch.zeros_like(t) for t in groups_content]
    for i in range(group_nums):
        zero_list[i] = adaptive_instance_normalization(groups_content[i], groups_style[i])
    target = torch.cat(zero_list, dim=1)
    return target

def cs_EFDM(groups_content, groups_style, group_nums):
    zero_list = [torch.zeros_like(t) for t in groups_content]
    for i in range(group_nums):
        zero_list[i] = exact_feature_distribution_matching(groups_content[i], groups_style[i])
    target = torch.cat(zero_list, dim=1)
    return target

def random_projection_style_transfer(content_feat, style_feat, compress_ratio, group_nums):
    # Time consumption for grouped projection   
    encode_start = torch.cuda.Event(enable_timing=True)
    encode_end = torch.cuda.Event(enable_timing=True)
    encode_start.record()  # Start timing
    groups_content, groups_style, A_content, content_c, content_h, content_w = sample_split(content_feat, style_feat, compress_ratio, group_nums)
    encode_end.record()  # End timing
    torch.cuda.synchronize()
    encode_time = encode_start.elapsed_time(encode_end)
    print(f"Grouped projection time consumption: {encode_time:.2f} ms")

    # Time consumption for style transfer
    encode_start = torch.cuda.Event(enable_timing=True)
    encode_end = torch.cuda.Event(enable_timing=True)
    encode_start.record()  # Start timing
    feat_transfer = cs_EFDM(groups_content, groups_style, group_nums)
    encode_end.record()  # End timing
    torch.cuda.synchronize()
    encode_time = encode_start.elapsed_time(encode_end)
    print(f"Style transfer time consumption: {encode_time:.2f} ms")

    # Time consumption for reconstruction
    encode_start = torch.cuda.Event(enable_timing=True)
    encode_end = torch.cuda.Event(enable_timing=True)
    encode_start.record()  # Start timing
    content_reconstruction = ista(torch.as_tensor(feat_transfer, device='cuda', dtype=torch.float),
                      torch.as_tensor(A_content, device='cuda', dtype=torch.float), alpha=0.5 * 1e-1, fast=True,
                      lr='auto', maxiter=50, tol=1e-7, backtrack=False, eta_backtrack=1.5, verbose=False)

    feat = content_reconstruction.reshape(content_c, content_h, content_w).unsqueeze(0)
    encode_end.record()  # End timing
    torch.cuda.synchronize()
    encode_time = encode_start.elapsed_time(encode_end)
    print(f"Reconstruction time consumption: {encode_time:.2f} ms")
    
    # mean = feat.mean().cuda()
    # std_dev = mean / 5
    # noise = torch.normal(mean=0, std=std_dev, size=feat.size()).cuda()
    # feat = feat + noise
    return feat


def _lipschitz_constant(W):                  # Function to calculate Lipschitz constant (maximum eigenvalue of the matrix)
    # L = torch.linalg.norm(W, ord=2) ** 2
    WtW = torch.matmul(W.t(), W)
    # L = torch.linalg.eigvalsh(WtW)[-1]
    L = eigsh(WtW.detach().cpu().numpy(), k=1, which='LM',
              return_eigenvectors=False).item()
    return L


def backtracking(z, x, weight, alpha, lr0, eta=1.5, maxiter=1000, verbose=False):   # Backtracking line search algorithm
    if eta <= 1:
        raise ValueError('eta must be > 1.')

    # store initial values
    resid_0 = torch.matmul(z, weight.T) - x   # Initial residual
    fval_0 = 0.5 * resid_0.pow(2).sum()       # Initial function value
    fgrad_0 = torch.matmul(resid_0, weight)   # Initial gradient

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
        z_next = F.softshrink(z - lr * fgrad_0, alpha * lr)    # Soft shrinkage (soft thresholding operation)
        F_next = calc_F(z_next)                                # Update parameters
        Q_next = calc_Q(z_next, lr)
        if verbose:
            print('iter: %4d,  t: %0.5f,  F-Q: %0.5f' % (i, lr, F_next-Q_next))
        if F_next <= Q_next:           # Terminate iteration if objective function value ≤ approximate function value
            break
        lr = lr / eta            # Update learning rate
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
            t_next = (1 + math.sqrt(1 + 4 * t**2)) / 2
            y = z_next + ((t-1)/t_next) * (z_next - z)
            t = t_next
        z = z_next

    return z


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


# def adaptive_instance_normalization(content_feat, style_feat):
#     assert (content_feat.size()[:2] == style_feat.size()[:2])
#     size = content_feat.size()
#     style_mean, style_std = calc_mean_std(style_feat)
#     content_mean, content_std = calc_mean_std(content_feat)
#
#     normalized_feat = (content_feat - content_mean.expand(
#         size)) / content_std.expand(size)
#     return normalized_feat * style_std.expand(size) + style_mean.expand(size)

# def exact_feature_distribution_matching(content_feat, style_feat):
#     assert (content_feat.size() == style_feat.size())
#     B, C, W, H = content_feat.size(0), content_feat.size(1), content_feat.size(2), content_feat.size(3)
#     value_content, index_content = torch.sort(content_feat.view(B,C,-1))  # sort conduct a deep copy here.
#     value_style, _ = torch.sort(style_feat.view(B,C,-1))  # sort conduct a deep copy here.
#     inverse_index = index_content.argsort(-1)
#     new_content = content_feat.view(B,C,-1) + (value_style.gather(-1, inverse_index) - content_feat.view(B,C,-1).detach())
#
#     return new_content.view(B, C, W, H)
#
# ## HM
# def histogram_matching(content_feat, style_feat):
#     assert (content_feat.size() == style_feat.size())
#     B, C, W, H = content_feat.size(0), content_feat.size(1), content_feat.size(2), content_feat.size(3)
#     x_view = content_feat.view(-1, W,H)
#     image1_temp = match_histograms(np.array(x_view.detach().clone().cpu().float().transpose(0, 2)),
#                                    np.array(style_feat.view(-1, W, H).detach().clone().cpu().float().transpose(0, 2)),
#                                    multichannel=True)
#     image1_temp = torch.from_numpy(image1_temp).float().to(content_feat.device).transpose(0, 2).view(B, C, W, H)
#     return content_feat + (image1_temp - content_feat).detach()



def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
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
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

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