import torch
import math
import numpy as np
import torch.nn.functional as F
from scipy.sparse.linalg import eigsh
import warnings
import matplotlib.pyplot as plt
import time
import os
import sys




# Define projection and grouping function
def sample_split(content_feat, style_feat, compress_ratio, group_nums):
    # Vectorization
    content_feat = content_feat.squeeze(0)
    style_feat = style_feat.squeeze(0)
    content_c, content_h, content_w = content_feat.shape
    # print('content_c, content_h, content_w',content_c, content_h, content_w)
    style_c, style_h, style_w = style_feat.shape
    content_feat_vector = content_feat.view(content_c, content_h*content_w)
    style_feat_vector = style_feat.view(style_c, style_h*style_w)

    # plt.hist(content_feat_vector.cpu().numpy(), bins=30, edgecolor='black')
    # plt.xlabel('Value')  # x-axis label
    # plt.ylabel('Frequency')  # y-axis label
    # plt.title('Distribution Histogram')  # Title
    # plt.grid(axis='y', alpha=0.75)  # Add grid lines
    # plt.show()  # Display plot

    # Calculate length after projection
    m_c = int(compress_ratio*content_h*content_w)
    m_s = int(compress_ratio*style_h*style_w)

# Four types of random matrices:

    # 1. Random Gaussian
    A_content=torch.randn(m_c,content_h*content_w).cuda()
    A_style = torch.randn(m_s, style_h * style_w).cuda()
    #
    # 2. Random Gaussian (set values >0 to 1, <0 to -1)
    # A_content = torch.randn(m_c, content_h * content_w).cuda()
    # A_content = torch.sign(A_content)
    # A_style = torch.randn(m_s, style_h * style_w).cuda()
    # A_style = torch.sign(A_style)

    # 3. Random {0, -1, 1} matrix
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

    # 4. {0,1} matrix (randomly set num_1=sqrt(m) elements to 1 per column, others to 0)
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

    # Matrix normalization and random projection
    compress_start = torch.cuda.Event(enable_timing=True)
    compress_end = torch.cuda.Event(enable_timing=True)
    compress_start.record()  # Start timing
    

    A_content = A_content/((A_content ** 2).sum(dim=1, keepdim=True)).sqrt()
    A_style = A_style/((A_style ** 2).sum(dim=1, keepdim=True)).sqrt()
    y_content = torch.matmul(A_content, content_feat_vector.T).T.cuda()
    y_style = torch.matmul(A_style, style_feat_vector.T).T.cuda()
    
    compress_end.record()  # End timing
    torch.cuda.synchronize()
    compress_time = compress_start.elapsed_time(compress_end)
    print(f"Compression module time consumption: {compress_time:.2f} ms")
    print('1111111111111111', y_style.size())

    # Grouping
    print('Size before grouping:', y_content.size())
    sizes_content = [int(m_c / group_nums)]*(group_nums - 1)  # Size of first n-1 groups
    sizes_content.append(m_c - (group_nums-1)*int(m_c/group_nums))  # Size of last group
    groups_content = torch.split(y_content, sizes_content, dim=1)
    sizes_style = [int(m_s / group_nums)]*(group_nums - 1)
    sizes_style.append(m_s-(group_nums-1)*int(m_s/group_nums))
    groups_style = torch.split(y_style, sizes_style, dim=1)

    return groups_content, groups_style, A_content, content_c, content_h, content_w


# Rewrite function for calculating mean and standard deviation
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
    # Introduce noise after projection to enhance diversity
    # normalized_feat = (content_feat - content_mean.expand(
    #     #.expand() makes content_mean the same size as content_feature
    #    size)+torch.normal(0, content_mean/10)) / content_std.expand(size)
    normalized_feat = (content_feat - content_mean.expand(
        # .expand() makes content_mean the same size as content_feature
        size)) / content_std.expand(size)

    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


# Grouped AdaIN (loop by group)
def cs_AdaIN(groups_content,groups_style, group_nums):
    # Define empty list for storage
    zero_list = [torch.zeros_like(t) for t in groups_content]
    for i in range(group_nums):
        zero_list[i] = adaptive_instance_normalization(groups_content[i], groups_style[i])
    # Merge each group
    target = torch.cat(zero_list, dim=1)
    return target


# Define function for the entire process
def random_projection_style_transfer(content_feat, style_feat, compress_ratio, group_nums):
    # Projection and grouping
    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    # Calculate time consumption of style transfer module
    proj_start = torch.cuda.Event(enable_timing=True)
    proj_end = torch.cuda.Event(enable_timing=True)
    proj_start.record()  # Start timing
    groups_content, groups_style, A_content, content_c, content_h, content_w = sample_split(content_feat, style_feat, compress_ratio, group_nums)
    proj_end.record()  # End timing
    torch.cuda.synchronize()  # Synchronize GPU to ensure accurate timing
    proj_time = proj_start.elapsed_time(proj_end)  # Unit: milliseconds
    print(f"Feature projection and grouping module time consumption: {proj_time:.2f} ms")
    
    if use_cuda:
        proj_start = torch.cuda.Event(enable_timing=True)
        proj_end = torch.cuda.Event(enable_timing=True)
        proj_start.record()
    else:
        proj_start = time.time()

    feat_transfer = cs_AdaIN(groups_content, groups_style, group_nums)
    print(feat_transfer.size())

    if use_cuda:
        proj_end.record()
        torch.cuda.synchronize()
        proj_time = proj_start.elapsed_time(proj_end)
    else:
        proj_time = (time.time() - proj_start) * 1000
    print(f" [Style transfer] Time consumption: {proj_time:.2f} ms")
    
    print('111',A_content.size())
    # Sparse reconstruction
    content_reconstruction = ista(torch.as_tensor(feat_transfer, device='cuda', dtype=torch.float),
                      torch.as_tensor(A_content, device='cuda', dtype=torch.float), alpha=0.5, fast=True,
                      lr='auto', maxiter=50, tol=1e-7, backtrack=False, eta_backtrack=1.5, verbose=True)
    # content_reconstruction = standard_omp(torch.as_tensor(feat_transfer, device='cuda', dtype=torch.float),
    #                   torch.as_tensor(A_content, device='cuda', dtype=torch.float),
    #                    max_iter=10, tol=1e-7, verbose=True)
    feat = content_reconstruction.reshape(content_c, content_h, content_w).unsqueeze(0)
    # mean = feat.mean().cuda()
    # std_dev = mean / 5
    # noise = torch.normal(mean=0, std=std_dev, size=feat.size()).cuda()
    # feat= feat+ noise
    return feat


# Below is the FISTA code
def _lipschitz_constant(W):                  # Function to calculate Lipschitz constant (maximum eigenvalue of the matrix)
    #L = torch.linalg.norm(W, ord=2) ** 2
    WtW = torch.matmul(W.t(), W)
    #L = torch.linalg.eigvalsh(WtW)[-1]
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



import torch

def standard_omp(x, dictionary, max_iter=10, tol=1e-6, verbose=False):
    """
    Implementation of Orthogonal Matching Pursuit (OMP) algorithm.

    Parameters:
        x (torch.Tensor): Observation signal matrix with shape (n_samples, n_measurements).
                          Each row vector represents a sample to be sparsely encoded.
        dictionary (torch.Tensor): Overcomplete dictionary matrix with shape (n_measurements, n_components).
                                   Each column vector represents a dictionary atom.
        max_iter (int): Maximum number of iterations (maximum number of atoms allowed to be selected).
        tol (float): Residual convergence threshold. The algorithm stops early when the L2 norm of the residual 
                     is less than this value.
        verbose (bool): Whether to print iteration process information.

    Returns:
        torch.Tensor: Sparse coefficient matrix with shape (n_samples, n_components).
                      Each row vector is the sparse representation of the corresponding input sample.
    """
    n_samples, n_measurements = x.shape
    n_components = dictionary.shape[1]

    # Initialization
    device = x.device
    dtype = x.dtype

    # Sparse coefficient matrix, initialized to all zeros
    z = torch.zeros(n_samples, n_components, device=device, dtype=dtype)
    # Residual matrix, initialized to input signal
    residuals = x.clone()
    # Record selected atom indices for each sample
    selected_atoms = [[] for _ in range(n_samples)]
    
    # Calculate L2 norm of dictionary for normalized inner product calculation to make correlation comparison fairer
    dict_norms = torch.norm(dictionary, p=2, dim=0, keepdim=True).t() # Shape: (n_components, 1)
    # Avoid division by zero
    dict_norms[dict_norms < 1e-10] = 1e-10

    for k in range(max_iter):
        # 1. Calculate correlation (inner product) between residual and all atoms, then normalize
        # Inner product shape: (n_samples, n_components)
        inner_products = torch.matmul(residuals, dictionary)
        # Normalize inner product to eliminate the influence of atom and residual energy
        residual_norms = torch.norm(residuals, p=2, dim=1, keepdim=True) # Shape: (n_samples, 1)
        normalized_correlations = inner_products / (residual_norms * dict_norms.t() + 1e-10)
        
        # 2. Select the atom with maximum correlation for each sample
        selected_idx = torch.argmax(torch.abs(normalized_correlations), dim=1) # Shape: (n_samples,)

        # 3. Iteratively update coefficients and residuals for each sample
        for i in range(n_samples):
            idx = selected_idx[i].item()
            
            # Skip if the atom is already selected to avoid duplication
            if idx in selected_atoms[i]:
                continue
            
            # Add new atom to selected list
            selected_atoms[i].append(idx)
            # Extract all selected atoms of current sample to form sub-dictionary
            # Note: Need to use .to(device) to ensure it's on the same device
            sub_dictionary = dictionary[:, selected_atoms[i]].to(device)
            
            # 4. Orthogonalize the selected sub-dictionary (Gram-Schmidt orthogonalization)
            # This is the core difference between standard OMP and greedy MP
            orthogonal_sub_dict = torch.zeros_like(sub_dictionary, device=device, dtype=dtype)
            for j in range(orthogonal_sub_dict.shape[1]):
                atom = sub_dictionary[:, j:j+1]
                # Subtract the projection of this atom on previous orthogonal atoms
                for p in range(j):
                    proj = torch.matmul(orthogonal_sub_dict[:, p:p+1].t(), atom)
                    atom = atom - proj * orthogonal_sub_dict[:, p:p+1]
                # Normalize the orthogonalized atom
                norm = torch.norm(atom)
                if norm > 1e-10:
                    atom = atom / norm
                orthogonal_sub_dict[:, j:j+1] = atom

            # 5. Solve optimal coefficients by least squares
            # Objective: Find z_i to minimize ||x_i - orthogonal_sub_dict * z_i||^2
            # Solution: z_i = (orthogonal_sub_dict^T * orthogonal_sub_dict)^{-1} * orthogonal_sub_dict^T * x_i
            # Due to orthogonality, orthogonal_sub_dict^T * orthogonal_sub_dict is an identity matrix, so the solution simplifies to inner product
            xi = x[i:i+1, :].t() # Transpose to column vector (n_measurements, 1)
            coeffs = torch.matmul(orthogonal_sub_dict.t(), xi)
            
            # 6. Update sparse coefficients
            z[i, selected_atoms[i]] = coeffs.squeeze()
            
            # 7. Update residual
            residuals[i:i+1, :] = x[i:i+1, :] - torch.matmul(z[i:i+1, :], dictionary.t())

        # Calculate average residual norm to monitor convergence
        avg_residual_norm = torch.mean(torch.norm(residuals, p=2, dim=1))
        
        if verbose:
            print(f"OMP Iteration {k+1}/{max_iter}, Average Residual Norm: {avg_residual_norm:.6f}")
        
        # Check if residuals of all samples have converged
        if avg_residual_norm < tol:
            if verbose:
                print(f"OMP converged early at iteration {k+1} with tolerance {tol}.")
            break

    return z