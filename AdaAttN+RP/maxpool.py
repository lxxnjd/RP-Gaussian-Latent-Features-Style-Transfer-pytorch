import torch

def max_pooling_non_overlapping(input_tensor):
    N, C, H, W = input_tensor.size()

    # 将输入张量展开成形状为(N * C, 1, H, W, 3, 3)的张量
    unfolded = input_tensor.unfold(2, 3, 3).unfold(3, 3, 3).reshape(N, C, 1, -1, 3, 3)

    # 在最后两个维度上取最大值
    max_val, _ = unfolded.max(dim=(4, 5), keepdim=True)

    # 将结果重塑回原始形状
    output_tensor = max_val.view(N, C, H, W)

    return output_tensor

# 示例用法
input_tensor = torch.randn(2, 3, 6, 6)  # 示例输入张量大小为[2, 3, 6, 6]
output_tensor = max_pooling_non_overlapping(input_tensor)
print(output_tensor)

A = input_tensor.numpy()
B = output_tensor.numpy()
print(output_tensor.size())