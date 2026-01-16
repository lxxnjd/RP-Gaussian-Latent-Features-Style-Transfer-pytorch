import torch

# 加载预训练权重
pretrained_dict = torch.load('checkpoints/AdaAttN/latest_net_adaattn_3.pth')

# 创建一个新的字典来存储调整后的键名和对应的值
new_pretrained_dict = {}

# 遍历预训练权重的键值对，添加"module."前缀后存储到新的字典中
for key, value in pretrained_dict.items():
    print('改之前', key)
    new_key = "module." + key  # 在键名前面添加"module."
    new_pretrained_dict[new_key] = value
    print('改之后', new_key)

torch.save(new_pretrained_dict, 'checkpoints/AdaAttN/latest_net_adaattn_3.pth_adjusted.pth')