'''
softmax example
'''
import torch

@torch.compile
def softmax(x):
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(x)
    denominator = numerator.sum(dim=1)
    return numerator / denominator[:, None]

t1 = torch.randn(10, 10, device="cuda")
print(softmax(t1))