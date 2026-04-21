'''
sine + cosine example
'''

import torch

t1 = torch.randn(10, 10, device="cuda")
t2 = torch.randn(10, 10, device="cuda")

@torch.compile
def opt_foo2(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b

print(opt_foo2(t1, t2))