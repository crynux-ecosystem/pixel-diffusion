import torch
from utils import platform

DEVICE = platform.get_accelerator()
STEPS = 500

def beta(t):
    L = torch.tensor(0.001, dtype=torch.float)
    H = torch.tensor(0.018, dtype=torch.float)
    return ((H - L) * t / STEPS + L).float()

def _alpha():
    at = torch.zeros(STEPS+1, dtype=torch.float).to(DEVICE)
    t = 1
    for i in range(STEPS+1):
        t *= 1 - beta(i)
        at[i] = t
    return at

ALPHA = _alpha()


# Returns (actual noise, noised images)
def noise(xs, ts):
    assert xs.shape[0] == ts.shape[0], f"Number of images ({xs.shape[0]}) must match number of timestamps ({ts.shape[0]})"
    assert ts.dtype == torch.long, "Times must have long datatype" # required for indexing

    device = platform.get_accelerator()
    alpha = ALPHA[ts]
    epsilon = torch.normal(torch.zeros_like(xs), 1).to(DEVICE)

    noised = torch.sqrt(alpha)[:,None,None,None] * xs + torch.sqrt(1 - alpha)[:,None,None,None] * epsilon
    return epsilon, noised

