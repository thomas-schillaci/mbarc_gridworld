import torch

def one_hot_encode(x, n):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    res = x.long()
    res = res.view((-1, 1))
    res = torch.zeros((len(res), n)).to(res).scatter(1, res, 1)
    res = res.view((*x.shape, n))
    res = res.type(x.dtype)
    return res


def standardize_frame(x):
    x_mean = torch.mean(x, dim=(-1, -2)).view((-1, 1, 1))
    x_var = torch.var(x, dim=(-1, -2)).view((-1, 1, 1))
    num_pixels = torch.tensor(x.shape[-1] * x.shape[-2], dtype=torch.float32).to(x.device)
    return (x - x_mean) / torch.max(torch.sqrt(x_var), torch.rsqrt(num_pixels))