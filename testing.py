import glob
import os
import torch

if __name__ == '__main__':
    img = torch.ones((1, 3, 256, 256)) * 2

    mask = torch.ones((1, 5, 256, 256))
    _, c, _, _ = mask.shape

    outs = []
    for i in range(c):
        mk = mask[:, i, :, :]

        out = torch.matmul(img, mk)
        outs.append(out)

    outs = torch.cat(outs, 1)
    print(outs.shape)
