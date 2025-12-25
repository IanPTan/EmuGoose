import torch as pt
from torch import nn


def vertical_flip(boards_encoding):
    n = boards_encoding.shape[0]
    reshaped_boards = boards_encoding.view(n, 12, 8, 8)
    flipped_spatial = pt.flip(reshaped_boards, dims=[2])
    
    flipped_final = pt.empty_like(flipped_spatial)
    flipped_final[:, 0:6, :, :] = flipped_spatial[:, 6:12, :, :]
    flipped_final[:, 6:12, :, :] = flipped_spatial[:, 0:6, :, :]

    return flipped_final.view(n, -1)


class NNUE(nn.Module):
    def __init__(self, layer_sizes=[512, 32, 32]):
        super().__init__()
        self.acm_linear = nn.Linear(768, layer_sizes[0] // 2)
        
        self.layers = nn.Sequential()
        layer_sizes.append(1)
        for i in range(len(layer_sizes) - 1):
            self.layers.add_module(f"relu_{i}", nn.ReLU())
            self.layers.add_module(f"linear_{i}", nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, sigmoid=True):
        acm1 = self.acm_linear(x)
        acm2 = self.acm_linear(vertical_flip(x))
        acm = pt.cat((acm1, acm2), dim=1)
        x = self.layers(acm)
        if sigmoid:
            x = self.sigmoid(x)
        return x

    def save(self, path):
        pt.save(self.state_dict(), path)

    def load(self, path):
        device = next(self.parameters()).device
        self.load_state_dict(pt.load(path, map_location=device))



