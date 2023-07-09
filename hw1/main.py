from mlp import MLP
import torch

if __name__ == '__main__':
    net = MLP(2, 10, torch.relu, 10, 2, torch.relu)
    x = torch.randn(1,2)
    y_pred = net.forward(x)
    DJdy_hat = torch.randn(1,2)
    net.backward(DJdy_hat)
    