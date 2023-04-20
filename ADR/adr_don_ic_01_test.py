import numpy as np
import torch
from adr_don_ic_01 import gp_sample, ADRNet
from adr_numerical_01 import plot

if "__main__" == __name__:
    torch.manual_seed(123)
    np.random.seed(123)

    test_ic_func = gp_sample(num=1)[0]

    func_x = np.linspace(0, 1, 100)
    u0_test = test_ic_func(func_x)
    u0_test = np.tile(u0_test, (10000, 1))
    u0_test = torch.from_numpy(u0_test).float()

    x = np.linspace(0, 1, 100)
    t = np.linspace(0, 1, 100)

    X, T = np.meshgrid(x, t)
    y_test = np.hstack([X.flatten()[:, None], T.flatten()[:, None]])
    y_test = torch.from_numpy(y_test).float()

    model = torch.load('adr_don_ic_01.pt')

    u_hat = model(u0_test, y_test).detach()

    plot(X, T, u_hat.reshape(100, -1))

    print()
