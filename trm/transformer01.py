import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = x + self.pe[:x.size(0), :]
        x = x + self.pe
        return self.dropout(x)


class MyTransformer(nn.Module):
    def __init__(self, seq_max_len, input_dim, output_dim, y_train_ic, d_model=2, n_heads=1, n_layers=2):
        super(MyTransformer, self).__init__()

        self.y_train_ic = y_train_ic
        self.loss_func = nn.MSELoss(reduction='mean')

        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=0.1, max_len=seq_max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
                                                   activation=nn.Tanh())
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
                                                   activation=nn.Tanh())
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        memory = self.encoder(x)
        tgt = memory.clone()
        start_point = torch.zeros(tgt.shape[1]).unsqueeze(0)
        shifted_tgt = torch.cat([start_point, tgt[:-1]])
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(tgt))
        x = self.decoder(tgt=shifted_tgt, memory=memory, tgt_mask=tgt_mask)
        x = self.fc(x)
        return x

    def loss(self, x_pde):
        x_pde.requires_grad = True
        y_hat = self.forward(x_pde)
        u_t = autograd.grad(y_hat, x_pde, torch.ones_like(y_hat), create_graph=True)[0]
        loss_pde = self.loss_func(u_t, torch.cos(x_pde))
        loss_ic = self.loss_func(self.y_train_ic, y_hat[0])
        return loss_pde + loss_ic


if "__main__" == __name__:
    # torch.manual_seed(1234)
    # np.random.seed(1234)
    torch.set_default_dtype(torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # 区间总数
    total_interval = 2
    # 总点数
    total_points = total_interval + 1
    # 区间长度
    h = 0.005
    x_lb = torch.tensor(0.)
    x_ub = torch.tensor(total_interval * h)

    y_train_ic = torch.tensor([0.])

    TRM = MyTransformer(seq_max_len=total_points, input_dim=1, output_dim=1, y_train_ic=y_train_ic)
    TRM.to(device)
    print(TRM)

    x_test = torch.linspace(x_lb, x_ub, total_points)

    # 配置点
    x_train_nf = x_test.unsqueeze(1)
    x_train_nf = x_train_nf.float().to(device)

    optimizer = torch.optim.Adam(TRM.parameters(), lr=1e-3, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, mode='min', factor=0.5,
                                                           patience=40000,
                                                           verbose=True)
    epoch = 0
    while True:
        epoch += 1
        loss = TRM.loss(x_train_nf)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if epoch % 100 == 0:
            print('epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())

        if loss.item() < 0.001:
            print('epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
            break

    TRM.cpu()
    nn_predict = TRM(x_test[:, None]).detach().numpy()
    # x_nn = nn_predict[:, [0]]
    # y_nn = nn_predict[:, [1]]
    # z_nn = nn_predict[:, [2]]

    fig, ax = plt.subplots()
    ax.plot(x_test, nn_predict, color='r', label='x')
    # ax.plot(x_test, y_nn, color='g', label='y')
    # ax.plot(x_test, z_nn, color='b', label='z')
    ax.set_title('$sin x$')
    ax.set_xlabel('t', color='black')
    ax.set_ylabel('f(t)', color='black', rotation=0)
    ax.legend(loc='upper right')
    plt.savefig('./images/transformer01.png')
    plt.show()
