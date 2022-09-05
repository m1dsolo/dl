import torch
import gzip
import os
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import bokeh
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, column
from bokeh.io import curdoc

device = 'cuda'

source_train_loss = ColumnDataSource(data=dict(x=[], y=[]))
source_train_acc = ColumnDataSource(data=dict(x=[], y=[]))
source_vali_loss = ColumnDataSource(data=dict(x=[], y=[]))
source_vali_acc = ColumnDataSource(data=dict(x=[], y=[]))

fig_loss = figure(toolbar_location='above', x_axis_label='epoch', y_axis_label='loss')
fig_acc = figure(toolbar_location='left', x_axis_label='epoch', y_axis_label='acc')
fig_loss.line(x='x', y='y', source=source_train_loss, legend_label='train_loss', color='blue', line_width=2)
fig_loss.line(x='x', y='y', source=source_vali_loss, legend_label='vali_loss', color='pink', line_width=2)
fig_acc.line(x='x', y='y', source=source_train_acc, legend_label='train_acc', color='blue', line_width=2)
fig_acc.line(x='x', y='y', source=source_vali_acc, legend_label='vali_acc', color='pink', line_width=2)

class Dataset_MNIST(Dataset):
    def __init__(self, is_train: bool):
        with gzip.open(os.path.join('dataset', 'x_train.gz' if is_train else 'x_test.gz'), 'rb') as f:
            x = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8, offset=16).float()
        with gzip.open(os.path.join('dataset', 'y_train.gz' if is_train else 'y_test.gz'), 'rb') as f:
            self.y = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8, offset=8)
        self.x = x.reshape((len(self.y), 1, 28, 28))

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)

train_iter = DataLoader(Dataset_MNIST(True), batch_size=1000, shuffle=True, num_workers=4)
test_iter = DataLoader(Dataset_MNIST(False), batch_size=1000, shuffle=False, num_workers=4)

def get_cnt(y_hat, y):
    return (y_hat.argmax(axis=1, keepdim=False) == y).sum()

def update_res(y_hat, y, res):
    for i in range(len(y)):
        # 注意int！如果是cuda上的tensor作为下标会有问题
        res[int(y[i]), int(y_hat[i].argmax())] += 1

def train(net, train_iter, num_train, num_epoch, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epoch):
        # train
        res_train = torch.zeros(3)
        net.train()
        for i, (x, y) in enumerate(train_iter):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            res_train += torch.tensor([l.sum(), get_cnt(y_hat, y), len(x)])
            if i + 1 == num_train:
                break

        # validation
        res_vali = torch.zeros(3)
        # ?
        net.eval()
        for x, y in train_iter:
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            l = loss(y_hat, y)
            res_vali += torch.tensor([l.sum(), get_cnt(y_hat, y), len(x)])

        train_loss = float(res_train[0] / res_train[2])
        train_acc = float(res_train[1] / res_train[2])
        vali_loss = float(res_vali[0] / res_vali[2])
        vali_acc = float(res_vali[1] / res_vali[2])
        print(f'{epoch + 1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, vali_loss={vali_loss:.4f}, vali_acc={vali_acc:.4f}')

        source_train_loss.stream({'x': [epoch], 'y': [train_loss]})
        source_train_acc.stream({'x': [epoch], 'y': [train_acc]})
        source_vali_loss.stream({'x': [epoch], 'y': [vali_loss]})
        source_vali_acc.stream({'x': [epoch], 'y': [vali_acc]})

def test(net, test_iter, device):
    net.eval()  # 开始评估模式（清除梯度，不调整权重）
    res = torch.zeros((10, 10), dtype=torch.int, device=device)
    for x, y in test_iter:
        x, y = x.to(device), y.to(device)
        update_res(net(x), y, res)
    return res

net1 = nn.Sequential(
    nn.Flatten(), 
    nn.Linear(28 * 28, 10)
    )

net2 = nn.Sequential(
    nn.Flatten(), 
    nn.Linear(28 * 28, 256), nn.ReLU(), 
    nn.Linear(256, 10)
    )

net3 = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(), 
    nn.AvgPool2d(kernel_size=2),    # stride默认等于kernel_size
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(), 
    nn.AvgPool2d(kernel_size=2), nn.Flatten(), 
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(), 
    nn.Linear(120, 84), nn.Sigmoid(), 
    nn.Linear(84, 10)
    )

def accuracy(res):
    return res.diag().sum() / res.sum()

def precision(res):
    return res.diag() / res.sum(dim=0)

def recall(res):
    return res.diag() / res.sum(dim=1)

def f_measure(res, beta):
    p, r = precision(res), recall(res)
    return (1 + beta * beta) * p * r / (beta * beta * p + r)

net = net3
# train(net, train_iter, num_train=int(5e4), num_epoch=50, lr=0.01, device=device)
# train(net, train_iter, num_train=int(5e4), num_epoch=50, lr=0.01, device=device)
train(net, train_iter, num_train=int(5e4), num_epoch=50, lr=1, device=device)
res = test(net, test_iter, device)

print(f'confusion matrix:\n{res}')
print(f'accuracy:{accuracy(res)}')
print(f'f_measure:\n{f_measure(res, 1)}')

show(column(fig_loss, fig_acc))
