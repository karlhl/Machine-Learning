import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn


n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1) # 均值为2方差为1的（100,2）
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1) # 均值为-2，方差为1的（100,2）
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor) # （200,2）
y = torch.cat((y0, y1), ).type(torch.LongTensor)

print(x.shape, y.shape)

plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn') # 前两个分别是x，y坐标，c是颜色，正好分为0,1
plt.show()

class Net(nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = nn.Linear(n_feature,n_hidden)
        self.out = nn.Linear(n_hidden,n_output)
    def forward(self,x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.out(x)
        return x
net = Net(2,10,2)
print(net)

optimizer = torch.optim.Adam(net.parameters(),lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()
for t in range(100):
    out = net(x)
    loss = loss_func(out,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    accuracy = 0
    if t%2==0 and t<30:
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.3f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()