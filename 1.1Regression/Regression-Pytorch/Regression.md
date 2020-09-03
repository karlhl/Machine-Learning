# Regression

### 生成数据

首先生成数据x是从[-1,1]之前取100个点。然后进行扩充unsqueeze,从100维度，扩充到（100,1）

y是x平方+0.05的均方的噪声。原文是用的rand,是生成均匀分布的噪声，我觉得不合适，因为最后拟合出来的线肯定不是原函数了，所以我改成了randn，生成标准正态分布的噪声。

```python
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  
y = x.pow(2) + 0.05*torch.randn(x.size())        
```



<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20200903123402178.png" alt="image-20200903123402178" style="zoom:50%;" />

这个橙色是未加噪声的，蓝色点是加了噪声的。

### 构造模型

```python
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)     # define the network
print(net)  # net architecture
```

```
Net(
  (hidden): Linear(in_features=1, out_features=10, bias=True)
  (predict): Linear(in_features=10, out_features=1, bias=True)
)
```

模型结构是1，10, 1的结构。

### 训练

```python
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

plt.ion()   # something about plotting

for t in range(200):
    prediction = net(x)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 20 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
```

流程很标准。。。

### 结果

<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20200903124246123.png" alt="image-20200903124246123" style="zoom:50%;" />

<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20200903124254974.png" alt="image-20200903124254974" style="zoom:50%;" />

<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20200903124300314.png" alt="image-20200903124300314" style="zoom:50%;" />



<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20200903124313142.png" alt="image-20200903124313142" style="zoom:50%;" />

可以看到拟合出来的线越来越靠近真实函数了。但是感觉loss还不够低，左边还是直线。。











