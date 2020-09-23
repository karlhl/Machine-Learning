# Classify

建立数据

```python
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1) # 均值为2方差为1的（100,2）
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1) # 均值为-2，方差为1的（100,2）
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor) # （200,2）
y = torch.cat((y0, y1), ).type(torch.LongTensor)
```

x最后是（200,2）的数据，其中前100行是均值为2，方差为1。后一百行均值为-1，方差为1，数据是两列是想同时作为横坐标纵坐标。

y前100是0后100是1，同时作为颜色显示以及label



绘制数据

```python
plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn') # 前两个分别是x，y坐标，c是颜色，正好分为0,1
plt.show()
```

![image-20200923164321496](https://gitee.com/karlhan/picgo/raw/master/img//image-20200923164321496.png)

建立模型

两层线性。

```python
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
"""
Net(
  (hidden): Linear(in_features=2, out_features=10, bias=True)
  (out): Linear(in_features=10, out_features=2, bias=True)
)
"""
```

优化训练

```python
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
```

![image-20200923164526736](https://gitee.com/karlhan/picgo/raw/master/img//image-20200923164526736.png)

![image-20200923164533597](https://gitee.com/karlhan/picgo/raw/master/img//image-20200923164533597.png)

![image-20200923164538016](https://gitee.com/karlhan/picgo/raw/master/img//image-20200923164538016.png)

![image-20200923164544074](https://gitee.com/karlhan/picgo/raw/master/img//image-20200923164544074.png)

