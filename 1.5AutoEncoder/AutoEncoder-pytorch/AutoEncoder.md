# AutoEncoder

### 一、实验数据

实验采用了MNIST数据集

```python
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)
train_loader = Data.DataLoader(dataset = train_data,batch_size=BATCH_SIZE,shuffle=True)

```

检测是否存在数据集目录，如果没有则下载

### 二、搭建模型

```python
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
        )
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded
```

就是一个Encoder-Decoder。先压缩再解压。中间数据一定要小于输入数据规模，否则模型会倾向于直接把输入复制到输出。


### 三、训练测试

```python
for epoch in range(EPOCH):
    for step,(x,b_label) in enumerate(train_loader):
        b_x = x.view(-1,28*28).cuda()
        b_y = x.view(-1,28*28).cuda()

        encoded,decoded = autoencoder(b_x)
        loss = loss_func(decoded,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 ==0:
            print("Epoch:{} ,train loss:{}".format(epoch,loss.data.cpu().numpy()))
            _,decoded_data = autoencoder(view_data)
    if epoch % 2 == 0:
        for i in range(N_TEST_IMG):
            a[1][i].clear()
            a[1][i].imshow(np.reshape(decoded_data.data.cpu().numpy()[i], (28, 28)), cmap='gray')
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())
        plt.draw()
        plt.pause(0.05)
plt.ioff()
plt.show()
```

loss是输入encoder和输出decoder直接进行计算，没有用到label。每两个epoch输出一次decoder解压后的图像。

### 四、实验结果分析

第一个epoch loss就已经降到0.04，很稳定了。之后一直都是0.03多一点点。

下图第一行是输入的图片，第二行是decoder输出的图片。

![image-20200920164735315](https://gitee.com/karlhan/picgo/raw/master/img//image-20200920164735315.png)

![image-20200920164742612](https://gitee.com/karlhan/picgo/raw/master/img//image-20200920164742612.png)

第一个数字是5，和8,3非常接近。在下图3D视角中也容易看见两部分有交叠。

<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20200920170524356.png" alt="image-20200920170524356" style="zoom:80%;" />

改了下模型，将隐藏层改成了8*8，输出了一下encoder之后的样子。

![image-20200920200244449](https://gitee.com/karlhan/picgo/raw/master/img//image-20200920200244449.png)