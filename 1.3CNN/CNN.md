# CNN

卷积是一种有效提取图片特征的方法 。 一般用一个正方形卷积核，遍历图片上的每一个像素点。图片与卷积 核 重合区域内 相对应的每一个像素值 乘卷积核内相对应点的权重，然后求和， 再 加上偏置后，最后得到输出图片中的一个像素值。

### torch.nn.Conv1d()

torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

$$ out(N_i, C_{out_j})=bias(C *{out_j})+\sum^{C*{in}-1}*{k=0}weight(C*{out_j},k)\bigotimes input(N_i,k) $$

**Parameters：**

- in_channels(`int`) – 输入信号的通道
- out_channels(`int`) – 卷积产生的通道
- kerner_size(`int` or `tuple`) - 卷积核的尺寸
- stride(`int` or `tuple`, `optional`) - 卷积步长
- padding (`int` or `tuple`, `optional`)- 输入的每一条边补充0的层数
- dilation(`int` or `tuple`, `optional``) – 卷积核元素之间的间距
- groups(`int`, `optional`) – 从输入通道到输出通道的阻塞连接数
- bias(`bool`, `optional`) - 如果`bias=True`，添加偏置

**shape:**
输入: (N,C_in,L_in)
输出: (N,C_out,L_out)
输入输出的计算方式：
$$L_{out}=floor((L_{in}+2*padding-dilation*(kernerl_size-1)-1)/stride+1)$$

**变量:**
weight(`tensor`) - 卷积的权重，大小是(`out_channels`, `in_channels`, `kernel_size`)
bias(`tensor`) - 卷积的偏置系数，大小是（`out_channel`）

**example:**

```python
m = nn.Conv1d(16, 33, 3, stride=2)
input = autograd.Variable(torch.randn(20, 16, 50))
output = m(input)
"""
input:(20,16,50)
卷积核m:(16,33,3,步长2)
output:(20,33,24) ,24=(50-3+1)/2
"""
```



### torch.nn.Conv2d()

class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

二维卷积层, 输入的尺度是(N, C_in,H,W)，输出尺度（N,C_out,H_out,W_out）的计算方式：

$$out(N_i, C_{out_j})=bias(C_{out_j})+\sum^{C_{in}-1}*{k=0}weight(C*{out_j},k)\bigotimes input(N_i,k)$$

**说明**
`bigotimes`: 表示二维的相关系数计算 `stride`: 控制相关系数的计算步长
`dilation`: 用于控制内核点之间的距离，详细描述在[这里](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)
`groups`: 控制输入和输出之间的连接： `group=1`，输出是所有的输入的卷积；`group=2`，此时相当于有并排的两个卷积层，每个卷积层计算输入通道的一半，并且产生的输出是输出通道的一半，随后将这两个输出连接起来。

参数`kernel_size`，`stride,padding`，`dilation`也可以是一个`int`的数据，此时卷积height和width值相同;也可以是一个`tuple`数组，`tuple`的第一维度表示height的数值，tuple的第二维度表示width的数值

**Parameters：**

- in_channels(`int`) – 输入信号的通道
- out_channels(`int`) – 卷积产生的通道
- kerner_size(`int` or `tuple`) - 卷积核的尺寸
- stride(`int` or `tuple`, `optional`) - 卷积步长
- padding(`int` or `tuple`, `optional`) - 输入的每一条边补充0的层数
- dilation(`int` or `tuple`, `optional`) – 卷积核元素之间的间距
- groups(`int`, `optional`) – 从输入通道到输出通道的阻塞连接数
- bias(`bool`, `optional`) - 如果`bias=True`，添加偏置

**shape:**
input: (N,C_in,H_in,W_in)
output: (N,C_out,H_out,W_out)
$$H_{out}=floor((H_{in}+2*padding[0]-dilation[0]*(kernerl_size[0]-1)-1)/stride[0]+1)$$

$$W_{out}=floor((W_{in}+2*padding[1]-dilation[1]*(kernerl_size[1]-1)-1)/stride[1]+1)$$

**变量:**
weight(`tensor`) - 卷积的权重，大小是(`out_channels`, `in_channels`,`kernel_size`)
bias(`tensor`) - 卷积的偏置系数，大小是（`out_channel`）

**example:**

```python
>>> # With square kernels and equal stride
>>> m = nn.Conv2d(16, 33, 3, stride=2)
>>> # non-square kernels and unequal stride and with padding
>>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
>>> # non-square kernels and unequal stride and with padding and dilation
>>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
>>> input = autograd.Variable(torch.randn(20, 16, 50, 100))
>>> output = m(input)
```



### class torch.nn.Conv3d()

class torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

三维卷积层, 输入的尺度是(N, C_in,D,H,W)，输出尺度（N,C_out,D_out,H_out,W_out）的计算方式：
$$out(N_i, C_{out_j})=bias(C_{out_j})+\sum^{C_{in}-1}*{k=0}weight(C*{out_j},k)\bigotimes input(N_i,k)$$

**说明**
`bigotimes`: 表示二维的相关系数计算 `stride`: 控制相关系数的计算步长
`dilation`: 用于控制内核点之间的距离，详细描述在[这里](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)
`groups`: 控制输入和输出之间的连接： `group=1`，输出是所有的输入的卷积；`group=2`，此时相当于有并排的两个卷积层，每个卷积层计算输入通道的一半，并且产生的输出是输出通道的一半，随后将这两个输出连接起来。
参数`kernel_size`，`stride`，`padding`，`dilation`可以是一个`int`的数据 - 卷积height和width值相同，也可以是一个有三个`int`数据的`tuple`数组，`tuple`的第一维度表示depth的数值，`tuple`的第二维度表示height的数值，`tuple`的第三维度表示width的数值

**Parameters：**

- in_channels(`int`) – 输入信号的通道
- out_channels(`int`) – 卷积产生的通道
- kernel_size(`int` or `tuple`) - 卷积核的尺寸
- stride(`int` or `tuple`, `optional`) - 卷积步长
- padding(`int` or `tuple`, `optional`) - 输入的每一条边补充0的层数
- dilation(`int` or `tuple`, `optional`) – 卷积核元素之间的间距
- groups(`int`, `optional`) – 从输入通道到输出通道的阻塞连接数
- bias(`bool`, `optional`) - 如果`bias=True`，添加偏置

**shape:**
`input`: (N,C_in,D_in,H_in,W_in)
`output`: (N,C_out,D_out,H_out,W_out)
$$D_{out}=floor((D_{in}+2*padding[0]-dilation[0]*(kernerl_size[0]-1)-1)/stride[0]+1)$$

$$H_{out}=floor((H_{in}+2*padding[1]-dilation[2]*(kernerl_size[1]-1)-1)/stride[1]+1)$$

$$W_{out}=floor((W_{in}+2*padding[2]-dilation[2]*(kernerl_size[2]-1)-1)/stride[2]+1)$$

**变量:**

- weight(`tensor`) - 卷积的权重，shape是(`out_channels`, `in_channels`,`kernel_size`)`
- bias(`tensor`) - 卷积的偏置系数，shape是（`out_channel`）

**example:**

```python
>>> # With square kernels and equal stride
>>> m = nn.Conv3d(16, 33, 3, stride=2)
>>> # non-square kernels and unequal stride and with padding
>>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
>>> input = autograd.Variable(torch.randn(20, 16, 10, 50, 100))
>>> output = m(input)
```















