# pytorch教程之nn.Module类详解

**前言**pytorch中对于一般的序列模型，直接使用torch.nn.Sequential类及可以实现，这点类似于keras，但是更多的时候面对复杂的模型，比如：多输入多输出、多分支模型、跨层连接模型、带有自定义层的模型等，就需要自己来定义一个模型了。本文将详细说明如何让使用Mudule类来自定义一个模型。

### **一、torch.nn.Module类概述**

个人理解，pytorch不像tensorflow那么底层，也不像keras那么高层，这里先比较keras和pytorch的一些小区别。

（1）keras更常见的操作是通过继承Layer类来实现自定义层，不推荐去继承Model类定义模型，详细原因可以参见官方文档

（2）pytorch中其实一般没有特别明显的Layer和Module的区别，不管是***\*自定义层、自定义块、自定义模型，都是通过继承Module类完成的\****，这一点很重要。其实Sequential类也是继承自Module类的。

**注意：**我们当然也可以直接通过继承torch.autograd.Function类来自定义一个层，但是这很不推荐，不提倡，至于为什么后面会介绍。

**总结：**pytorch里面一切自定义操作基本上都是继承nn.Module类来实现的

本文仅仅先讨论使用Module来实现自定义模块，自定义层先不做讨论。

### **二、torch.nn.Module类的简介**

```python
class Module(object):
    def __init__(self):
    def forward(self, *input):
 
    def add_module(self, name, module):
    def cuda(self, device=None):
    def cpu(self):
    def __call__(self, *input, **kwargs):
    def parameters(self, recurse=True):
    def named_parameters(self, prefix='', recurse=True):
    def children(self):
    def named_children(self):
    def modules(self):  
    def named_modules(self, memo=None, prefix=''):
    def train(self, mode=True):
    def eval(self):
    def zero_grad(self):
    def __repr__(self):
    def __dir__(self):
'''
有一部分没有完全列出来
'''
```

我们在定义自已的网络的时候，需要继承nn.Module类，并**重新实现构造函数__init__构造函数和forward这两个方法**。但有一些注意技巧：

（1）一般把网络中具有可学习参数的层（如全连接层、卷积层等）放在构造函数__init__()中，当然我也可以吧不具有参数的层也放在里面；

（2）一般把不具有可学习参数的层(如ReLU、dropout、BatchNormanation层)可放在构造函数中，也可不放在构造函数中，如果不放在构造函数__init__里面，则在forward方法里面可以使用nn.functional来代替

（3）forward方法是必须要重写的，它是实现模型的功能，实现各个层之间的连接关系的核心。

```python
import torch
 
class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()  # 第一句话，调用父类的构造函数
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.relu1=torch.nn.ReLU()
        self.max_pooling1=torch.nn.MaxPool2d(2,1)
 
        self.conv2 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.relu2=torch.nn.ReLU()
        self.max_pooling2=torch.nn.MaxPool2d(2,1)
 
        self.dense1 = torch.nn.Linear(32 * 3 * 3, 128)
        self.dense2 = torch.nn.Linear(128, 10)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pooling1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pooling2(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
 
model = MyNet()
print(model)
'''运行结果为：
MyNet(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu1): ReLU()
  (max_pooling1): MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu2): ReLU()
  (max_pooling2): MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
  (dense1): Linear(in_features=288, out_features=128, bias=True)
  (dense2): Linear(in_features=128, out_features=10, bias=True)
)
'''
```

**注意：**上面的是将所有的层都放在了构造函数__init__里面，但是只是定义了一系列的层，各个层之间到底是什么连接关系并没有，而是在forward里面实现所有层的连接关系，当然这里依然是顺序连接的。下面再来看一下一个例子：

```python

import torch
import torch.nn.functional as F
 
class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()  # 第一句话，调用父类的构造函数
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(3, 32, 3, 1, 1)
 
        self.dense1 = torch.nn.Linear(32 * 3 * 3, 128)
        self.dense2 = torch.nn.Linear(128, 10)
 
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
 
model = MyNet()
print(model)
'''运行结果为：
MyNet(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (dense1): Linear(in_features=288, out_features=128, bias=True)
  (dense2): Linear(in_features=128, out_features=10, bias=True)
)

```

注意：此时，将没有训练参数的层没有放在构造函数里面了，所以这些层就不会出现在model里面，但是运行关系是在forward里面通过functional的方法实现的。

**总结：**所有放在构造函数__init__里面的层的都是这个模型的“固有属性”.

### **三、torch.nn.Module类的的多种实现**

##### 方式一

```python
import torch.nn as nn
from collections import OrderedDict
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.dense_block = nn.Sequential(
            nn.Linear(32 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    # 在这里实现层之间的连接关系，其实就是所谓的前向传播
    def forward(self, x):
        conv_out = self.conv_block(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense_block(res)
        return out
 
model = MyNet()
print(model)
'''运行结果为：
MyNet(
  (conv_block): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (dense_block): Sequential(
    (0): Linear(in_features=288, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=10, bias=True)
  )
)
'''
```

**3.2 Module类的几个常见方法使用**

**特别注意：**Sequential类虽然继承自Module类，二者有相似部分，但是也有很多不同的部分，集中体现在：

Sequenrial类实现了整数索引，故而可以使用model[index] 这样的方式获取一个曾，但是Module类并没有实现整数索引，不能够通过整数索引来获得层，那该怎么办呢？它提供了几个主要的方法，如下：

```python
def children(self):
 
def named_children(self):
 
def modules(self):
 
def named_modules(self, memo=None, prefix=''):
 
'''
注意：这几个方法返回的都是一个Iterator迭代器，故而通过for循环访问，当然也可以通过next
'''
```

**（1）model.children()方法**

```python
import torch.nn as nn
from collections import OrderedDict
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv_block=torch.nn.Sequential()
        self.conv_block.add_module("conv1",torch.nn.Conv2d(3, 32, 3, 1, 1))
        self.conv_block.add_module("relu1",torch.nn.ReLU())
        self.conv_block.add_module("pool1",torch.nn.MaxPool2d(2))
 
        self.dense_block = torch.nn.Sequential()
        self.dense_block.add_module("dense1",torch.nn.Linear(32 * 3 * 3, 128))
        self.dense_block.add_module("relu2",torch.nn.ReLU())
        self.dense_block.add_module("dense2",torch.nn.Linear(128, 10))
 
    def forward(self, x):
        conv_out = self.conv_block(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense_block(res)
        return out
 
model = MyNet()
 
for i in model.children():
    print(i)
    print(type(i)) # 查看每一次迭代的元素到底是什么类型，实际上是 Sequential 类型,所以有可以使用下标index索引来获取每一个Sequenrial 里面的具体层
 
'''运行结果为：
Sequential(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
<class 'torch.nn.modules.container.Sequential'>
Sequential(
  (dense1): Linear(in_features=288, out_features=128, bias=True)
  (relu2): ReLU()
  (dense2): Linear(in_features=128, out_features=10, bias=True)
)
<class 'torch.nn.modules.container.Sequential'>
'''
```

**（2）model.named_children()方法**

```python
for i in model.children():
    print(i)
    print(type(i)) # 查看每一次迭代的元素到底是什么类型，实际上是 返回一个tuple,tuple 的第一个元素是
 
'''运行结果为：
('conv_block', Sequential(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
))
<class 'tuple'>
('dense_block', Sequential(
  (dense1): Linear(in_features=288, out_features=128, bias=True)
  (relu2): ReLU()
  (dense2): Linear(in_features=128, out_features=10, bias=True)
))
<class 'tuple'>
'''
```

（1）model.children()和model.named_children()方法***\*返回的是迭代器iterator\****；

（2）model.children():每一次迭代返回的***\*每一个元素实际上是 Sequential 类型\****,而Sequential类型又可以使用下标index索引来获取每一个Sequenrial 里面的具体层，比如conv层、dense层等；

（3）model.named_children():每一次迭代返回的***\*每一个元素实际上是 一个元组类型\****，元组的第一个元素是名称，第二个元素就是对应的层或者是Sequential。

**（3）model.modules()方法**

```
for i in model.modules():
    print(i)
    print("==================================================")
'''运行结果为：
MyNet(
  (conv_block): Sequential(
    (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (relu1): ReLU()
    (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (dense_block): Sequential(
    (dense1): Linear(in_features=288, out_features=128, bias=True)
    (relu2): ReLU()
    (dense2): Linear(in_features=128, out_features=10, bias=True)
  )
)
==================================================
Sequential(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
==================================================
Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
==================================================
ReLU()
==================================================
MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
==================================================
Sequential(
  (dense1): Linear(in_features=288, out_features=128, bias=True)
  (relu2): ReLU()
  (dense2): Linear(in_features=128, out_features=10, bias=True)
)
==================================================
Linear(in_features=288, out_features=128, bias=True)
==================================================
ReLU()
==================================================
Linear(in_features=128, out_features=10, bias=True)
==================================================
'''
```

**（4）model.named_modules()方法**

```python
for i in model.named_modules():
    print(i)
    print("==================================================")
'''运行结果是：
('', MyNet(
  (conv_block): Sequential(
    (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (relu1): ReLU()
    (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (dense_block): Sequential(
    (dense1): Linear(in_features=288, out_features=128, bias=True)
    (relu2): ReLU()
    (dense2): Linear(in_features=128, out_features=10, bias=True)
  )
))
==================================================
('conv_block', Sequential(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
))
==================================================
('conv_block.conv1', Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
==================================================
('conv_block.relu1', ReLU())
==================================================
('conv_block.pool1', MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
==================================================
('dense_block', Sequential(
  (dense1): Linear(in_features=288, out_features=128, bias=True)
  (relu2): ReLU()
  (dense2): Linear(in_features=128, out_features=10, bias=True)
))
==================================================
('dense_block.dense1', Linear(in_features=288, out_features=128, bias=True))
==================================================
('dense_block.relu2', ReLU())
==================================================
('dense_block.dense2', Linear(in_features=128, out_features=10, bias=True))
==================================================
'''
```

（1）model.modules()和model.named_modules()方法***\*返回的是迭代器iterator\****；

（2）model的modules()方法和named_modules()方法都会将整个模型的所有构成（包括包装层、单独的层、自定义层等）***\*由浅入深依次遍历出来\****，只不过modules()返回的每一个元素是直接返回的层对象本身，而named_modules()返回的每一个元素是一个元组，第一个元素是名称，第二个元素才是层对象本身。

（3）如何理解children和modules之间的这种差异性。注意pytorch里面不管是模型、层、激活函数、损失函数都可以当成是Module的拓展，所以modules和named_modules会层层迭代，由浅入深，将每一个自定义块block、然后block里面的每一个层都当成是module来迭代。而children就比较直观，就表示的是所谓的“孩子”，所以没有层层迭代深入。

 

**注意：**上面这四个方法是以层包装为例来说明的，如果没有层的包装，我们依然可以使用这四个方法，其实结果也是类似的这样去推，这里就不再列出来了。



参考：

[pytorch教程之nn.Module类详解——使用Module类来自定义模型](https://blog.csdn.net/qq_27825451/article/details/90550890)













