# Pytorch

### Torch &  numpy

创建numpy和tensor以及相互转换

```python
np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
print(
    '\nnumpy array:', np_data,          # [[0 1 2], [3 4 5]]
    '\ntorch tensor:', torch_data,      #  0  1  2 \n 3  4  5    [torch.LongTensor of size 2x3]
    '\ntensor to array:', tensor2array, # [[0 1 2], [3 4 5]]
)

# abs
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # 32-bit floating point
print(
    '\nabs',
    '\nnumpy: ', np.abs(data),          # [1 2 1 2]
    '\ntorch: ', torch.abs(tensor)      # [1 2 1 2]
)

# sin
print(
    '\nsin',
    '\nnumpy: ', np.sin(data),      # [-0.84147098 -0.90929743  0.84147098  0.90929743]
    '\ntorch: ', torch.sin(tensor)  # [-0.8415 -0.9093  0.8415  0.9093]
)

# mean
print(
    '\nmean',
    '\nnumpy: ', np.mean(data),         # 0.0
    '\ntorch: ', torch.mean(tensor)     # 0.0
)

# matrix multiplication
data = [[1,2], [3,4]]
tensor = torch.FloatTensor(data)  # 32-bit floating point
# correct method
print(
    '\nmatrix multiplication (matmul)',
    '\nnumpy: ', np.matmul(data, data),     # [[7, 10], [15, 22]]
    '\ntorch: ', torch.mm(tensor, tensor)   # [[7, 10], [15, 22]]
)
# incorrect method
data = np.array(data)
print(
    '\nmatrix multiplication (dot)',
    '\nnumpy: ', data.dot(data),        # [[7, 10], [15, 22]]
    '\ntorch: ', tensor.dot(tensor)     # this will convert tensor to [1,2,3,4], you'll get 30.0
)
```

### variable

```python
import torch
from torch.autograd import Variable

# Variable in torch is to build a computational graph,
# but this graph is dynamic compared with a static graph in Tensorflow or Theano.
# So torch does not have placeholder, torch can just pass variable to the computational graph.

tensor = torch.FloatTensor([[1,2],[3,4]])            # build a tensor
variable = Variable(tensor, requires_grad=True)      # build a variable, usually for compute gradients

print(tensor)       # [torch.FloatTensor of size 2x2]
print(variable)     # [torch.FloatTensor of size 2x2]

# till now the tensor and variable seem the same.
# However, the variable is a part of the graph, it's a part of the auto-gradient.

t_out = torch.mean(tensor*tensor)       # x^2
v_out = torch.mean(variable*variable)   # x^2
print(t_out)    # tensor(7.5000)
print(v_out)    # tensor(7.5000, grad_fn=<MeanBackward0>)

v_out.backward()    # backpropagation from v_out
# v_out = 1/4 * sum(variable*variable)
# the gradients w.r.t the variable, d(v_out)/d(variable) = 1/4*2*variable = variable/2
print(variable.grad)
'''
 0.5000  1.0000
 1.5000  2.0000
'''

print(variable)     # this is data in variable format
"""
Variable containing:
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable.data)    # this is data in tensor format
"""
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable.data.numpy())    # numpy format
"""
[[ 1.  2.]
 [ 3.  4.]]
"""
```

### Activation

```python
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 200)  # x data (tensor), shape=(200, 1)
x = Variable(x)
x_np = x.data.numpy()   # numpy array for plotting

y_relu = F.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()

%matplotlib inline

plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()
```

![image-20200902152243828](https://gitee.com/karlhan/picgo/raw/master/img//image-20200902152243828.png)























参考：

	1.  [莫愁](https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/201_torch_numpy.py)
 	2.  