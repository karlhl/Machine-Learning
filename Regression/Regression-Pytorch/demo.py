import torch
import matplotlib.pyplot as plt
x = torch.linspace(-1,1,100)
x = torch.unsqueeze(x,dim=1)
print(x.shape)

y = x.pow(2) + 0.05*torch.randn(x.size())
y2 = x.pow(2)

print(y)

plt.scatter(x.data.numpy(), y.data.numpy())
plt.scatter(x.data.numpy(), y2)
plt.show()