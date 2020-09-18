import torch
import torchvision.datasets as dsets
import torch.nn as nn
import torchvision.transforms as transforms

import os

EPOCH = 30               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.01               # learning rate
DOWNLOAD_MNIST = False   # set to True if haven't download the data
use_cuda = False

if torch.cuda.is_available():
    use_cuda = True

if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = dsets.MNIST(
    root="./mnist/",
    train=True,
    transform = transforms.ToTensor(),

    download=DOWNLOAD_MNIST,
)

train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

test_data = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())

test_x = test_data.test_data.type(torch.FloatTensor).cuda()/255.
test_y = test_data.test_labels.cuda()

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        self.out = nn.Linear(64,10)

    def forward(self,x):
        r_out,(h_n,h_c) = self.rnn(x,None)
        out = self.out(r_out[:,-1,:])
        return out

rnn = RNN()
if use_cuda:
    rnn.cuda()

optimizer = torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step,(b_x,b_y) in enumerate(train_loader):
        b_x = b_x.view(-1, 28, 28)

        if use_cuda:
            b_x = b_x.cuda()
            b_y = b_y.cuda()

        output = rnn(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            test_out = rnn(test_x)
            pred_y = torch.max(test_out,1)[1].cuda().data

            accuracy = torch.sum(pred_y==test_y).type(torch.FloatTensor)/test_y.size(0)

            print('epoch:{}, step:{}, loss:{},acc:{:.4f}'.format(epoch,step,loss.data.cpu().numpy(),accuracy))

