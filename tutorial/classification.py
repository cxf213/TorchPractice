import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import  torch.utils.data as Data

# torch.manual_seed(1)    # reproducible

# make fake data
n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)  # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)  # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2 * n_data, 1)  # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)  # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)  # shape (200,) LongTensor = 64-bit integer

BATCH_SIZE = 5
torch_dataset = Data.TensorDataset(x,y)
loader = Data.DataLoader()

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.out(x)
        return x


net = Net(n_feature=2, n_hidden=4, n_output=2)  # define the network
# print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

# plt.ion()  # something about plotting


for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        out = net(x)  # input x and predict based on x
        loss = loss_func(out, y)  # must be (1. nn output, 2. target), the target label is NOT one-hotted

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())


for t in range(100):
    out = net(x)  # input x and predict based on x
    loss = loss_func(out, y)  # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    if t % 10 == 0:
        # plot and show learning process
        # plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        # plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        print(accuracy)
        # plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        #plt.pause(0.1)

# plt.ioff()
# plt.show()
