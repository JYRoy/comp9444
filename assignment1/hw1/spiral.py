# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        self.linear1 = nn.Linear(2, num_hid)
        self.linear2 = nn.Linear(num_hid, 1)

    def forward(self, input):
        x = input[:, 0]
        y = input[:, 1]
        r = torch.sqrt(x*x + y*y)
        a = torch.atan2(y, x)
        input = torch.stack((r,a),1)
        self.hidden_layer_1 = torch.tanh(self.linear1(input))

        # Changing the activation to relu from tanh in 6(3)
        #self.hidden_layer_1 = nn.functional.relu(self.linear1(input))

        output = torch.sigmoid(self.linear2(self.hidden_layer_1))
        return output

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.linear1 = nn.Linear(2, num_hid)
        self.linear2 = nn.Linear(num_hid, num_hid)
        self.linear3 = nn.Linear(num_hid, 1)

        # Adding a hidden layer in 6(3)
        # self.linear1 = nn.Linear(2, num_hid)
        # self.linear2 = nn.Linear(num_hid, num_hid)
        # self.linear3 = nn.Linear(num_hid, num_hid)
        # self.linear4 = nn.Linear(num_hid, 1)

    def forward(self, input):
        self.hidden_layer_1 = torch.tanh(self.linear1(input))
        self.hidden_layer_2 = torch.tanh(self.linear2(self.hidden_layer_1))
        output = torch.sigmoid(self.linear3(self.hidden_layer_2))

        # Adding the third hidden layer in 6(3)
        # self.hidden_layer_1 = torch.tanh(self.linear1(input))
        # self.hidden_layer_2 = torch.tanh(self.linear2(self.hidden_layer_1))
        # self.hidden_layer_3 = torch.tanh(self.linear3(self.hidden_layer_2))
        # output = torch.sigmoid(self.linear4(self.hidden_layer_3))

        return output

def graph_hidden(net, layer, node):
    pass
    # plt.clf()
    # xrange = torch.arange(start=-7, end=7.1, step=0.01, dtype=torch.float32)
    # yrange = torch.arange(start=-6.6, end=6.7, step=0.01, dtype=torch.float32)
    # xcoord = xrange.repeat(yrange.size()[0])
    # ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    # grid = torch.cat((xcoord.unsqueeze(1), ycoord.unsqueeze(1)), 1)
    #
    # with torch.no_grad():  # suppress updating of gradients
    #     net.eval()  # toggle batch norm, dropout
    #     output = net(grid)
    #     net.train()  # toggle batch norm, dropout back again
    #
    #     if layer == 1:
    #         pred = (net.hidden_layer_1[:, node] >= 0.5).float()
    #     else:
    #         pred = (net.hidden_layer_2[:, node] >= 0.5).float()
    #
    #     #pred = (output >= 0.5).float()
    #
    #     # plot function computed by model
    #     plt.clf()
    #     plt.pcolormesh(xrange, yrange, pred.cpu().view(yrange.size()[0], xrange.size()[0]), shading='auto',
    #                    cmap='Wistia')



