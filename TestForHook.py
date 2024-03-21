import torch
import torch.nn as nn

class TestForHook(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=2, out_features=2)
        self.linear2 = nn.Linear(in_features=2, out_features=1)
        self.relu = nn.ReLU()
        self.relu6 = nn.ReLU6()
        self.initialize()

    def forward(self, x):
        linear_1 = self.linear1(x)
        linear_2 = self.linear2(linear_1)
        relu = self.relu(linear_2)
        relu_6 = self.relu6(relu)
        layers_in = (x, linear_1, linear_2)
        layers_out = (linear_1, linear_2, relu)
        return relu_6, layers_in, layers_out

    def initialize(self):
        self.linear1.weight = torch.nn.Parameter(torch.FloatTensor([[1, 1], [1, 1]]))
        self.linear1.bias = torch.nn.Parameter(torch.FloatTensor([[1, 1]]))
        self.linear2.weight = torch.nn.Parameter(torch.FloatTensor([[1, 1]]))
        self.linear2.bias = torch.nn.Parameter(torch.FloatTensor([[1]]))
        return True

module_name = []
features_in_hook = []
features_out_hook = []

def hook(module, feature_in, feature_out):
    module_name.append(module.__class__)
    features_in_hook.append(feature_in)
    features_out_hook.append(feature_out)
    return None

net = TestForHook()
net_children = net.children()
for child in net_children:
    # if not isinstance(child, nn.ReLU6):
    child.register_forward_hook(hook=hook)

x = torch.FloatTensor([[0.2, 0.2], [0.1, 0.1]])

a1, a2, a3 = net(x)
print(features_out_hook, a3)