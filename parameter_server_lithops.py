import torch
import os
from glob2 import glob
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.autograd as dist_autograd

import lithopsext
import lithops

lithops.utils.setup_lithops_logger(log_level='DEBUG')

# def upload_data():
#     storage = lithops.storage.Storage()
#     for obj in glob(os.path.expanduser('~/Downloads/mnist_png/*/*/*')):
#         # print(obj)
#         with open(obj, 'rb') as file:
#             data = file.read()
#             key = obj.split('mnist_png/', 1)[1]
#             storage.put_object(bucket='aitor-data', key='mnist_png/{}'.format(key), body=data)
#
# upload_data()

train_dataset = lithopsext.datasets.ObjectBag.s3_glob('s3://aitor-data/mnist_png/training/*',
                                                      batch_size=1000,
                                                      lazy_loading=True)
print(train_dataset)


def get_accuracy(test_loader, model):
    model.eval()
    correct_sum = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            out = model(data, -1)
            pred = out.argmax(dim=1, keepdim=True)
            pred, target = pred, target
            correct = pred.eq(target.view_as(pred)).sum().item()
            correct_sum += correct

    print(f"Accuracy {correct_sum / len(test_loader.dataset)}")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # Move tensor to next device if necessary
        next_device = next(self.fc1.parameters()).device
        x = x.to(next_device)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def training_loop(training_data_chunk, iterations, test_loader):
    net = Net()

    for _ in range(iterations):
        data, target = next(training_data_chunk)
        net.zero_grad()
        output = net(training_data_chunk)
        loss = F.nll_loss(output, target)
        loss.backward()
        return self.model.get_gradients()


run_training_loop(train_data, test_data)
