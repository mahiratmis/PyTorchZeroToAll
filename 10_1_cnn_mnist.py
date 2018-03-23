# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os.path
import time
from ozu_config import settings as cfg
from tensorboardX import SummaryWriter
from torchviz import make_dot


# Training settings
batch_size = 64
num_epochs = 10000
print_every = 100
start_epoch = 0
best_accuracy = torch.FloatTensor([0])

# CUDA?
cuda = torch.cuda.is_available()

# Seed for reproducibility
torch.manual_seed(1)
if cuda:
    torch.cuda.manual_seed(1)


# MNIST Dataset
train_dataset = datasets.MNIST(root=cfg['DATASETS_DIR']+'mnist/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root=cfg['DATASETS_DIR']+'mnist/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return F.log_softmax(x)


model = Net()

# If you are running a GPU instance, load the model on GPU
if cuda:
    model.cuda()

# #### Loss and Optimizer ####
# Softmax is internally computed.
loss_fn = nn.CrossEntropyLoss()
# If you are running a GPU instance, compute the loss on GPU
if cuda:
    loss_fn.cuda()

# Set parameters to be updated.
optimizer = optim.Adam(model.parameters(), lr=0.01)
model_pth = cfg['DATASETS_DIR'] + 'output/checkpoint2.pth.tar'
# If exists a best model, load its weights!
if os.path.isfile(model_pth):
    print("=> loading checkpoint '{}' ...".format(model_pth))
    if cuda:
        checkpoint = torch.load(model_pth)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(model_pth,
                                map_location=lambda storage,
                                loc: storage)
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (trained for {} epochs)".format(
        model_pth,
        checkpoint['epoch']))


def train(epoch):
    """Perform a full training over dataset"""
    average_time = 0
    # Model train mode
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # measure data loading time
        batch_time = time.time()
        images = Variable(images)
        labels = Variable(labels)

        if cuda:
            images, labels = images.cuda(), labels.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        if epoch == 1:
            # visualize execution graph
            make_dot(outputs, params=dict(model.named_parameters()))
        loss = loss_fn(outputs, labels)

        # Load loss on CPU
        if cuda:
            loss.cpu()

        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time = time.time() - batch_time
        # Accumulate over batch
        average_time += batch_time

        # ### Keep track of metric every batch
        # Accuracy Metric
        prediction = outputs.data.max(1)[1]   # first column has actual prob.
        accuracy = 100. * prediction.eq(labels.data).sum() / batch_size

        # Log
        if (i + 1) % print_every == 0:
            print (('Epoch:{}/{}, Step:{}/{}, Loss:{:.4f}, Accur: {:.4f}' +
                    ', Batch time: {:.4f}').format(
                epoch + 1,
                num_epochs,
                i + 1,
                len(train_dataset) / batch_size,
                loss.data[0],
                accuracy,
                average_time/print_every))  # Average


def eval(model):
    """Eval over test set"""
    model.eval()
    correct = 0
    # Get Batch
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        if cuda:
            data, target = data.cuda(), target.cuda()
        # Evaluate
        output = model(data)
        # Load output on CPU
        if cuda:
            output.cpu()
        # Compute Accuracy
        prediction = output.data.max(1)[1]
        correct += prediction.eq(target.data).sum()
    return correct


def save_checkpoint(state, is_best, filename='/output/checkpoint.pth.tar'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print ("=> Saving a new best")
        if not os.path.exists(filename):
            with open(filename, 'w'):
                pass
        torch.save(state, filename)  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")


dataset_length = len(test_loader.dataset)

# to visualize using tensorboard
# to viualize the logs run the following on a terminal
# tensorboard --log_dir=/home/vvglab/tblogs port=6006
# tensorboard --logdir=cfg['TENSORBOARD_LOGDIR'] port=6006
# open a browser and type http://localhost:6006

writer = SummaryWriter(log_dir=cfg['TENSORBOARD_LOGDIR'])

for epoch in range(start_epoch, num_epochs):
    train(epoch)
    best_accuracy
    acc = eval(model)
    acc = 100. * acc / dataset_length
    print('=> Test set: Accuracy: {:.2f}%'.format(acc))
    acc = torch.FloatTensor([acc])
    # Get bool not ByteTensor
    is_best = bool(acc.numpy() > best_accuracy.numpy())
    # Get greater Tensor to keep track best acc
    best_accuracy = torch.FloatTensor(max(acc.numpy(), best_accuracy.numpy()))
    # Save checkpoint if is a new best
    fname = cfg['DATASETS_DIR']+'output/checkpoint'+str(epoch+1)+'.pth.tar'
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_accuracy': best_accuracy
    }, is_best, filename=fname)
    writer.add_scalar('data/Accuracy', acc, epoch)
    writer.add_scalar('data/BestAcc', best_accuracy, epoch)

writer.close()
