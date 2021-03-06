"""All Torch Linear Regression."""
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torchviz import make_dot

from ozu_config import settings


class LinearModel(torch.nn.Module):
    """Linear Model."""

    def __init__(self):
        """In the constructor we instantiate one nn.Linear module."""
        super(LinearModel, self).__init__()
        # Both input and output are one dimensional
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        """Forward pass with input x.

        In the forward function we accept a Variable of input data and
        we must return a Variable of output data. We can use Modules defined
        in the constructor as well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred


x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

# our model
model = LinearModel()
# to visualize using tensorboard
# to viualize the logs run the following on a terminal
# tensorboard --log_dir=/home/vvglab/tblogs port=6006
# open a browser and type http://localhost:6006

writer = SummaryWriter(log_dir=settings['TENSORBOARD_LOGDIR'])

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data[0])
    print(epoch, model.linear.weight.data[0][0])

    # Log loss and weight values
    writer.add_scalar('data/Loss', loss.data[0], epoch)
    writer.add_scalar('data/Weight', model.linear.weight, epoch)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    # print(epoch, model.linear.weight.grad.data[0][0])
    optimizer.step()


writer.close()

# After training
hour_var = Variable(torch.Tensor([[4.0]]))
y_pred = model(hour_var)
print("predict (after training)", 4, model(hour_var).data[0][0])
# visualize execution graph
make_dot(y_pred, params=dict(model.named_parameters()))
