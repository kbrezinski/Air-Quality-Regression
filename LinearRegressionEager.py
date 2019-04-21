
from utils import *
import time

import tensorflow as tf
import tensorflow.contrib.eager as tfe

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

tfe.enable_eager_execution()
file = "AirQualityUCI.xlsx"

# Import Dataset
data = pd.ExcelFile(file).parse()
data = data.iloc[:,2:].astype(np.float32)
data = data.mask(data == -200)

dataset = tf.data.Dataset.from_tensor_slices((data.loc[:,['T','RH','AH']],
                                              data.loc[:,['CO(GT)']]))

# Create trainable variables
w = tfe.Variable(tfe.random_normal([3,1]))
b = tfe.Variable(0.0)

# Create hypothesis
def hypothesis(x):
    return x * w + b

# Create root mean squared error function
def rmse(y, y_predicted):
    return (y - y_predicted) ** 2

# Create loss function
def train(loss_fn, lr):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

def loss_gradient(x, y):
    return loss_fn(y, hypothesis(x))

# Creates a function which returns the loss and gradients in a tuple
grad_fn = tfe.implicit_value_and_gradients(loss_gradient)

## Start the training step
start = time.time()
for epoch in range(n_epochs):

    total_loss = 0.0

    for x_i, y_i in tfe.Iterator(dataset):

        loss, gradients = grad_fn(x_i, y_i)
        optimizer.apply_gradients(gradients)
        total_loss += loss

    if epoch % 10 == 0:
        print('Epoch {0}: {1}'.format(epoch, total_loss))
print('Took: %f' % (time.time() - start))

# Set hyperparameters
lr = 0.01
n_epochs = 100

train(rmse,lr)
