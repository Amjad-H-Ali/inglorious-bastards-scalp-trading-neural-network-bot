import numpy as np
import pandas as pd
import random


# Read the data back from CSV
df = pd.read_csv('stock_data.csv', index_col=0)

# Drop the index column from the dataframe
df.reset_index(drop=True, inplace=True)

df = df.sample(frac=1).reset_index(drop=True)

# Separate the output from the input tuples
output = np.array(df['Output'])

netinput = np.array(df.drop(columns='Output'))

# Assume netinput is a numpy array
# Find the minimum and maximum of the data
min_val = np.min(netinput, axis=0)
max_val = np.max(netinput, axis=0)

# Normalize the data
netinput = (netinput - min_val) / (max_val - min_val)

# initialize a new array of shape (len(output), 2) with zeros
netoutput = np.zeros((output.shape[0], 2))

# assign 1 to the column that corresponds to the value in output
netoutput[np.arange(output.shape[0]), output] = 1

n, _ = netoutput.shape

n_half = n//2

# Calculate the index for the quarter and three quarters points in your dataset
n_quarter = n_half // 2

test_set = [(x, y) for x, y in zip(netinput[:n_half], netoutput[:n_half])]

train_set = [(x, y) for x, y in zip(netinput[n_half:], netoutput[n_half:])]

# Split your training data into two halves
test_set_1 = test_set[:n_quarter]
test_set_2 = test_set[n_quarter:]

# Filter the second half of your testing data to only include rows where the second value in the tuple is 1
test_set_1_filtered = [(x, y) for x, y in test_set_1 if y[1] == 1]

# Filter the second half of your testing data to only include rows where the second value in the tuple is 1
test_set_2_filtered = [(x, y) for x, y in test_set_2 if y[1] == 1]

print(len(test_set_1_filtered), len(test_set_2_filtered))


# data = np.array(pd.read_csv('train.csv'))
# m, n = data.shape
# labels = np.zeros((m, 10))

# # Set the index corresponding to the label in "data" to 1
# for i in range(m):
#     labels[i, data[i, 0]] = 1

# # Normalize the pixel intensities to the range 0-1
# data = data[:, :] / 255.0

# # Remove the first element from each subarray in data
# data = data[:, 1:]

# test_set = [(x, y) for x, y in zip(data[:1000], labels[:1000])]

# train_set = [(x, y) for x, y in zip(data[1000:], labels[1000:])]



class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb 
                        for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])

        nabla_b[-1] = delta
        # nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_w[-1] = np.dot(delta.reshape(len(delta), 1), activations[-2].reshape(1, len(activations[-2])))

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta.reshape(len(delta), 1), activations[-l-1].reshape(1, len(activations[-l-1])))
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""

        return (output_activations-y)
        
#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))



net = Network([40,2,2])
net.SGD(test_set_1, 30, 10, 3.0, test_data=test_set_2_filtered)




# class NeuralNetwork:
#     def __init__(self, learning_rate=0.1):
#         # Weights
#         self.w1 = np.random.rand(16, 784)
#         self.w2 = np.random.rand(10, 16)

#         # Biases
#         self.b1 = np.zeros(16)
#         self.b2 = np.zeros(10)

#         self.learning_rate = learning_rate

#     def sigmoid(self, x):
#         return 1 / (1 + np.exp(-x))

#     def derivative_sigmoid(self, x):
#         return self.sigmoid(x) * (1 - self.sigmoid(x))

#     def feedforward(self, x):
#         # Hidden layer
#         h = self.sigmoid(np.dot(self.w1, x) + self.b1)

#         # Output layer
#         o = self.sigmoid(np.dot(self.w2, h) + self.b2)

#         return h, o

#     def backpropagation(self, x, y, h, o):

#         # Output layer
#         d_L_d_o = 2 * (o - y) 
#         d_o_d_z2 = self.derivative_sigmoid(np.dot(self.w2, h) + self.b2)
#         d_L_d_z2 = d_L_d_o * d_o_d_z2

#         # Hidden layer
#         d_z2_d_h = self.w2
#         d_L_d_h = np.dot(d_L_d_z2, d_z2_d_h)
#         d_h_d_z1 = self.derivative_sigmoid(np.dot(self.w1, x) + self.b1)
#         d_L_d_z1 = d_L_d_h * d_h_d_z1

#         # Gradients for weights and biases
#         d_z2_d_w2 = h
#         d_L_d_w2 = d_z2_d_w2 * d_L_d_z2

#         d_z1_d_w1 = x
#         d_L_d_w1 = d_z1_d_w1 * d_L_d_z1

#         # Update weights and biases
#         self.w1 -= self.learning_rate * d_L_d_w1
#         self.b1 -= self.learning_rate * d_L_d_z1
#         self.w2 -= self.learning_rate * d_L_d_w2
#         self.b2 -= self.learning_rate * d_L_d_z2

#     def train(self, data, labels):
#         for _ in range(1000):  # epochs
#             for x, y in zip(data, labels):
#                 h, o = self.feedforward(x)
#                 self.backpropagation(x, y, h, o)

# Initialize and train the neural network
# n = Network([784,30,10])
# nn.train(data, labels)
