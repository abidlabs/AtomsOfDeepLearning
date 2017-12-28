import tensorflow as tf
import numpy as np
import time

class Timer():
    def __init__(self):
        pass
    def start(self):
        self.time = time.time()
    def end(self):
        return time.time() - self.time
    def end_and_print(self):
        print("Elapsed time:",np.round(time.time()-self.time,3),"s")

        
### COMMON FUNCTIONS ###


def polynomial_to_power(power=2):
    from scipy.misc import factorial
    def polynomial_to_power_function(x_values):
        return 1/factorial(power)*np.power(np.sum(x_values, axis=1),power)
    return polynomial_to_power_function

def sin(omega=6):
    def sin_function(x_values):
        return np.sin(omega*x_values)
    return sin_function

def polynomial(coefs=[1,1,1]):
    def polynomial_function(x_values):
        return np.polynomial.polynomial.polyval(x_values,coefs)
    return polynomial_function

### END COMMON FUNCTIONS ###

'''
Helper function to define a multi-layer perceptron.
x: input tensorflow node
num_nodes: array that contains the number of nodes in each hidden layer
num_input: number of nodes in input layer
num_output: number of nodes in output layer
activation: the tensorflow activation function to user
'''
def multilayer_perceptron(x, num_nodes, num_input=1, num_output=1, activation=tf.nn.sigmoid):
    n_prev = num_input
    out = x
    for n in num_nodes:
        w = tf.Variable(tf.random_normal([n_prev, n]))
        b = tf.Variable(tf.random_normal([n]))
        
        out = activation(tf.add(tf.matmul(out,w),b))
        n_prev = n
        
    w_out = tf.Variable(tf.random_normal([n, num_output]))
    b_out = tf.Variable(tf.random_normal([num_output]))
    out = tf.add(tf.matmul(out,w_out),b_out)
    return out

class Experiment():
    def __init__(self):
        pass
    def initialize(self):
        np.random.seed(0)
        tf.set_random_seed(0)

'''
Experiment 1: Why do we use neural networks?
Description: Performs regression using a neural network with 1 hidden layer and different number of units. Returns the original x-values, true y-values, and predicted y-values, along with the MSE loss.
'''
class Experiment1(Experiment):
    def __init__(self):
        pass
        
    def run(self,
            n_hidden = 2,
            learning_rate = 0.01,
            num_steps = 10000,
            num_values = 100,
            function = sin(omega=6),
            verbose=True):
        
        
        x_values = np.linspace(-1,1, num_values).reshape(-1,1)
        y_values = function(x_values).reshape(-1,1)

        tf.reset_default_graph()
        x = tf.placeholder(dtype="float", shape=[None,1])
        y = tf.placeholder(dtype="float", shape=[None,1])
        y_ = multilayer_perceptron(x, num_nodes=[n_hidden])

        loss_op = tf.reduce_mean(tf.square(y_ - y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            y_preds = list()
            for step in range(num_steps):
                _, loss, y_pred = sess.run([train_op, loss_op, y_], feed_dict={x:x_values,y:y_values})
                if (step%(num_steps/10)==0 and verbose):
                    print(loss)
                    y_preds.append(y_pred.squeeze())

        return x_values.squeeze(), y_values.squeeze(), y_pred.squeeze(), loss

'''
Experiment 2: Why are Deeper Networks Better?
'''
class Experiment2(Experiment):
    def __init__(self):
        pass
        
    def run(self,
            d = 4,
            n=1000, 
            n_hidden=[8],
            num_steps=10000,
            learning_rate = 0.01,
            verbose=True,
            function=polynomial_to_power(power=4)):
        
        
        x_values = np.random.normal(0,1,(n,d))
        y_values = function(x_values).reshape(-1,1)

        tf.reset_default_graph()
        x = tf.placeholder(dtype="float", shape=[None,d])
        y = tf.placeholder(dtype="float", shape=[None,1])
        losses = []

        y_ = multilayer_perceptron(x, num_input=d, num_nodes=n_hidden)

        loss_op = tf.reduce_mean(tf.square(y_ - y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            for step in range(num_steps):
                _, loss, y_pred = sess.run([train_op, loss_op, y_], feed_dict={x:x_values,y:y_values})

        return loss
