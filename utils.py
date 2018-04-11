import tensorflow as tf
import numpy as np
from IPython.display import clear_output, Image, display, HTML
import warnings
import time    

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))


class Timer():
    def __init__(self):
        pass
    def start(self):
        self.time = time.time()
    def end(self):
        return time.time() - self.time
    def end_and_print(self):
        print("Time needed to run experiment:",np.round(time.time()-self.time,3),"s")
    def end_and_md_print(self):
        from IPython.display import Markdown, display
        string = "Time needed to run experiment: " + str(np.round(time.time()-self.time,3)) + " s"
        display(Markdown(string))


##

import matplotlib.pyplot as plt

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Credit: https://gist.github.com/craffel/2d727968c3aaebd10359
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)


##
        
### COMMON ANALYTICAL FUNCTIONS ###

def random_batch(x_values, y_values,size=64):
    assert x_values.shape[0]==y_values.shape[0]
    n = x_values.shape[0]
    indices = np.random.permutation(n)[:size]
    return x_values[indices], y_values[indices]

def random_values():
    def random_functions(x_values):
        n, d = x_values.shape
        return np.random.normal(0,1,n)
    return random_functions
    

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def sigmoid_of_sigmoid():
    def sigmoid_of_sigmoid_function(x_values):
        y_values = sigmoid(sigmoid(x_values[:,0]+x_values[:,1])+sigmoid(x_values[:,2]+x_values[:,3]))
        return y_values
    return sigmoid_of_sigmoid_function

def polynomial_composition(power=2):
    def polynomial_composition_function(x_values):
        n, d = x_values.shape
        x_values = np.add.reduceat(x_values, axis=1, indices=range(0,d,2)) #adds adjacent columns together
        x_values = x_values**power
        n, d = x_values.shape
        x_values = np.add.reduceat(x_values, axis=1, indices=range(0,d,2)) #adds adjacent columns together
        x_values = x_values**power
        return np.sum(x_values,axis=1)
    return polynomial_composition_function

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

def sparse_trig():
    def sparse_trig_function(x_values):
        return 2*(2*np.cos(x_values)**2-1)**2-1
    return sparse_trig_function

### END COMMON FUNCTIONS ###

'''
Takes the dataset and maps each column to be between 0 and 1
'''
def normalize(array):
    if array.ndim>1:
        return (array - array.min(axis=0)) / array.ptp(axis=0)
    else:
        return (array - array.min()) / array.ptp()        

'''
Helper function to define a multi-layer perceptron.
x: input tensorflow node
num_nodes: array that contains the number of nodes in each hidden layer
num_input: number of nodes in input layer
num_output: number of nodes in output layer
activation: the tensorflow activation function to user
'''
def multilayer_perceptron(x, num_nodes, num_input=1, num_output=1, activation=tf.nn.sigmoid, bias=True, initializer=tf.contrib.layers.xavier_initializer(), return_weight_tensors=False):
    n_prev = num_input
    out = x
    num_layer = 0
    weights = list()
    
    for n in num_nodes:
        w = tf.get_variable("w"+str(num_layer),[n_prev, n], initializer=initializer)
        weights.append(w)
        if bias:
            b = tf.get_variable("b"+str(num_layer),[n], initializer =initializer)
            out = activation(tf.add(tf.matmul(out,w),b),name="out"+str(num_layer))
        else:
            out = activation(tf.matmul(out,w),name="out"+str(num_layer))
            
        n_prev = n
        num_layer += 1
        
    w_out = tf.get_variable("w"+str(num_layer),[n, num_output], initializer =initializer)
    weights.append(w_out)
    
    if bias:
        b_out = tf.get_variable("b"+str(num_layer),[num_output], initializer =initializer)
        out = tf.add(tf.matmul(out,w_out),b_out,name="out"+str(num_layer))
    else:
        out = tf.matmul(out,w_out,name="out"+str(num_layer))
    
    if return_weight_tensors:
        return out, weights
    return out


# Modified MLP for use with experiment 2
def recurrent_multilayer_perceptron(x, num_nodes, num_input=1, num_output=1, activation=tf.nn.sigmoid):
    n_prev = num_input
    
    assert all(x == num_nodes[0] for x in num_nodes) #for a recurrent multilayer perceptron, the number of neurons in each hidden layer should be the same
    
    w_in = tf.get_variable("w_in",[n_prev, num_nodes[0]])
    b_in = tf.get_variable("b_in",[num_nodes[0]])
    
    w = tf.get_variable("w_shared",[num_nodes[0], num_nodes[0]])
    b = tf.get_variable("b_shared",[num_nodes[0]])
    
    for i in range(len(num_nodes)+1):
        if i==0:
            out = activation(tf.add(tf.matmul(x,w_in),b_in),name="out"+str(i))
        else:
            out = activation(tf.add(tf.matmul(out,w),b),name="out"+str(i))
        
    w_out = tf.get_variable("w_out",[num_nodes[0], num_output])
    b_out = tf.get_variable("b_out",[num_output])
    out = tf.add(tf.matmul(out,w_out),b_out,name="out_final")
    
    return out

'''
A class to organize methods that generate datasets for some of the experiments
'''
class Dataset():
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    
    @classmethod
    def generate_moons(cls, n, d=2, test_size=0.2, one_hot=False, normalize_x=False, noise=0):
        from sklearn.datasets import make_moons       
        assert (d%2==0),"d should be even"
        
        X, y = make_moons(n, noise=noise)
        
        if normalize_x:
            X = normalize(X)
                
        if (one_hot):
            y = y.reshape(-1,1)
            enc = cls.OneHotEncoder(n_values=2,sparse=False)
            y = enc.fit_transform(y)
        
        X_train, X_test, y_train, y_test = cls.train_test_split(X, y, test_size=test_size)
        
        return X_train, X_test, y_train, y_test

    
    @classmethod
    def generate_mixture_of_gaussians(cls, n, d, class_seps=[1], covariance_scale=1, test_size=0.2, one_hot=False, randomly_labeled=False, class_ratio=1, return_covariance=False, cov=None, resample=False, normalize_x=False):
        
        if len(class_seps)==d:
            pass
        elif len(class_seps)==1:
            class_seps = np.repeat(class_seps,d)
        else:
            raise ValueError("class_seps must be an array of length 1 or length d")
        
        if cov is None:
            c = covariance_scale*np.random.random((d,d))
            cov = c.T.dot(c)
        
        assert class_ratio>=1, "parameter: class_ratio must be greater than or equal to 1"
        n_pos = int(n/(class_ratio+1))
        n_neg = int(n-n_pos)
        X1 = np.random.multivariate_normal([0]*d, cov, size=n_pos)
        X2 = np.random.multivariate_normal(class_seps, cov, size=n_neg)
        if resample==True: #resamples the minority class
            X1 = np.tile(X1, (class_ratio, 1))
            n_pos = n_pos*class_ratio
        X = np.concatenate([X1,X2])
        
        if normalize_x:
            X = normalize(X)
        
        if randomly_labeled==True:
            y = np.random.randint(0,2,(n_pos+n_neg))
        else:
            y = np.array([0]*n_pos + [1]*n_neg)
        
        if (one_hot):
            y = y.reshape(-1,1)
            enc = cls.OneHotEncoder(n_values=2,sparse=False)
            y = enc.fit_transform(y)
        
        X_train, X_test, y_train, y_test = cls.train_test_split(X, y, test_size=test_size)
        
        if return_covariance:
            return X_train, X_test, y_train, y_test, cov
        return X_train, X_test, y_train, y_test
    
    def generate_MNIST(n_train, n_test, subset=range(10)):
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        y_train = mnist.train.labels
        
def pretty_plotting_styles():
    plt.rc("font",family="sans-serif",size=20)
    plt.rcParams["font.sans-serif"] = "Arial"



    
'''
Returns an RNN with the following parameters:
    window_size: the number of previous time_steps to use to make the prediction
    dim: dimensionality of the input data
    units: the number of hidden units in the LSTM
'''
def RNN(window_size=5, dim=1, units=32):
    import keras
    from keras.models import Model
    from keras.layers import Dense, Input, LSTM
        
    x = Input(shape=(window_size, dim))
    z, sh, sc = LSTM(units=units, return_state=True)(x)
    z = Dense(1, activation='tanh')(z)
    model = Model(inputs=[x],outputs=[z])
    model.compile(loss='mse', optimizer='adam')
    return model



'''
Converts a time-series into a form that can be used to train and validate an RNN
'''
def create_windowed_dataset(time_series, window_size=5, frac_train=0.8):
    time_series = normalize(time_series)
    X_train, y_train, X_test, y_test = [], [], [], []
    n = len(time_series)-window_size-1
    n_train = int(n*frac_train)
    for i in range(n):
        a = time_series[i:(i+window_size)]
        if a.ndim==1:
            a = a.reshape(-1, 1)
        if i < n_train: 
            X_train.append(a)
            y_train.append(time_series[i+window_size])
        else:
            X_test.append(a)
            y_test.append(time_series[i+window_size])
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def mse(y, y_):
    y = y.flatten()
    y_ = y_.flatten()
    assert len(y)==len(y_), "arrays must be of the same length"
    return np.round(np.sqrt(np.mean(np.square(y-y_))),2)

'''
Helper method to train and graph the results of RNN prediction
'''
def train_and_plot(time_series, window_sizes=None, hidden_units=None,epochs=20, figsize=None):
    plt.rc("font",family="sans-serif",size=14)
    
    if not(figsize is None):
        plt.figure(figsize=figsize)
    if hidden_units is None:
        if figsize is None:
            plt.figure(figsize=[4*len(window_sizes),4])
        for w, window_size in enumerate(window_sizes):
            plt.subplot(1, len(window_sizes), w+1)
            X_train, y_train, X_test, y_test = create_windowed_dataset(time_series, window_size=window_size)
            rnn = RNN(window_size=window_size)
            rnn.fit(X_train, y_train, epochs=epochs, verbose=0)
            y_ = rnn.predict(X_test)
            plt.plot(y_test)
            plt.plot(y_,marker='.')
            plt.title('Window size: '+str(window_size)+', RMSE: ' + str(mse(y_, y_test)))
    elif window_sizes is None:
        if figsize is None:
            plt.figure(figsize=[4*len(hidden_units),4])
        for h, hidden_unit in enumerate(hidden_units):
            plt.subplot(1, len(hidden_units), h+1)
            X_train, y_train, X_test, y_test = create_windowed_dataset(time_series)
            rnn = RNN(units=hidden_unit)
            rnn.fit(X_train, y_train, epochs=epochs, verbose=0)
            y_ = rnn.predict(X_test)
            plt.plot(y_test)
            plt.plot(y_,marker='.')
            plt.title('# Hidden Units: '+str(hidden_unit)+', RMSE: ' + str(mse(y_, y_test)))            
    else:
        if figsize is None:
            plt.figure(figsize=[4*len(window_sizes), 4*len(hidden_units)])
        count = 0 
        for w, window_size in enumerate(window_sizes):
            for h, hidden_unit in enumerate(hidden_units):
                count += 1
                plt.subplot(len(window_sizes), len(hidden_units), count)
                X_train, y_train, X_test, y_test = create_windowed_dataset(time_series, window_size=window_size)
                rnn = RNN(units=hidden_unit, window_size=window_size)
                rnn.fit(X_train, y_train, epochs=epochs, verbose=0)
                y_ = rnn.predict(X_test)
                plt.plot(y_test)
                plt.plot(y_,marker='.')
                plt.title('Window: '+str(window_size)+', Hidden: '+str(hidden_unit)+', RMSE: ' + str(mse(y_, y_test)))
    plt.legend(['Real','Predicted'])
    
def plot_decision_boundary(X, y, grid_pred):
    xx, yy = np.meshgrid(np.arange(0, 1.02, 0.02), np.arange(0, 1.02, 0.02))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    plt.scatter(*X.T, marker='.', c=np.argmax(y, axis=1), alpha=1, cmap='RdBu')
    zz = grid_pred[:,1].reshape(xx.shape)
    plt.contourf(xx, yy, zz, cmap='RdBu', alpha=.2)
    plt.xlim([0, 1]); plt.ylim([0,1])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    
    