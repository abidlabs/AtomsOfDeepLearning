import tensorflow as tf
import numpy as np
import time    
import warnings
from utils import *
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

'''
 All of the Experiments, starting from a base class
'''
class Experiment():
    def __init__(self):
        pass
    
    #ideally, this would make experiments completely reproducible, but because jobs are distributed over multiple cores, small differences may persist in practice
    def initialize(self, seed=0, fix_seed=True):
        if fix_seed:
            np.random.seed(seed)
            tf.set_random_seed(seed)
        self.timer = Timer()
        self.timer.start()

    def conclude(self):
        self.timer.end_and_print()
'''
Experiment 1: Why do we use neural networks?
Description: Performs regression using a neural network with 1 hidden layer and different number of units. Returns the original x-values, true y-values, and predicted y-values, along with the MSE loss.
'''
class Experiment1(Experiment):
    def __init__(self):
        pass
        
    def run(self,
            n_hidden = 2,
            learning_rate = 0.003,
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
            n=16, 
            n_hidden=[10],
            num_steps=15000,
            learning_rate = 0.003,
            verbose=False,
            recurrent=True):
        
        
        x_values = np.linspace(0,1-1/n,n).reshape(-1,1)
        y_values = np.resize([[0,1],[1,0]], (n,2))
        
        tf.reset_default_graph()
        x = tf.placeholder(dtype="float", shape=[None,1])
        y = tf.placeholder(dtype="float", shape=[None,2])

        if recurrent:
            y_ = recurrent_multilayer_perceptron(x, num_input=1, num_output=2, num_nodes=n_hidden,activation=tf.nn.relu)
        else:
            y_ = multilayer_perceptron(x, num_input=1, num_output=2, num_nodes=n_hidden,bias=bias,activation=tf.nn.relu)

        
        n_params = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.global_variables()])        
        #show_graph(tf.get_default_graph().as_graph_def())
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_))
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        #loss_op = tf.reduce_mean(tf.square(y_ - y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            for step in range(num_steps):
                x_batch, y_batch = random_batch(x_values, y_values)
                _, loss, y_pred = sess.run([train_op, loss_op, y_], feed_dict={x:x_batch,y:y_batch})
                if (step%(num_steps/10)==0 and verbose):
                    print(loss)
        
            accuracy, loss, y_pred = sess.run([accuracy_op, loss_op, y_], feed_dict={x:x_values,y:y_values})

        return x_values.squeeze(), y_values.squeeze(), y_pred.squeeze(), loss, accuracy, n_params

    
'''
Experiment 3: Does More Data Favor Deeper Neural Networks?
'''
class Experiment3(Experiment):
    def __init__(self):
        pass
        
    def run(self,
            classifiers, 
            d = 12,
            class_seps = [1],
            ns = np.logspace(2,4,10),
            iters = 3,
            covariance_scale = 1,
            test_size = 0.2,
            accuracy_on = 'test',
            recurrent=True):        
        
        acc = np.zeros((len(ns),len(classifiers),iters))
        n_max = int(np.max(ns))

        for k in range(iters):
        
            X_train, X_test, y_train, y_test = Dataset.generate_mixture_of_gaussians(n=n_max, 
                                                                                     d=d,
                                                                                     class_seps=class_seps, 
                                                                                     covariance_scale=covariance_scale, 
                                                                                     test_size=test_size)
            for i, n in enumerate(ns):
                for j, clf in enumerate(classifiers):
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore') #MLP throws annoying errors whenever it doesn't fully converge                     
                        n_train = int(n*(1-test_size))
                        clf.fit(X_train[:n_train],y_train[:n_train]) #choose a subset of the training data
                        if accuracy_on=='train':
                            acc[i,j,k] = clf.score(X_train[:int(n*(1-test_size))],y_train[:int(n*(1-test_size))])
                        elif accuracy_on=='test':
                            acc[i,j,k] = clf.score(X_test,y_test)
                        else:
                            raise ValueError("accuracy_on must be 'test' or 'train'") 

        return acc


'''
Experiment 4: Does Unbalanced Data Hurt Neural Networks?
'''
class Experiment4(Experiment):
   
    def __init__(self):
        pass
        
    def run(self,
            d = 12,
            iters = 3,
            covariance_scale = 1,
            test_size = 0.2,
            resample=False,
            n = 1200,
            load_covs = None,
            hidden_layer_sizes=(100,100),
            ratios = [1]):
        
        acc = np.zeros((len(ratios),iters))
        class_seps = [1/(i+1) for i in range(d)]
        clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
        saved_covs = []
        
        counter = 0
        for k in range(iters):
            for r, ratio in enumerate(ratios):
                # load covariance matrices for reproducibility
                if load_covs is None:
                    cov = None
                else:
                    cov = load_covs[counter]
                counter += 1
                    
                    
                X_train, _, y_train, _, cov = Dataset.generate_mixture_of_gaussians(n=n, 
                                                                             d=d,
                                                                             class_seps=class_seps, 
                                                                             covariance_scale=covariance_scale, 
                                                                             test_size=0,
                                                                             cov = cov,
                                                                             class_ratio=ratio,
                                                                             resample=resample,
                                                                             return_covariance=True)
                saved_covs.append(cov)
                X_test, _, y_test, _ = Dataset.generate_mixture_of_gaussians(n=int(n/4), 
                                                                             d=d,
                                                                             class_seps=class_seps, 
                                                                             covariance_scale=covariance_scale, 
                                                                             test_size=0,
                                                                             cov=cov)
                

                clf.fit(X_train, y_train) 
                acc[r,k] = clf.score(X_test, y_test)
                
        return acc, saved_covs


'''
## 7. Does Unsupervised Feature Reduction Help or Hurt?
'''
class Experiment7(Experiment):
   
    def __init__(self):
        pass
        
    def run(self,
            d = 100,
            iters = 3,
            covariance_scale = 0.2,
            test_size = 0.2,
            class_seps = [1 for i in range(100)],
            n = 100,
            load_covs = None,
            hidden_layer_sizes=(100,100),
            reduced_dims = [100]):
        
        acc = np.zeros((len(reduced_dims),iters))
        
        clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
        saved_covs = []        
        for k in range(iters):
            if load_covs is None:
                cov = None
            else:
                cov = load_covs[k]
                
            X_train, X_test, y_train, y_test, cov = Dataset.generate_mixture_of_gaussians(n=n, 
                                                             d=d,
                                                             class_seps=class_seps, 
                                                             covariance_scale=covariance_scale, 
                                                             test_size=0.2,
                                                             cov = cov,
                                                             return_covariance=True)

            saved_covs.append(cov)
            
            for rd, reduced_dim in enumerate(reduced_dims):
                pca = PCA(n_components=reduced_dim)
                X_train_reduced = pca.fit_transform(X_train)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore') #MLP throws annoying errors whenever it doesn't fully converge
                    clf.fit(X_train_reduced, y_train)
                X_test_reduced = pca.transform(X_test)
                acc[rd,k] = clf.score(X_test_reduced, y_test)
        
        return acc, saved_covs
