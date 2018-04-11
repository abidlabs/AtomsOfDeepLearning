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
            num_steps=500,
            learning_rate = 0.003,
            verbose=False,
            load_covs = None,
            classify_grid = False,
            hidden_layer_sizes=(100,100),
            ratios = [1]):
        
        acc_matrix = np.zeros((len(ratios),iters))
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
                                                                             normalize_x=True,
                                                                             one_hot=True,
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
                                                                             normalize_x = True,
                                                                             one_hot=True,
                                                                             class_seps=class_seps, 
                                                                             covariance_scale=covariance_scale, 
                                                                             test_size=0,
                                                                             cov=cov)
                

                
                
                tf.reset_default_graph()
                x = tf.placeholder(dtype="float", shape=[None,d])
                y = tf.placeholder(dtype="float", shape=[None,2])
                y_ = multilayer_perceptron(x, num_input=d, num_output=2, num_nodes=hidden_layer_sizes)

                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
                correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
                accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train_op = optimizer.minimize(cross_entropy)
                init_op = tf.global_variables_initializer()

                with tf.Session() as sess:
                    sess.run(init_op)
                    y_preds = list()
                    for step in range(num_steps):
                        _, loss, acc, cp = sess.run([train_op, cross_entropy, accuracy_op, correct_prediction], feed_dict={x:X_train,y:y_train})
                        if (step%(num_steps/10)==0 and verbose):
                            print(loss, acc, cp)               
                
                    accuracy, y_pred = sess.run([accuracy_op, y_], feed_dict={x:X_test,y:y_test})
                    acc_matrix[r,k] = accuracy
                
                    if classify_grid:
                        xx, yy = np.meshgrid(np.arange(0, 1.02, 0.02), np.arange(0, 1.02, 0.02))
                        grid_points = np.c_[xx.ravel(), yy.ravel()]
                        grid_preds = sess.run(y_, feed_dict={x:grid_points})
            
        if classify_grid:
            return acc_matrix, saved_covs, X_train, y_train, X_test, y_test, y_pred, grid_preds
        return acc_matrix, saved_covs


'''
5. Are Neural Networks Memorizing Or Generalizing During Training?
'''
class Experiment5(Experiment):
   
    def __init__(self):
        pass
        
    def run(self,
            d = 6,
            iters = 3,
            covariance_scale = 1,
            test_size = 0.2,
            class_seps = [1 for i in range(6)],
            ns = [500],
            return_accuracy_per_epoch=False,
            randomize=False,
            verbose=False,
            learning_rate = 0.003,
            num_steps=2500,
            hidden_layer_sizes=(100,100)):
        
        if return_accuracy_per_epoch:
            acc = np.zeros((10, len(ns),iters))
        else:
            acc = np.zeros((len(ns),iters))      
        n_max = np.max(ns)
        
        for k in range(iters):
            X_train_, _, y_train_, _ = Dataset.generate_mixture_of_gaussians(n=n_max, 
                                                             d=d,
                                                             class_seps=class_seps, 
                                                             covariance_scale=covariance_scale, 
                                                             one_hot=True,
                                                             test_size=0)
            if randomize:
                y_train_ = np.random.permutation(y_train_)
                
            for n_i, n in enumerate(ns):
                step_multiple = 0
                tf.reset_default_graph()
                X_train = X_train_[:n]; y_train = y_train_[:n] 
                x = tf.placeholder(dtype="float", shape=[None,d])
                y = tf.placeholder(dtype="float", shape=[None,2])
                y_ = multilayer_perceptron(x, num_nodes=hidden_layer_sizes, num_input=d, num_output=2)

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
                        x_batch, y_batch = random_batch(X_train, y_train)
                        _, accuracy, y_pred = sess.run([train_op, accuracy_op, y_], feed_dict={x:x_batch,y:y_batch})
                        if (step%(num_steps/10)==0 and verbose):
                            print(accuracy)
                        if (step%(num_steps/10)==0 and return_accuracy_per_epoch):
                            accuracy = sess.run(accuracy_op, feed_dict={x:X_train,y:y_train})
                            acc[step_multiple, n_i, k] = accuracy 
                            step_multiple += 1

                    accuracy, loss, y_pred = sess.run([accuracy_op, loss_op, y_], feed_dict={x:X_train,y:y_train})
                
                if not(return_accuracy_per_epoch):
                    acc[n_i,k] = accuracy
        
        return acc
    
    
'''
## 6. Does Unsupervised Feature Reduction Help or Hurt?
'''
class Experiment6(Experiment):
   
    def __init__(self):
        pass
        
    def run(self,
            d = 10,
            iters = 3,
            covariance_scale = 0.2,
            test_size = 0.2,
            n = 100,
            dummy_dims = [0],
            pca_dims = [None],
            verbose=False,
            noise_level = 0,
            learning_rate = 0.003,
            num_steps=500,
            hidden_layer_sizes=(100,100)):
        
        from scipy.stats import special_ortho_group
        
        class_seps = [1 for i in range(d)]
        acc = np.zeros((iters, len(dummy_dims),len(pca_dims)))      
        
        for k in range(iters):            
            X_train_, X_test_, y_train, y_test = Dataset.generate_mixture_of_gaussians(n=n, 
                                                             d=d,
                                                             class_seps=class_seps, 
                                                             covariance_scale=covariance_scale, 
                                                             one_hot=True,
                                                             test_size=test_size)
            
            for d_i, dummy_dim in enumerate(dummy_dims):
                X_train = np.concatenate((X_train_, noise_level*np.random.random(size=(X_train_.shape[0], dummy_dim))),axis=1);
                X_test = np.concatenate((X_test_, noise_level*np.random.random(size=(X_test_.shape[0], dummy_dim))),axis=1);
                
                rotation_matrix = np.random.random(size=(d+dummy_dim,d+dummy_dim))
                X_train = X_train.dot(rotation_matrix)
                X_test = X_test.dot(rotation_matrix)

                
                for p_i, pca_dim in enumerate(pca_dims):
                    pca = PCA(n_components = pca_dim)
                    if not(pca_dim is None):
                        X_train = pca.fit_transform(X_train)
                        X_test = pca.transform(X_test)
                    if pca_dim is None:
                        pca_dim = d+dummy_dim
                    
                    tf.reset_default_graph()
                    x = tf.placeholder(dtype="float", shape=[None,pca_dim])
                    y = tf.placeholder(dtype="float", shape=[None,2])
                    y_ = multilayer_perceptron(x, num_nodes=hidden_layer_sizes, num_input=pca_dim, num_output=2)

                    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_))
                    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
                    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))

                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                    train_op = optimizer.minimize(loss_op)
                    init_op = tf.global_variables_initializer()

                    with tf.Session() as sess:
                        sess.run(init_op)
                        for step in range(num_steps):
                            _, accuracy, y_pred = sess.run([train_op, accuracy_op, y_], feed_dict={x:X_train,y:y_train})
                            if (step%(num_steps/10)==0 and verbose):
                                print(accuracy)

                        accuracy, loss, y_pred = sess.run([accuracy_op, loss_op, y_], feed_dict={x:X_test,y:y_test})
                    acc[k,d_i,p_i] = accuracy
        return acc

    
'''
7. Can Any Non-linearity Be Used As the Activation Function?
'''
class Experiment7(Experiment):
   
    def __init__(self):
        pass
        
    def run(self,
            iters = 1,
            d = 2,
            test_size = 0.2,
            n = 500,
            noise = 0.1,
            verbose=False,
            activations = [tf.nn.sigmoid, tf.square],
            learning_rate = 0.003,
            num_steps=800,
            hidden_layer_sizes=(30,30)):
        
        
        acc = np.zeros((iters, 10, len(activations)))      
        n_max = n
        
        for k in range(iters):                            
            X_train, X_test, y_train, y_test = Dataset.generate_moons(n=n_max, 
                                                                        test_size=0.2, 
                                                                        one_hot=True, 
                                                                        noise=noise)
            
            for a_i, a in enumerate(activations):
                step_counter = 0
                tf.reset_default_graph()
                x = tf.placeholder(dtype="float", shape=[None,d])
                y = tf.placeholder(dtype="float", shape=[None,2])
                y_ = multilayer_perceptron(x, num_nodes=hidden_layer_sizes, num_input=d, num_output=2, activation=a)

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
                        x_batch, y_batch = random_batch(X_train, y_train)
                        _, accuracy, y_pred = sess.run([train_op, accuracy_op, y_], feed_dict={x:x_batch,y:y_batch})
                        if (step%(num_steps/10)==0):
                            accuracy, loss, y_pred = sess.run([accuracy_op, loss_op, y_], feed_dict={x:X_test,y:y_test})
                            acc[k, step_counter, a_i] = accuracy
                            step_counter += 1
                            if verbose:
                                print(accuracy)


        
        return acc

'''
8. How Does Batch Size Affect the Results?
'''    
class Experiment8(Experiment):
   
    def __init__(self):
        pass
        
    def run(self,
            d = 12,
            iters = 3,
            covariance_scale = 1,
            test_size = 0.2,
            n = 500,
            batch_sizes = [32],
            return_accuracy_per_epoch=False,
            verbose=False,
            learning_rate = 0.003,
            num_epochs=150,
            store_every=10,
            hidden_layer_sizes=(100,100)):
        
        class_seps = [1 for i in range(12)]
        timer = Timer()
        if return_accuracy_per_epoch:
            acc = np.zeros((int(num_epochs/store_every)-1, len(batch_sizes),iters))
        else:
            acc = np.zeros((len(batch_sizes),iters))      
        
        runtimes = np.zeros((len(batch_sizes))) 
        for k in range(iters):
            X_train, X_test, y_train, y_test = Dataset.generate_mixture_of_gaussians(n=n, 
                                                             d=d,
                                                             class_seps=class_seps, 
                                                             covariance_scale=covariance_scale, 
                                                             one_hot=True,
                                                             test_size=test_size)
            for b_i, batch_size in enumerate(batch_sizes):
                timer.start()
                step_multiple = 0
                tf.reset_default_graph()
                x = tf.placeholder(dtype="float", shape=[None,d])
                y = tf.placeholder(dtype="float", shape=[None,2])
                y_ = multilayer_perceptron(x, num_nodes=hidden_layer_sizes, num_input=d, num_output=2)

                loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_))
                correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
                accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))

                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train_op = optimizer.minimize(loss_op)
                init_op = tf.global_variables_initializer()

                with tf.Session() as sess:
                    sess.run(init_op)
                    num_steps = int(num_epochs*n/batch_size)
                    store_acc_threshold = num_steps/num_epochs*store_every
                    for step in range(num_steps):
                        x_batch, y_batch = random_batch(X_train, y_train, size=batch_size)
                        _, accuracy, y_pred = sess.run([train_op, accuracy_op, y_], feed_dict={x:x_batch,y:y_batch})
                        if (step%(num_steps/num_epochs)==0 and verbose):
                            print(accuracy)
                        if (step>store_acc_threshold and return_accuracy_per_epoch):
                            accuracy = sess.run(accuracy_op, feed_dict={x:X_train,y:y_train})
                            acc[step_multiple, b_i, k] = accuracy 
                            step_multiple += 1
                            store_acc_threshold += num_steps/num_epochs*store_every

                    accuracy, loss, y_pred = sess.run([accuracy_op, loss_op, y_], feed_dict={x:X_test,y:y_test})
                
                if not(return_accuracy_per_epoch):
                    acc[b_i,k] = accuracy # otherwise, this is stored earlier
                runtimes[b_i] = timer.end()
                
        return runtimes, acc


    
'''
9. How Does the Loss Function Matter?
'''
class Experiment9(Experiment):
   
    def __init__(self):
        pass
        
    def run(self,
            d = 12,
            iters = 1,
            covariance_scale = 1,
            test_size = 0.2,
            n = 500,
            randomize=False,
            verbose=False,
            loss_functions = ['cross_entropy', 'mean_squared_error'],
            learning_rate = 0.003,
            num_steps=500,
            hidden_layer_sizes=(100,100)):
        
        class_seps = [1/(i+1) for i in range(d)]
        acc = np.zeros((iters, 10, len(loss_functions)))      
        n_max = n
        LOSS_FUNCTIONS = ['cross_entropy',
                  'mean_abs_error',
                  'mean_squared_error',
                  'mean_fourth_pow_error', 
                  'hinge_loss', 
                  'constant']
        
        for k in range(iters):
            X_train, X_test, y_train, y_test = Dataset.generate_mixture_of_gaussians(n=n_max, 
                                                             d=d,
                                                             class_seps=class_seps, 
                                                             covariance_scale=covariance_scale, 
                                                             one_hot=True)
            if randomize:
                y_train_ = np.random.permutation(y_train_)
                    

            for l_i, l in enumerate(loss_functions):
                step_counter = 0
                if not(l in LOSS_FUNCTIONS):
                    raise ValueError("Valid loss functions are " + str(LOSS_FUNCTIONS))

                tf.reset_default_graph()
                x = tf.placeholder(dtype="float", shape=[None,d])
                y = tf.placeholder(dtype="float", shape=[None,2])
                y_ = multilayer_perceptron(x, num_nodes=hidden_layer_sizes, num_input=d, num_output=2)

                if l=='cross_entropy':
                    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_))
                elif l=='mean_squared_error':
                    loss_op = tf.reduce_mean(tf.square(y_ - y))
                elif l=='mean_abs_error':
                    loss_op = tf.reduce_mean(tf.abs(y_ - y))
                elif l=='hinge_loss':
                    loss_op = tf.losses.hinge_loss(labels=y, logits=y_)
                elif l=='mean_fourth_pow_error':
                    loss_op = tf.reduce_mean(tf.pow(y_ - y, 4))
                elif l=='constant':
                    loss_op = 0*tf.reduce_mean(tf.square(y_ - y))

                correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
                accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))

                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train_op = optimizer.minimize(loss_op)
                init_op = tf.global_variables_initializer()

                with tf.Session() as sess:
                    sess.run(init_op)
                    for step in range(num_steps):
                        x_batch, y_batch = random_batch(X_train, y_train)
                        _, accuracy, y_pred = sess.run([train_op, accuracy_op, y_], feed_dict={x:x_batch,y:y_batch})
                        if (step%(num_steps/10)==0):
                            accuracy, loss, y_pred = sess.run([accuracy_op, loss_op, y_], feed_dict={x:X_test,y:y_test})
                            acc[k, step_counter, l_i] = accuracy
                            step_counter += 1
                            if verbose:
                                print(accuracy)
        
        return acc

'''
10. How Does the Initialization Affect Performance?
'''
class Experiment10(Experiment):
   
    def __init__(self):
        pass
        
    def run(self,
            d = 12,
            iters = 1,
            covariance_scale = 1,
            test_size = 0.2,
            n = 500,
            randomize=False,
            verbose=False,
            initializers = [tf.contrib.layers.xavier_initializer()],
            learning_rate = 0.003,
            num_steps=500,
            hidden_layer_sizes=(100,100)):
        
        class_seps = [1/(i+1) for i in range(d)]
        acc = np.zeros((iters, 10, len(initializers)))
        
        for k in range(iters):
            X_train, X_test, y_train, y_test = Dataset.generate_mixture_of_gaussians(n=n, 
                                                             d=d,
                                                             class_seps=class_seps, 
                                                             covariance_scale=covariance_scale, 
                                                             one_hot=True)
                    
            for i_i, initializer in enumerate(initializers):    
                step_counter = 0
                tf.reset_default_graph()
                x = tf.placeholder(dtype="float", shape=[None,d])
                y = tf.placeholder(dtype="float", shape=[None,2])
                y_ = multilayer_perceptron(x, num_nodes=hidden_layer_sizes, num_input=d, num_output=2, initializer=initializer)

                loss_op = tf.reduce_mean(tf.square(y_ - y))                       
                correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
                accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))

                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train_op = optimizer.minimize(loss_op)
                init_op = tf.global_variables_initializer()

                with tf.Session() as sess:
                    sess.run(init_op)
                    for step in range(num_steps):
                        x_batch, y_batch = random_batch(X_train, y_train)
                        _, accuracy, y_pred = sess.run([train_op, accuracy_op, y_], feed_dict={x:x_batch,y:y_batch})
                        if (step%(num_steps/10)==0):
                            accuracy, loss, y_pred = sess.run([accuracy_op, loss_op, y_], feed_dict={x:X_test,y:y_test})
                            acc[k, step_counter, i_i] = accuracy
                            step_counter += 1
                            if verbose:
                                print(accuracy)

                    accuracy, loss, y_pred = sess.run([accuracy_op, loss_op, y_], feed_dict={x:X_test,y:y_test})

        
        return acc

'''
11. Do Weights in Different Layers Evolve At Different Speeds?
'''
class Experiment11(Experiment):
   
    def __init__(self):
        pass
        
    def run(self,
            d = 12,
            covariance_scale = 1,
            test_size = 0.2,
            n = 500,
            store_every=2,
            randomize=False,
            verbose=False,
            learning_rate = 0.003,
            num_steps=500,):
        
        class_seps = [1/(i+1) for i in range(d)]
        hidden_layer_sizes=(50,50,50)
        weights = []
        accs = []
        
        X_train, X_test, y_train, y_test = Dataset.generate_mixture_of_gaussians(n=n, 
                                                         d=d,
                                                         class_seps=class_seps, 
                                                         covariance_scale=covariance_scale, 
                                                         one_hot=True)

        step_counter = 0

        tf.reset_default_graph()
        x = tf.placeholder(dtype="float", shape=[None,d])
        y = tf.placeholder(dtype="float", shape=[None,2])
        y_, wts = multilayer_perceptron(x, num_nodes=hidden_layer_sizes, num_input=d, num_output=2, return_weight_tensors=True)

        loss_op = tf.reduce_mean(tf.square(y_ - y))                       
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            for step in range(num_steps):
                x_batch, y_batch = random_batch(X_train, y_train)
                _, accuracy, y_pred = sess.run([train_op, accuracy_op, y_], feed_dict={x:x_batch,y:y_batch})
                if (step%2==0):                        
                    accuracy, w0, w1, w2, w3 = sess.run([accuracy_op, wts[0], wts[1], wts[2], wts[3]], feed_dict={x:X_test,y:y_test})
                    weights.append([w0, w1, w2, w3])
                    accs.append(accuracy)

                    if verbose:
                        print(accuracy)

        return weights, accs
    
'''
12. How Does Regularization Affect Weight Evolution?
'''
class Experiment12(Experiment):
   
    def __init__(self):
        pass
        
    def run(self,
            d = 12,
            covariance_scale = 1,
            test_size = 0.2,
            n = 500,
            regularization_type = 'L2',
            regularization_strength = 0,
            store_every=2,
            randomize=False,
            verbose=False,
            learning_rate = 0.003,
            num_steps=500,):
        
        class_seps = [1/(i+1) for i in range(d)]
        hidden_layer_sizes=(50,50)
        weights = []
        accs = []
        
        X_train, X_test, y_train, y_test = Dataset.generate_mixture_of_gaussians(n=n, 
                                                         d=d,
                                                         class_seps=class_seps, 
                                                         covariance_scale=covariance_scale, 
                                                         one_hot=True)

        step_counter = 0

        tf.reset_default_graph()
        x = tf.placeholder(dtype="float", shape=[None,d])
        y = tf.placeholder(dtype="float", shape=[None,2])
        y_, wts = multilayer_perceptron(x, num_nodes=hidden_layer_sizes, num_input=d, num_output=2, return_weight_tensors=True)
        
        if regularization_type=='L2':
            loss_op = tf.reduce_mean(tf.square(y_ - y)) + regularization_strength*(tf.reduce_mean(tf.square(wts[0])) + tf.reduce_mean(tf.square(wts[1])) + tf.reduce_mean(tf.square(wts[2])))
        elif regularization_type=='L1':
            loss_op = tf.reduce_mean(tf.square(y_ - y)) + regularization_strength*(tf.reduce_mean(tf.abs(wts[0])) + tf.reduce_mean(tf.abs(wts[1])) + tf.reduce_mean(tf.abs(wts[2])))
        else:
            raise ValueError("regularization_type must be 'L1' or 'L2'")
            
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            for step in range(num_steps):
                x_batch, y_batch = random_batch(X_train, y_train)
                _, accuracy, y_pred = sess.run([train_op, accuracy_op, y_], feed_dict={x:x_batch,y:y_batch})
                if (step%2==0):                        
                    accuracy, w0, w1, w2 = sess.run([accuracy_op, wts[0], wts[1], wts[2]], feed_dict={x:X_test,y:y_test})
                    weights.append([w0, w1, w2])
                    accs.append(accuracy)

                    if verbose:
                        print(accuracy)

        return weights, accs
