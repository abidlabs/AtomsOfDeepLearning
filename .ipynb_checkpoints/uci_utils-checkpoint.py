from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pandas as pd, numpy as np
import warnings
from IPython.display import Markdown, display


class UCI_Dataset_Loader():
    @classmethod
    def adult(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        data=pd.read_csv(url, header=None, )
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        return features, labels

    @classmethod
    def car(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
        data=pd.read_csv(url, header=None, )
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        return features, labels
    
    @classmethod
    def credit_default(cls):
        try:
            import xlrd
        except:
            raise ImportError("To load this dataset, you need the library 'xlrd'. Try installing: pip install xlrd")
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
        data=pd.read_excel(url, header=1)
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        return features, labels  
    
    @classmethod
    def dermatology(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data"
        data=pd.read_csv(url, header=None, )
        features = data.iloc[:,1:]
        features = pd.get_dummies(features)
        labels = data.iloc[:,0]
        labels = labels.astype('category').cat.codes
        return features, labels
    
    @classmethod
    def diabetic_retinopathy(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00329/messidor_features.arff"
        data=pd.read_csv(url, skiprows=24, header=None)
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        return features, labels
    
    @classmethod
    def ecoli(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data"
        data=pd.read_csv(url, header=None, sep='\s+')
        features = data.iloc[:,1:-1]
        features = pd.get_dummies(features)
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        return features, labels    
    
    @classmethod
    def eeg_eyes(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
        data=pd.read_csv(url, skiprows=19, header=None, sep=',')
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        return features, labels        
    
    @classmethod
    def haberman(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
        data=pd.read_csv(url, skiprows=0, header=None, sep=',')
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        return features, labels            
    
    @classmethod
    def ionosphere(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
        data=pd.read_csv(url, skiprows=0, header=None, sep=',')
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        return features, labels                
    
    @classmethod
    def ionosphere(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
        data=pd.read_csv(url, skiprows=0, header=None, sep=',')
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        return features, labels                    
    
    @classmethod
    def mice_protein(cls):
        try:
            import xlrd
        except:
            raise ImportError("To load this dataset, you need the library 'xlrd'. Try installing: pip install xlrd")
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls"
        data=pd.read_excel(url, header=0, na_values=['', ' '])
        features = data.iloc[:,1:-4]
        features = features.fillna(value=0)
        features = pd.get_dummies(features)
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        return features, labels    
    
    @classmethod
    def nursery(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"
        data=pd.read_csv(url, header=0)
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        return features, labels                            
    
    @classmethod
    def seeds(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
        data=pd.read_csv(url, header=0, sep='\s+')
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        return features, labels          
    
    @classmethod
    def seismic(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff"
        data=pd.read_csv(url, skiprows=154, header=0, sep=',')
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        return features, labels              
    
    @classmethod
    def soybean(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-small.data"
        data=pd.read_csv(url, skiprows=0, header=0, sep=',')
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        return features, labels                  
    
    @classmethod
    def teaching_assistant(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/tae/tae.data"
        data=pd.read_csv(url, skiprows=0, header=0, sep=',')
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        return features, labels                      
    
    @classmethod
    def tic_tac_toe(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data"
        data=pd.read_csv(url, skiprows=0, header=0, sep=',')
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        return features, labels                          
    
    @classmethod
    def website_phishing(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00379/PhishingData.arff"
        data=pd.read_csv(url, skiprows=14, header=None, sep=',')
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        return features, labels                              
    
    @classmethod
    def wholesale_customers(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
        data=pd.read_csv(url, skiprows=0, header=0, sep=',')
        features = data.iloc[:,2:]
        features = pd.get_dummies(features)
        labels = data.iloc[:,1]
        labels = labels.astype('category').cat.codes
        return features, labels                                  
    
    

classifiers = [
    SVC(),
    GaussianNB(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(hidden_layer_sizes=(100)),
    MLPClassifier(hidden_layer_sizes=(100,100)),
    MLPClassifier(hidden_layer_sizes=(100,100,100)),]

names = [
    'Support Vector',
    'Naive Bayes',
    'Decision Tree',
    'Random Forests',
    '1-layer NN',
    '2-layer NN',
    '3-layer NN',
]

def print_stats(X_train, X_test, y_train, y_test):
    string = "Training set size: " + str(X_train.shape) + ", Test set size: " + str(X_test.shape) + ", \# of classes: " + str(len(np.unique(y_train)))
    display(Markdown(string))

def print_best(scores):
    eps = 1e-3
    best = np.max(scores)
    indices = np.where(scores > best - eps)[1]
    string = 'Best classifier: **'
    for i, idx in enumerate(indices):
        if i > 0:
            string += ', '
        string += names[idx]
    string += '**'
    display(Markdown(string))
    
all_data = list()

def compute_test_accuracies(X, y, train_size=0.8, verbose=1, append=True, iters=3):
    scores = np.zeros((iters,len(classifiers)))
    for i in range(iters):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore') #MLP throws annoying errors whenever it doesn't fully converge                     
            X_train, X_test, y_train, y_test =  train_test_split(X,y,train_size=train_size)
        if verbose>=1 and i==0:
            print_stats(X_train, X_test, y_train, y_test)
        for c, clf in enumerate(classifiers):
            if verbose>=2:
                print(names[c])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore') #MLP throws annoying errors whenever it doesn't fully converge                     
                clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            scores[i,c] = score
    scores = np.mean(scores,axis=0).reshape(1,-1)
    if append:
        n, d = X.shape
        c = len(np.unique(y))            
        all_data.append(np.concatenate([[[n, d, c]], scores], axis=1))
    return scores
    
def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    eps = 1e-3
    best = s.max()
    return ['background-color: #5fba7d' if v>best-eps else '' for v in s]

def highlight_max_excluding_first_three(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    eps = 1e-3
    best = s[3:].max()
    return ['background-color: #5fba7d' if (v>best-eps and i>3) else '' for i, v in enumerate(s)]
