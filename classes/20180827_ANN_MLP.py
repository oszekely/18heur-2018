
# coding: utf-8

# # Multi Layer Perceptron Neural Network  (MLP ANN) - Heuristics on weights hyperparameters
# 

# In this notebook we would like to test whether discussed heuristic approaches could be applied on MLP for weights hyperparameters estimation without training the network.

# ### Objective function description
# As an objective function for a heuristic approach we use a loss function of the MLP. We defined 2 loss functions - **Mean Square Error** and **Cross Entropy Loss**.

# #### Mean Square Loss
# $$C(w,b)=\frac{1}{2n}\sum_x||y(x)-a||^2 + \frac{\lambda}{2n}\sum_w||w||^2$$
# where:
# * $w$ is a MLP weights tensor
# * $b$ is a MLP bias tensor
# * $n$ is a number of data records in training dataset
# * $x$ is a particular datum record
# * $y(x)$ is a class membership vector predicted by MLP
# * $a$ is a ground truth memebership vector
# * $\lambda$ is a regularization term coeficient
# 
# Implementation of the loss function is located in `src/heur_aux.py`, class `MSRLoss`. In does not inherit interface from `ObjFun` because based on the logic of ANN, it makes sence to implement core common MLP primitives into class `ANNMLPClassifier`, located in `src/objfun_ann_mlp.py` and use `MSRLoss` as its attribute.

# #### Cross Entropy Loss
# $$C(w,b)=-[y(x)\mathrm{ln}(a)+(1-y(x))\mathrm{ln}(1-a)] + \frac{\lambda}{2n}\sum_w||w||^2$$
# where:
# * $w$ is a MLP weights tensor
# * $b$ is a MLP bias tensor
# * $n$ is a number of data records in training dataset
# * $x$ is a particular datum record
# * $y(x)$ is a class membership vector predicted by MLP
# * $a$ is a ground truth memebership vector
# * $\lambda$ is a regularization term coeficient
# 
# Implementation of the loss function is located in `src/heur_aux.py`, class `CrossEntropyLoss`. In does not inherit interface from `ObjFun` because based on the logic of ANN, it makes sence to implement core common MLP primitives into class `ANNMLPClassifier`, located in `src/objfun_ann_mlp.py` and use `CrossEntropyLoss` as its attribute.

# More loss functions could be added by creating a new particular class with reguired interface.

# ### Activation Function
# We use **Sigmoid** activation function (class `SigmoidFunction` in a `src/heur_aux.py`) which has following shape
# <img src="img/sigmoid.png">
# More activation functions could be added by creating a new particular class with reguired interface.
# 

# ### Used dataset
# Our implementation can work with any datasets. It is desired that data matrix has shape ($m$,$n$) where $m$ is records number and $n$ is features count (float). Labels for each class has to be only integers - shape($m$,1), starts with 0 and incremented 1 by 1. Objective function automatically divide datasets into training, testing split and automatically shuffled.
# 
# In this notebook we use **Iris** dataset.
# 

# In[125]:


# Import path to source directory (bit of a hack in Jupyter)
import sys
import os
pwd = get_ipython().magic(u'pwd')
sys.path.append(os.path.join(pwd, os.path.join('..', 'src')))

# Ensure modules are reloaded on any change (very useful when developing code on the fly)
get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

# Import extrenal librarires
import numpy as np
import matplotlib
get_ipython().magic(u'matplotlib notebook')
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from tqdm import tqdm_notebook


# In[126]:


from sklearn import datasets

iris_dataset = datasets.load_iris()

iris_features = iris_dataset.data[:, :]
iris_labels = iris_dataset.target


# Iris features:

# In[127]:


iris_dataset["feature_names"]


# Number of records:

# In[128]:


iris_features.shape[0]


# Number of classes:

# In[129]:


np.unique(iris_labels).shape[0]


# ### MLP Training
# As a first step we need to train MLP to obtain $f^*$. For training it is implemented Stochastic Gradient Descent. Training is performed automatically when constructor of the MLP objective function is called.

# In[130]:


# Import our code
from objfun_ann_mlp import ANNMLPClassifier


# We need to set MLP parameters. Now we test MLP without any hidden neurons.

# In[131]:


neurons_hidden_layers = []
number_of_epochs = 2000
batch_size_sgd = 20
learning_rate = 0.01
reg_lambda = 0.01
loss_function = "MSR"
activation_function = "sigmoid"
features = iris_features
labels = iris_labels
training_data_size_percentage_split = 0.8


# **Note:**
# * `batch_size_sgd` should be lower, equal to number of **training records**. 
# * `loss_function` can take values `MSR` or `cross-entropy`.
# * `activation_function` can take values `sigmoid`.

# In[132]:


mlp = ANNMLPClassifier(neurons_hidden_layers, number_of_epochs, batch_size_sgd, 
                         learning_rate, reg_lambda, loss_function, activation_function, 
                         features, labels, training_data_size_percentage_split)


# Lets print final epoch statistics:

# In[133]:


mlp.trainingStatusInfo[-1]


# We see that this dataset could be handled without any hidden layer, so we could add shallow hidden layers to demo purposes. We add 10 hidden neurons in one hidden layer.

# In[134]:


neurons_hidden_layers = [5]
number_of_epochs = 2000
batch_size_sgd = 20
learning_rate = 0.1
reg_lambda = 0.01
loss_function = "MSR"
activation_function = "sigmoid"
features = iris_features
labels = iris_labels
training_data_size_percentage_split = 0.8


# In[135]:


mlp = ANNMLPClassifier(neurons_hidden_layers, number_of_epochs, batch_size_sgd, 
                         learning_rate, reg_lambda, loss_function, activation_function, 
                         features, labels, training_data_size_percentage_split)


# Lets print final epoch statistics:

# In[ ]:


mlp.trainingStatusInfo[-1]


# We can see that one hidden layer increased the precision of the classifier. We can experiment with different shape of hidden layers and MLP training parameters.

# ### MLP (MSE loss function) Heuristics
# 
# Based on the previous training we obtained the $f^*$ which corresponds to the loss for **testing** data in the last epoch.

# In[136]:


mlp.get_fstar()


# There is a question whether the domain of weights should be bounded or not. During the training there are no bounds for the weights.

# numberOfLayers = len(mlp.weightsTensor)
# 
# for tensorInd in range(0,numberOfLayers-1):
#     plt.figure(tensorInd)
#     plt.title("Hidden Layer {}".format(tensorInd+1))
#     plt.hist(mlp.weightsTensor[tensorInd].flatten(), bins='auto')
#     plt.show()
#     
# plt.figure(numberOfLayers-1)
# plt.title("Output Layer")
# plt.hist(mlp.weightsTensor[tensorInd].flatten(), bins='auto')
# plt.show()

# We can see that weights values are bounded and have shape like normal distribution. It is caused by fact that normal distribution was used during weights initialization. 
# 
# We set weights bounds to $\left[\lfloor{\mathrm{min}(w)\rfloor}, \lceil{\mathrm{max}(w)\rceil}\right] $ across all weights elements

# In[137]:


mlp.get_bounds()


# Lets see the dimension of the weights

# In[138]:


mlp.weightsDim


# #### Shoot and Go

# Lets use **Shoot and Go** to perform a heuristics over weights. Because the search is very time consuming, we will use smaller number of runs per parameter value.
# 
# We take into count the high dimension of the task. Thus, we will increase the number of evaluations per heuristic run.

# In[139]:


maxeval = 5000
NUM_RUNS = 300


# In[140]:


from heur_sg import ShootAndGo


# In[141]:


def experiment_sg(of, maxeval, num_runs, hmax, random_descent):
    method = 'RD' if random_descent else 'SD'
    results = []
    for i in tqdm_notebook(range(num_runs), 'Testing method={}, hmax={}'.format(method, hmax)):
        result = ShootAndGo(of, maxeval=maxeval, hmax=hmax, random_descent=random_descent).search() # dict with results of one run
        result['run'] = i
        result['heur'] = 'SG_{}_{}'.format(method, hmax) # name of the heuristic
        result['method'] = method
        result['hmax'] = hmax
        results.append(result)
    
    return pd.DataFrame(results, columns=['heur', 'run', 'method', 'hmax', 'best_x', 'best_y', 'neval'])


# In[142]:


table_sg = pd.DataFrame()
    
for hmax in [0, 1, 2, 5, 10, 20, 50, np.inf]:
    res = experiment_sg(of=mlp, maxeval=maxeval, num_runs=NUM_RUNS, hmax=hmax, random_descent=False)
    table_sg = pd.concat([table_sg, res], axis=0)


# In[143]:


table_sg.head()


# In[144]:


def mean_loss(x):
    return np.mean(x)

def std_dev(x):
    return np.std(x)


# In[146]:


stats_sg = table_sg.pivot_table(
    index=['heur'],
    values=['best_y'],
    aggfunc=(mean_loss, std_dev)
)['best_y']
stats_sg = stats_sg.reset_index()


# In[148]:


stats_sg


# In[150]:


stats_sg.sort_values(by=['mean_loss'])


# #### Testing performance on classification

# In[160]:


maxeval = 5000
heuristicsRes = ShootAndGo(mlp, maxeval=maxeval, hmax=0, random_descent=False).search()


# In[164]:


print('SG training data loss: {}'.format(heuristicsRes['best_y']))


# Lets compare it with loss of SGD after first epoch and in the last.

# In[165]:


print('SGD training data loss (epoch 1): {}'.format(mlp.trainingStatusInfo[0]["loss_training"]))


# In[166]:


print('SGD training data loss (epoch {}): {}'.format(len(mlp.trainingStatusInfo), mlp.trainingStatusInfo[-1]["loss_training"]))


# We can see that the loss value of SG is lower then the loss in the first epochs of SGD.

# Lets check precision:

# In[170]:


print('GD training data precision: {}'.format(mlp.getTrainingDataPrecision(heuristicsRes['best_x'])))


# In[172]:


print('SGD training data precision: {}'.format(mlp.trainingStatusInfo[-1]["precision_testing"]))


# In[173]:


print('GD testing data precision: {}'.format(mlp.getTestingDataPrecision(heuristicsRes['best_x'])))


# In[174]:


print('SGD training data precision: {}'.format(mlp.trainingStatusInfo[-1]["precision_testing"]))


# #### Fast Simulated Annealing

# Lets use **Fast Simulated Annealing** to perform a heuristics over weights with Cauchy Mutation. Because the search is very time consuming, we will use smaller number of runs per parameter value.
# 
# We take into count the high dimension of the task. Thus, we will increase the number of evaluations per heuristic run.

# In[175]:


maxeval = 5000
NUM_RUNS = 300


# In[176]:


from heur_fsa import FastSimulatedAnnealing
from heur_aux import Correction, CauchyMutation


# In[177]:


def experiment_fsa(of, maxeval, num_runs, T0, n0, alpha, r):
    results = []
    for i in tqdm_notebook(range(num_runs), 'Testing T0={}, n0={}, alpha={}, r={}'.format(T0, n0, alpha, r)):
        mut = CauchyMutation(r=r, correction=Correction(of))
        result = FastSimulatedAnnealing(of, maxeval=maxeval, T0=T0, n0=n0, alpha=alpha, mutation=mut).search()
        result['run'] = i
        result['heur'] = 'FSA_{}_{}_{}_{}'.format(T0, n0, alpha, r) # name of the heuristic
        result['T0'] = T0
        result['n0'] = n0
        result['alpha'] = alpha
        result['r'] = r
        results.append(result)
    
    return pd.DataFrame(results, columns=['heur', 'run', 'T0', 'n0', 'alpha', 'r', 'best_x', 'best_y', 'neval'])


# In[179]:


table_fsa = pd.DataFrame()

for T0 in [1e-10, 1e-2, 1, np.inf]:
    res = experiment_fsa(of=mlp, maxeval=maxeval, num_runs=NUM_RUNS, T0=T0, n0=1, alpha=2, r=0.5)
    table_fsa = pd.concat([table_fsa, res], axis=0)


# In[180]:


table_fsa.head()


# In[181]:


def mean_loss(x):
    return np.mean(x)

def std_dev(x):
    return np.std(x)


# Because the task is continuous in all parameters, it does not make sense to measure number of evaluations to find $f^*$. Instead we measure the mean loss and standard deviation to check how the loss function cnverge.

# In[184]:


stats_fsa = table_fsa.pivot_table(
    index=['heur'],
    values=['best_y'],
    aggfunc=(mean_loss, std_dev)
)['best_y']
stats_fsa = stats_fsa.reset_index()


# In[185]:


stats_fsa


# Based on the statistics we see that the loss function convergence does not depend on the initial temperature. On the other hand we can see that infinite value is not suitable for the task.
# 
# Leets investigae the parameters space for $T_0=1$

# #### Analysis
# 
# **Can we improve the best configuration ($T_0=1$)?**

# In[186]:


table_fsa = pd.DataFrame()
NUM_RUNS = 300

for alpha in [1, 2, 4]:
    for cooling_par in [1, 2, 4]:
        res = experiment_fsa(of=mlp, maxeval=maxeval, num_runs=NUM_RUNS, T0=1, n0=cooling_par, alpha=alpha, r=0.5)
        table_fsa = pd.concat([table_fsa, res], axis=0)


# Lets compare the results with the results in SGD in the first epoch and in the last one.

# In[187]:


stats_fsa = table_fsa.pivot_table(
    index=['heur'],
    values=['best_y'],
    aggfunc=(mean_loss, std_dev)
)['best_y']
stats_fsa = stats_fsa.reset_index()


# In[188]:


stats_fsa.sort_values(by=['mean_loss'])


# Based on the results it seems that results are quite same. Thus we decide to use $\alpha=2$ and $n_0=2$

# #### Testing performance on classification

# In[189]:


maxeval = 5000
heuristicsRes = FastSimulatedAnnealing(mlp, maxeval=maxeval, T0=1, n0=2, alpha=2, mutation=CauchyMutation(r=0.5, correction=Correction(mlp))).search()


# In[190]:


print('FSA training data loss: {}'.format(heuristicsRes['best_y']))


# Lets compare it with loss of SGD after first epoch and in the last.

# In[191]:


print('SGD training data loss (epoch 1): {}'.format(mlp.trainingStatusInfo[0]["loss_training"]))


# In[192]:


print('SGD training data loss (epoch {}): {}'.format(len(mlp.trainingStatusInfo), mlp.trainingStatusInfo[-1]["loss_training"]))


# We can see that the loss value of FSA is lower then the loss in the first epochs of SGD.

# Lets check precision:

# In[193]:


print('FSA training data precision: {}'.format(mlp.getTrainingDataPrecision(heuristicsRes['best_x'])))


# In[194]:


print('SGD training data precision: {}'.format(mlp.trainingStatusInfo[-1]["precision_training"]))


# In[195]:


print('FSA testing data precision: {}'.format(mlp.getTestingDataPrecision(heuristicsRes['best_x'])))


# In[196]:


print('SGD testing data precision: {}'.format(mlp.trainingStatusInfo[-1]["precision_testing"]))


# We can see that we achieve quite similar performance as for SGD approach.

# ### Cross Entropy loss function Heuristics

# We tested FSA on MSE loss function. Lets try the Cross Entropy function to check whether the algorithm performs in similar way.

# In[197]:


neurons_hidden_layers = [5]
number_of_epochs = 2000
batch_size_sgd = 20
learning_rate = 0.1
reg_lambda = 0.01
loss_function = "cross-entropy"
activation_function = "sigmoid"
features = iris_features
labels = iris_labels
training_data_size_percentage_split = 0.8


# In[198]:


mlp = ANNMLPClassifier(neurons_hidden_layers, number_of_epochs, batch_size_sgd, 
                         learning_rate, reg_lambda, loss_function, activation_function, 
                         features, labels, training_data_size_percentage_split)


# In[199]:


heuristicsRes = FastSimulatedAnnealing(mlp, maxeval=maxeval, T0=1, n0=2, alpha=2, mutation=CauchyMutation(r=0.5, correction=Correction(mlp))).search()


# In[200]:


print('FSA training data loss: {}'.format(heuristicsRes['best_y']))


# Lets compare it with loss of SGD after first epoch and in the last.

# In[201]:


print('SGD training data loss (epoch 1): {}'.format(mlp.trainingStatusInfo[0]["loss_training"]))


# In[202]:


print('SGD training data loss (epoch {}): {}'.format(len(mlp.trainingStatusInfo), mlp.trainingStatusInfo[-1]["loss_training"]))


# We can see that the loss value of FSA is lower then the loss in the first epochs of SGD.

# Lets check precision:

# In[203]:


print('FSA training data precision: {}'.format(mlp.getTrainingDataPrecision(heuristicsRes['best_x'])))


# In[204]:


print('SGD training data precision: {}'.format(mlp.trainingStatusInfo[-1]["precision_training"]))


# In[205]:


print('FSA testing data precision: {}'.format(mlp.getTestingDataPrecision(heuristicsRes['best_x'])))


# In[206]:


print('SGD training data precision: {}'.format(mlp.trainingStatusInfo[-1]["precision_testing"]))

