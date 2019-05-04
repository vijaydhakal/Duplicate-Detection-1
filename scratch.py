#!/usr/bin/env python
# coding: utf-8

# # Importing the initial dependencies

# In[1]:


from random import seed
from random import randrange
from csv import reader
from math import sqrt


# ## Data Loading Helper Function 

# In[2]:


#Loading the csv files
def load_csv(fname):
    # Initialiaze a dataset as a list
    dataset = list()
    #open the file in read mode
    with open(fname, 'r') as f:
        csv_reader = reader(f)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# In[5]:


import pandas as pd

filename = 'all49_1.csv'
data = pd.read_csv(filename)
data = data.drop(['id','question1','question2','text1_nostop','text2_nostop','text1_lower','text2_lower','word_overlap'],axis=1)
data.to_csv('final.csv', index = False)
filename = 'final.csv'
dataset = load_csv(filename)
print(dataset[0],end='\n')
for i in range(len(dataset)):
    dataset[i].append(dataset[i][0])
    dataset[i].remove(dataset[i][0])
print(dataset[0])


# In[6]:


def str_column_to_float(dataset,columns):
    for row in dataset:
        for column in columns:
            row[column] = float(row[column])


# ## Decision tree algorithm helper Functions

# In[7]:


def cross_validation_split(dataset,n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset)/ n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def test_split(index,value,dataset):
    left,right = list(),list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def accuracy_metric(actual,predicted):
    correct = 0 
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual))* 100.0

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set,[])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set,test_set,*args)
        actual = [row[-1]  for row in fold]
        accuracy = accuracy_metric(actual,predicted)
        scores.append(accuracy)   
    return scores


# Caluculate the gini index for a split dataset as an cost
# function used to evaluate the split

def gini_index(groups,class_values):
    # class_values contains the final label . i,e 0 and 1 in our case
    # groups is a type tuple containing the left and the right node for of a parent node
    
    gini =0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))
    return gini
                     
# Select the best split point for a dataset

def get_split(dataset, n_features):
    class_values = list(set([row[-1] for row in dataset]))
    
#     class_values.remove('is_duplicate')
#     class_values= list()
#     for i in x:
#         if x not in class_values:
#             class_values.append(x) 
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset[0])-1)
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataset:
            # When selecting the best split and using it as a new node for the tree 
            # we will store the index of the chosen attribute, the value of that attribute 
            # by which to split and the two groups of data split by the chosen split point.
            ## Each group of data is its own small dataset of just those rows assigned to the 
            # left or right group by the splitting process. You can imagine how we might split 
            # each group again, recursively as we build out our decision tree.
            groups = test_split(index,row[index],dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index,b_value, b_score, b_groups = index, row[index],gini, groups
                     
    
    return {'index':b_index, 'value': b_value,'groups':b_groups}
        
# Create the terminal node value

def to_terminal(group):
    out = [row[-1] for row in group]
    outcomes = list()
    for x in out:
        if x not in outcomes:
            outcomes.append(x)
    return max(outcomes, key=outcomes.count)
                     
                     
#Create child splits for a node or make terminal
#Building a decision tree involves calling the above developed get_split() function over 
#and over again on the groups created for each node.
#New nodes added to an existing node are called child nodes. 
#A node may have zero children (a terminal node), one child (one side makes a prediction directly) 
#or two child nodes. We will refer to the child nodes as left and right in the dictionary representation 
#of a given node.
#Once a node is created, we can create child nodes recursively on each group of data from 
#the split by calling the same function again.
                     
def split(node, max_depth, min_size, n_features,depth):
    #Firstly, the two groups of data split by the node are extracted for use and 
    #deleted from the node. As we work on these groups the node no longer requires access to these data.
    left, right = node['groups']
    del(node['groups'])
    
    # Check whether left or right group of rows are empty 
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return 
    if depth >= max_depth:
        node['left'] , node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) < min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left,n_features)
        split(node['left'],max_depth,min_size,n_features, depth+1)
                     
    if len(right) < min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right,n_features)
        split(node['right'],max_depth,min_size,n_features, depth+1)
                     
                     
def build_tree(train, max_depth , min_size, n_features):
    # Creating root node
    root = get_split(train,n_features)

    # Calling the split method that calls itself recursively and forms a tree
    split(root, max_depth, min_size, n_features, 1)
    return root
                     
# Make predictions with a decision tree
                     
def predict(node,row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'],dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'],dict):
            return predict(node['right'],row)
        else:
            return node['right']

                     
# Creating random subsample form the dataset with replacement

                     
def subsample(dataset,ratio):
    sample = list()
    n_sample  = round(len(dataset)* ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return list(sample)


        
        
    


# # Main Code
# 

# In[8]:


# Make a prediction with the list of bagged trees responsible for making a prediction with each decicion tree and combining thee predictions into a single return value 

def bagging_predict(trees, row):
    predictions = [predict(tree,row) for tree in trees]
    return max(set(predictions), key=predictions.count)


# Random Forest Algorithm 

def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth,min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return(predictions)


seed(1)
# Load and prepare forest algorithm 

# Load and prepare data

n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_features = int(sqrt(len(dataset[0])-1))

for n_trees in [25]:
    scores = evaluate_algorithm(dataset, random_forest,n_folds,max_depth,min_size,sample_size,n_trees,n_features)
    print(f'Trees : {n_trees}')
    print(f'Scores: {scores}')
    print(f'Mean Accuracy: {sum(scores)/float(len(scores))}')
        
        
        
        
        
        
        
        
        
        


# In[13]:


list(set([row[-1] for row in dataset]))


# In[ ]:




