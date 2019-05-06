import _pickle as pickle
import pandas as pd
import engineer
from csv import reader


def load_csv(fname):
    # Initialiaze a dataset as a list
    dataset = list()
    # open the file in read mode
    with open(fname, 'r') as f:
        csv_reader = reader(f)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

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



def bagging_predict(trees, row):
    predictions = [predict(tree,row) for tree in trees]
    return max(set(predictions), key=predictions.count)





# test_df = pd.read_csv('../../data/test-20.csv')
# data = engineer.engineering(test_df)
# data = data.drop(['id','question1','question2','text1_nostop','text2_nostop','text1_lower','text2_lower','word_overlap'],axis=1)
filename = 'test.csv'
predicion_data = load_csv(filename)
print(predicion_data)

with open('../trees.obj','rb') as f:
    trees = pickle.load(f)

predictions = [bagging_predict(trees, row) for row in predicion_data]
print(predictions)
