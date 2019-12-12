'''
Authors: Ashwani Kashyap, Anshul Pardhi
'''

from DecisionTree import *
import pandas as pd
from sklearn import model_selection

# default data set
df = pd.read_csv('data_set/Social_Network_Ads.csv')
header = list(df.columns)

# overwrite your data set here
# header = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']
# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])
# data-set link: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/
# df = pd.read_csv('data_set/breast-cancer.csv')


lst = df.values.tolist()

# splitting the data set into train and test
trainDF, testDF = model_selection.train_test_split(lst, test_size=0.2)

# building the tree
t = build_tree(trainDF, header)

# get leaf and inner nodes
print("\nLeaf nodes ****************")
leaves = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))

print("\nNon-leaf nodes ****************")
innerNodes = getInnerNodes(t)

for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))

# print tree
maxAccuracy = computeAccuracy(testDF, t)
print("\nTree before pruning with accuracy: " + str(maxAccuracy*100) + "\n")
print_tree(t)

# TODO: You have to decide on a pruning strategy
# Pruning strategy
nodeIdToPrune = -1
for node in innerNodes:
    if node.id != 0:
        prune_tree(t, [node.id])
        currentAccuracy = computeAccuracy(testDF, t)
        print("Pruned node_id: " + str(node.id) + " to achieve accuracy: " + str(currentAccuracy*100) + "%")
        # print("Pruned Tree")
        # print_tree(t)
        if currentAccuracy > maxAccuracy:
            maxAccuracy = currentAccuracy
            nodeIdToPrune = node.id
        t = build_tree(trainDF, header)
        if maxAccuracy == 1:
            break
    
if nodeIdToPrune != -1:
    t = build_tree(trainDF, header)
    prune_tree(t, [nodeIdToPrune])
    print("\nFinal node Id to prune (for max accuracy): " + str(nodeIdToPrune))
else:
    t = build_tree(trainDF, header)
    print("\nPruning strategy did'nt increased accuracy")

print("\n********************************************************************")
print("*********** Final Tree with accuracy: " + str(maxAccuracy*100) + "%  ************")
print("********************************************************************\n")
print_tree(t)
