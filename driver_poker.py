# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 20:51:33 2018

@author: Shruti Agrawal (sxa178830), Abhisek Banerjee (axb180050)
"""

from DecisionTree import *
import pandas as pd
from sklearn import model_selection
from sklearn import datasets
import random


header = ['S1', 'C1', 'S2', 'C2', 'S3' , 'C3' , 'S4' , 'C4' , 'S5' , 'C5' , 'Class']
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data', header=None, names=['S1', 'C1', 'S2', 'C2', 'S3' , 'C3' , 'S4' , 'C4' , 'S5' , 'C5' , 'Class'])
lst = df.values.tolist()
t = build_tree(lst, header)
print_tree(t)

# Function to prune the tree by pruning one layer of the leaf nodes at a time.
# The function takes the num of layer to prune as an input
def prune_layer(t, num_layers_to_prune):
    t_pruned = t
    node_list_to_prune = []
    i = 1
    acc = []
    while (i <= num_layers_to_prune):
        innerNodes = getInnerNodes(t_pruned)
        for node_to_be_pruned in innerNodes:
            if (isinstance(node_to_be_pruned.true_branch, Leaf) and isinstance(node_to_be_pruned.false_branch, Leaf)):
                node_list_to_prune.append(node_to_be_pruned.id)
        t_pruned = prune_tree(t, node_list_to_prune)
        acc.append(computeAccuracy(test, t_pruned))
        i = i + 1
    return (t_pruned, acc)

print("********** Leaf nodes ****************")
leaves = getLeafNodes(t)
max_count = 0
for leaf in leaves:
    label_to_predict = ""
    max_count = 0 
    for label in leaf.predictions:
        if max_count < leaf.predictions[label]:
            max_count = leaf.predictions[label]
            label_to_predict = label
    print("id = " + str(leaf.id) + " depth = " + str(leaf.depth) + " Label = " + str(label_to_predict))
print("********** Non-leaf nodes ****************")

innerNodes = getInnerNodes(t)

for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))

trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header)
print("*************Tree before pruning*******")
print_tree(t)
train_acc = computeAccuracy(train, t)
print("Accuracy on train data is = "+ "%0.1f"%(train_acc*100) + "%")
test_acc = computeAccuracy(test, t)
print("Accuracy on test data is = " + "%0.1f"%(test_acc*100) + "%")

print("*************Tree after pruning*******")
acc_pruned = []
t_pruned, pruned_acc = prune_layer(t, 12)
print_tree(t_pruned)
i=1
for a in pruned_acc:      
    print("Accuracy of test data on pruned tree level "+ str(i) + " is = " + "%0.1f"%(a*100) + "%")
    i = i + 1

    




