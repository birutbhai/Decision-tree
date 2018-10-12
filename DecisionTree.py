"""
@author: Shruti Agrawal (sxa178830), Abhisek Banerjee (axb180050)
"""

from __future__ import print_function
import  math

def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])

#######
# Demo:
# unique_vals(training_data, 0)
# unique_vals(training_data, 1)
#######


def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

#######
# Demo:
# class_counts(training_data)
#######




def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)

#######
# Demo:
# is_numeric(7)
# is_numeric("Red")
#######


class Question:
    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value, header):
        self.column = column
        self.value = value
        self.header = header

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
           # print(str(val) + "********" + str(self.value) )
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            self.header[self.column], condition, str(self.value))


def partition(rows, question):
    """Partitions a dataset.

    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    """Calculate the Gini Impurity for a list of rows.

    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


def entropy(rows):
    counts = class_counts(rows)
    
    entropy = 0
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        if prob_of_lbl == 0:
            continue
        entropy -= prob_of_lbl * math.log(prob_of_lbl, 2)
    return entropy
               

def info_gain(left, right, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))

    
    return current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)


def find_best_split(rows, header):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    n_features = len(rows[0]) - 1  # number of columns
    current_uncertainty = entropy(rows)

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val, header)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows, id, depth):
        self.predictions = class_counts(rows)
        self.id = id
        self.depth = depth



class Decision_Node:
    '''
    A Decision Node asks a question.
    This holds a reference to the question, and to the two child nodes.
    '''
    def __init__(self, question, true_branch, false_branch, depth, id, rows, pruned=0):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.depth = depth
        self.id = id
        self.rows = rows
        self.pruned = pruned





def build_tree(rows, header, depth=0, id=0):
    """Builds the tree.

    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """
    # depth = 0
    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.

    gain, question = find_best_split(rows, header)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows, id, depth)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    # nodeLst.append(id)
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows, header, depth + 1, 2 * id + 2 )

    # Recursively build the false branch.
    false_branch = build_tree(false_rows, header, depth + 1, 2 * id + 1)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # depending on on the answer.
    return Decision_Node(question, true_branch, false_branch, depth, id, rows)


def prune_tree(node, prunedList):
    """Builds the tree.

    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node
    # If we reach a pruned node, make that node a leaf node and return. Since it becomes a leaf node, the nodes
    # below it are automatically not considered
    if int(node.id) in prunedList:
        return Leaf(node.rows, node.id, node.depth)

    # Call this function recursively on the true branch
    node.true_branch = prune_tree(node.true_branch, prunedList)

    # Call this function recursively on the false branch
    node.false_branch = prune_tree(node.false_branch, prunedList)

    return node


def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    # Compare the number of counts for each label at a leaf to find the classified label
    if isinstance(node, Leaf):
        label_to_predict = ""
        max_count = 0
        for label in node.predictions:
            if max_count < node.predictions[label]:
                max_count = node.predictions[label]
                label_to_predict = 	label			
        return label_to_predict

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + "id " + str(node.id))
        print(spacing + "Depth " + str(node.depth))
        print (spacing + "Predict", node.predictions)
        label_to_predict = ""
        max_count = 0 
        for label in node.predictions:
            if max_count < node.predictions[label]:
                max_count = node.predictions[label]
                label_to_predict = label
        print (spacing + "Label", label_to_predict)
        return

    # Print the question at this node
    print (spacing + str(node.question))
    print (spacing + "id", node.id)
    print (spacing + "Depth", node.depth)
    # Print the depth of this node
    # print (spacing + str(node.depth))
	# Print the id of this node
	# print (spacing + str(node.id))
    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

def bfs (node, innerNodes =[], leafNodes=[]):
    temp_nodes=[]
    temp_nodes.append(node)
    while temp_nodes:
        temp_node  = temp_nodes.pop(0)
        if (isinstance(temp_node, Leaf)) :
            leafNodes.append(temp_node)
        else:
            temp_nodes.append(temp_node.true_branch)
            temp_nodes.append(temp_node.false_branch)
            innerNodes.append(temp_node)
 
    return innerNodes, leafNodes

def getLeafNodes(node):
    leafNodes =[]
    # Returns a list of all leaf nodes of a tree
    innerNodes = []
    innerNodes, leafNodes = bfs(node, innerNodes, leafNodes)  

    return leafNodes


def getInnerNodes(node):
    innerNodes =[]
    # Returns a list of all non-leaf nodes of a tree
    leafNodes = []
    innerNodes, leafNodes = bfs(node, innerNodes, leafNodes)
    return innerNodes




def computeAccuracy(rows, node):
    totalRows = len ( rows )
    numAccurate = 0
    for row in rows:
        if row[-1] == classify(row, node):
            numAccurate = numAccurate + 1
    return round(numAccurate/totalRows , 2)

