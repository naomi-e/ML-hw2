import numpy as np
np.random.seed(42)

chi_table = {0.01  : 6.635,
             0.005 : 7.879,
             0.001 : 10.828,
             0.0005 : 12.116,
             0.0001 : 15.140,
             0.00001: 19.511}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity of the dataset.    
    """
    gini = 0.0
    ###########################################################################
    labels = np.unique(data[:,-1])
    print("labels:")
    print(labels)
    rows, columns = data.shape
    num_of_examples = rows
    sum = 0
    
    for label in labels:
        labeled_data = data[np.where(data[:,-1] == label)]
        rows, _ = labeled_data.shape
        sum += ((rows / num_of_examples)**2)
        print(sum)
     
    gini = 1 - sum                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.    
    """
    entropy = 0.0
    ###########################################################################
    labels = np.unique(data[:,-1])
    rows, columns = data.shape
    num_of_examples = rows
    
    sum = 0
    
    for label in labels:
        labeled_data = data[np.where(data[:,-1] == label)] 
        rows, _ = labeled_data.shape
        Si_over_S = rows / num_of_examples
        sum += Si_over_S * (np.log2(Si_over_S))
    
    entropy = -sum
                                        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

# calculates gain based on the impurity measure given
def calc_gain(impurity_measure, data):
    
    gain = 0.0
   
    if impurity_measure == 'gini':
        gain = calc_gini(data)
    else:
        gain = calc_entropy(data)
        
    return gain
    
    
class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic 
    # functionality as described in the notebook. It is highly recommended that you 
    # first read and understand the entire exercise before diving into this class.
    
    def __init__(self, feature, value):
        self.feature = feature # column index of criteria being tested
        self.value = value # value necessary to get a true result
        
    def add_child(self, node):
        self.children.append(node)


def build_tree(data, impurity_measure):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure. 

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """
    root = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    #initialize our queue to hold the entire data set
    root = DecisionNode(data)
    nodes_queue = queue.Queue()
    nodes_queue.put(root)

    #while there are items in our queue
    while( ! nodes_queue.empty() )
    
        node = nodes_queue.get()
        best_gain = node.impurity #start off the best gain as the current one
        
        for attribute in node.attributes
            for threshold in thresholds
                this_gain = node.impurity - calc_gain(node.data, impurity)
                
        if best_gain == node.impurity 
           #if we didn't find any attribute+threshold that improves our gain - what to do?
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return root

    

def predict(node, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    pred = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pred

def calc_accuracy(node, dataset):
    """
    calculate the accuracy starting from some node of the decision tree using
    the given dataset.

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy

def print_tree(node):
    '''
    prints the tree according to the example in the notebook

	Input:
	- node: a node in the decision tree

	This function has no return value
	'''

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################    
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
