import queue
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
    
    if data.shape[0] == 1:
        print("calc_gini: data")
        print(data)
        print(type(data))
        return gini
    if data.shape[0] < 1:
        print("calc_gini: data")
        print(data)
        print(type(data))
        raise Exception
        
    labels = np.unique(data[:,-1])
    #print("labels:")
    #print(labels)
    rows, columns = data.shape
    num_of_examples = rows
    sum = 0
    
    for label in labels:
        labeled_data = data[np.where(data[:,-1] == label)]
        rows, _ = labeled_data.shape
        sum += ((rows / num_of_examples)**2)
        #print(sum)
     
    gini = 1 - sum                                           
   
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
    if data.shape[0] <= 1:
        print("calc_entropy: data ")
        print(data)
        print(type(data))
        return entropy
    
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

    
class NodeExpansion:
    
    def __init__(self, decision_node, impurity_of_node, data):
        self.decision_node = decision_node
        self.impurity_of_node = impurity_of_node
        self.data = data
        #TODO: set label to be majority of the data's classification
        
        
class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic 
    # functionality as described in the notebook. It is highly recommended that you 
    # first read and understand the entire exercise before diving into this class.
    
    
    def __init__(self, feature, value):
        self.feature = feature # column index of criteria being tested
        self.value = value # value necessary to get a true result. the may be a function (boolean state X>Y)
        self.label = None
        self.is_root = False 
        self.children = []
    
    class DecisionError(Exception):
        def __init__(self):
            self.string = "No Child fits this instance!" 
    
    
    def add_child(self, node):
        self.children.append(node)
    
    def set_as_root():
        self.is_root = True
    
    def set_class(self, label):
        self.label = label
        
    def make_desicion(self, instance):
        if (len(self.children) is 0):
            return self.label
        
        for child in self.children:
            if (child.value(instance[child.feature])):
                return (child.make_desicion(instance))

        print("No Child fits this instance")
        raise DecisionError
                

def weighted_impurity(impurity_func, nun_instances_part, nun_instances_full, data_part):
        weight = nun_instances_part/nun_instances_full
        #print("call impurity_func from weighted_impurity")
        
        part_impurity = impurity_func(data_part)
        return (weight*part_impurity)
    
# calculates gain based on the impurity measure given
def calc_gain(impurity_func, attribute, data, TH, impurity_value_of_the_father):
     
    #values = np.unique(data[:,attribute])
    #labeled_data = data[np.where(data[:,-1] == label)] 
    data_over_TH  = data[np.where(data[:,attribute].astype(int) > TH)] 
    data_under_TH = data[np.where(data[:,attribute].astype(int) <= TH)] 
   
    if ((data_over_TH.shape[0] < 1) or (data_over_TH.shape[1] < 1)):
        print("data_over_TH empty! TH=")
        print(TH)
        raise Exception
        return 0
    
    if ((data_under_TH.shape[0] < 1) or (data_under_TH.shape[1] < 1)):
        print("data_under_TH empty! TH=")
        print(TH)
        raise Exception
        return 0
    
    num_instances_over_TH, _ = data_over_TH.shape
    num_instances_under_TH, _ = data_under_TH.shape
    num_instances, _ = data.shape
    
    weighted_impurity_over_TH = (weighted_impurity(impurity_func, num_instances_over_TH, num_instances, data_over_TH))
    weighted_impurity_under_TH = (weighted_impurity(impurity_func, num_instances_under_TH, num_instances, data_under_TH))
    
    return (impurity_value_of_the_father - (weighted_impurity_over_TH + weighted_impurity_under_TH))
    
                
def build_tree(data, impurity):
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
                
    #initialize our queue to hold the root node
    # TODO: set our root node (set_as_root), and set its value and feature
    nodes_queue = queue.Queue()
    nodes_queue.put(NodeExpansion(DecisionNode(None, None), impurity(data), data))

    #while there are items in our queue
    while( not nodes_queue.empty() ):
        
        # TODO: we need to be able to set the feature and value of the ROOT NODE 
                
        node = nodes_queue.get()
        if node.data.shape[0]<=1:
            print("STOP CONDITION REACHED 1")
            print(len(node.data) == 0)
            node.decision_node.set_class(node.data[0][-1])
            continue
            
        if node.impurity_of_node == 0: #stop condition
            print("STOP CONDITION REACHED 2")
            print(node.data.shape[0])
            node.decision_node.set_class(node.data[0][-1]) 
            continue
                
        best_gain = 0#node.impurity_of_node #start off the best gain as the current one
        best_attribute = 0
        best_threshold = 0
                
        #num_attributes = data.shape[1]
        num_attributes = node.data.shape[1]
        
        #try splitting by each attribute at this node
        for attribute in range (num_attributes):
                
                #for each attribute test each threshold
                thresholds = get_thresholds(node.data, attribute)
                for threshold in thresholds:

                    this_gain = calc_gain(impurity, attribute, node.data, threshold, node.impurity_of_node)

                    if this_gain > best_gain:
                        
                        best_gain = this_gain
                        best_attribute = attribute
                        best_threshold = threshold
       
    
        print("Got here!")
        print("best_attribute: %d" %(best_attribute))
        print("best_threshold: %d" %(best_threshold))

        child_left, child_right = set_children(best_attribute, best_threshold, node.data, impurity)
        
        if (child_left is not None):
            node.decision_node.add_child(child_left)
            nodes_queue.put(child_left)
            print("Adding child_left:")
            print(child_left)
            
        if (child_right is not None):
            node.decision_node.add_child(child_right)
            nodes_queue.put(child_right)
            print("Adding child_right:")
            print(child_right)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return root
 
def set_children(attribute, threshold, data, impurity):
    """
    sets two new child nodes for a given attribute, threshold and data
    
    Input: attribute, threshold, data
    Output: two NodeExpansion nodes: one for values under the threshold and one above it.
    """  
    
    print("set_children 1")
    left_dn = DecisionNode(attribute, data[:,attribute].astype(int) <= threshold)
    right_dn = DecisionNode(attribute, data[:,attribute].astype(int) > threshold)
    
    print("set_children 2")
    left_data = data[np.where(data[:,attribute].astype(int) >= threshold)]
    right_data = data[np.where(data[:,attribute].astype(int) < threshold)]
    
     
    print("set_children 3")
    print (type(data))
    print (data.shape)
    print ((left_data))
    print ((right_data))
    if ((left_data.shape[0] < 1) or (left_data.shape[1] < 1)):
        print("DATA IS LESS THAN 1")
        left_impurity = 0
        return (None, None)
    else: 
        left_impurity = impurity(left_data)

    print("set_children 4")    
    if ((right_data.shape[0] < 1) or (right_data.shape[1] < 1)):
        print("DATA IS LESS THAN 1")
        right_impurity = 0
        return (None, None)
    else: 
        right_impurity = impurity(right_data)
    
    
    print("set_children 5")
    if ((left_impurity == 0) or (right_impurity == 0)):
        print("STOP CONDITION REACHED 3")
        #print(node.data.shape)
  
    print("set_children 6")
    child_left = NodeExpansion(left_dn, left_impurity, left_data)
    child_right = NodeExpansion(right_dn, right_impurity, right_data)
                
    return child_left, child_right
                
def get_thresholds(data, attribute):
    """
    Get each unique value for this attribute
    Organize them in size order, and then get the median between every two points
    
    Input: data and attribute
    Output: array of thresholds for deciding on this attribute
    """
   # print(data)
    values = np.unique(data[:,attribute])
    #print(values)
    avarage = ((np.append ([values],[0])) + (np.append([0],[values])))/2 
   # print(avarage)
    avarage = np.delete(avarage, 0)
  #  print(avarage)
    thresholds = np.delete(avarage, -1)
    
    # TODO: Implement the function.  
    print("values:")
    print(values)
    
    print("thresholds:")
    print(thresholds)
    return thresholds

    
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
