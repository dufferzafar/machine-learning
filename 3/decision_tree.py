from collections import deque
from functools import partial

import numpy as np

from tqdm import tqdm

from read_data import ATTRIBUTES, ATTRIBUTES_NUMERICAL
from common import accuracy


def attribute_is_numerical(split_attr):
    """Check if an attribute is numerical in the Rich/Poor dataset."""
    return ATTRIBUTES[split_attr] in ATTRIBUTES_NUMERICAL


class Node():

    """Node of a tree."""

    def __init__(self, parent, nsamples, split_attr=None, children=[]):

        # These stored at all nodes - leaf / internal
        self.parent = parent

        # No. of samples of different classes coming at this node
        self.nsamples = nsamples

        # Majority class of the data coming in at this node
        # (this is used to make predictions)
        self.cls = nsamples.argmax()

        # These are only stored on internal (decision nodes)

        # Index of the attribute that this nodes makes decision on
        self.split_attr = split_attr

        # Value of the splitting attribute
        # This is stored at all nodes except root
        self.split_value = None

        # If split_attr is numerical, this will store the median value
        self.median_value = None

        # A list child nodes
        self.children = children

    def __repr__(self):
        if self.children:
            r = (self.cls, len(self.children), self.split_attr)
            r = "<Node: cls=%r, children=%r, split_attr=%r" % r

            if self.split_value:
                r += ", split_value=%r" % self.split_value
            if self.median_value:
                r += ", median=%r" % self.median_value

            r += ">"
            return r
        else:
            return "<Leaf: cls=%r>" % self.cls


def entropy(Y):
    """
    Calculate entropy from class labels.
    """
    probabilities = np.bincount(Y) / len(Y)
    return -1 * sum(p * np.log2(p) for p in probabilities if p)


def partition(Xa, binarize=False):
    """
    Partition a column based on the unique values it takes.

    { value: [indices where that value occurs in the column] }
    """
    if binarize:
        Xa = (Xa >= np.median(Xa)).astype(int)
    return {v: np.where(Xa == v)[0] for v in np.unique(Xa)}


class DecisionTree():

    def __init__(self, data=None, binarize_median=True):
        self.binarize_median = binarize_median

        # This allows us to create a tree with no data - for testing purposes
        if data is None:
            self.root = None
        else:
            self.root = self._build_tree(data)

    def _build_tree(self, data, parent=None):
        """
        Build a decision tree using ID3 / information gain.

        First column (data[:, 0]) is the output class.
        """

        Y = data[:, 0]
        nsamples = np.bincount(Y)

        # if data is "pure" i.e has examples of a single class
        # then return a leaf node predicting that class
        if len(set(Y)) <= 1:
            return Node(parent, nsamples)

        # Find the attribute that maximizes the gain
        gain, split_attr = self._best_attribute(data)

        # print("Splitting on:", ATTRIBUTES[split_attr],
        #       "Data len ", data[:, split_attr])

        # Split if gain is positive
        # Does this is result in pre-pruned trees?
        if gain > 0:

            this_node = Node(parent, nsamples, split_attr, children=[])

            # There's some code duplicacy here - _best_attribute has exact same line
            binarize = not self.binarize_median and attribute_is_numerical(split_attr)

            if binarize:
                this_node.median_value = np.median(data[:, split_attr])

            partitions = partition(data[:, split_attr], binarize).items()

            # This handles the case when splitting on attributes after binarizing
            # there may be a single partition which is not pure (wrt class label)
            # NOTE: Not too sure about this.
            if binarize and len(partitions) == 1:
                return this_node

            # Create children of this node
            for val, part in partitions:

                child = self._build_tree(data[part], parent=this_node)
                child.split_value = val

                this_node.children.append(child)

            return this_node

        # Otherwise create a leaf node that predicts the majority class
        else:
            return Node(parent, nsamples)

    def _best_attribute(self, data):
        """
        Use information gain to decide which attribute to split on.
        """

        # Need to find these parameters
        best_gain = -1
        best_attr = -1

        Y = data[:, 0]   # First column is label
        X = data[:, 1:]  # All other columns are data

        # Iterate over each attribute
        for i, Xa in enumerate(X.T):

            # If the data was not already binarized on medians
            # It will contain numerical attributes that need to be binarized now
            binarize = not self.binarize_median and attribute_is_numerical(i)

            # Create partitions over this attribute
            entropy_Y_Xa = sum((len(p) / len(Xa)) * entropy(Y[p])
                               for p in partition(Xa, binarize).values())

            gain = entropy(Y) - entropy_Y_Xa

            # NOTE: In case of a tie, Does this choose the attribute which
            # appears first in the ordering as given in the training data.
            if gain > best_gain:
                best_gain = gain

                # NOTE: +1 because the data contains output variables at 1st column
                # so attributes/features start from 2nd column
                best_attr = i + 1

        return best_gain, best_attr

    def _predict(self, dtree, x):
        """Predict a single example using dtree."""
        if not dtree.children:
            return dtree.cls
        else:
            # Value of the record at attribute on which split will happen
            x_val = x[dtree.split_attr]

            # If this tree is splitting on a median value
            if dtree.median_value is not None:
                x_val = int(x_val >= dtree.median_value)

            # Decide which child to go next to based on split values
            children = {c.split_value: c for c in dtree.children}
            child = children.get(x_val)

            # If there isn't a correct outgoing edge
            # then just return the majority class
            if not child:
                return dtree.cls
            else:
                return self._predict(child, x)

    def score(self, data):
        """Find accuracy of dtree over data."""
        predictions = [self._predict(self.root, x) for x in data]
        return 100 * accuracy(data[:, 0], predictions)

    def height(self):
        return self._height(self.root)

    @staticmethod
    def _height(dtree):
        if not dtree.children:
            return 0
        else:
            return 1 + max(map(DecisionTree._height, dtree.children))

    def node_count(self):
        return self._node_count(self.root)

    @staticmethod
    def _node_count(dtree):
        if not dtree.children:
            return 1
        else:
            return 1 + sum(map(DecisionTree._node_count, dtree.children))

    @staticmethod
    def _remove_node(node):
        parent = node.parent

        # Root can not be removed
        if parent is None:
            return

        parent.children.remove(node)

        return parent

    # There is a discrepancy in how the tree was built
    # and how it is being iterated upon; fix?
    def nodes(self):
        """Iterate over all nodes in the tree in BFS order."""

        q = deque([self.root])

        while q:
            node = q.popleft()
            q.extend(node.children)

            yield node

    def prune_single_pass(self, valid_data):
        """Prune by making a single pass over data."""

        nodes = list(self.nodes())
        nodes.reverse()

        # Iteate over all nodes and decide whether to keep this or not.
        for node in nodes:

            # No point in checking a leaf node
            if not node.children:
                continue

            # Using majority class, this fraction of data is incorrectly classified
            node_err_rate = min(node.nsamples) / sum(node.nsamples)

            children_err_rate = sum(
                min(child.nsamples) / sum(node.nsamples)
                for child in node.children
            )

            # Creating children is increasing the error
            if children_err_rate > node_err_rate:

                # Remove the subtree rooted at this node
                # and make this node a leaf
                node.children = []

    def prune_brute(self, train_data, test_data, valid_data):
        """Prune by Brute-force - calculating accuracy before and after removing a node."""

        nodecounts = []
        accuracies = {"train": [], "test": [], "valid": []}

        nodes = list(self.nodes())
        nodes.reverse()

        # Iteate over all nodes and decide whether to keep this or not.
        for node in tqdm(nodes, ncols=80, ascii=True):

            # No point in checking a leaf node
            if not node.children:
                continue

            # Accuracy before removing the node
            val_acc_before = self.score(valid_data)

            # Remove the node
            # self._remove_node(node)
            _children_backup = node.children
            node.children = []

            # Accuracy before removing the node
            val_acc_after = self.score(valid_data)

            # The tree remains pruned if accuracy increased afterwards
            if val_acc_after > val_acc_before:
                nodecounts.append(self.node_count())

                accuracies["train"].append(self.score(train_data))
                accuracies["test"].append(self.score(test_data))
                accuracies["valid"].append(val_acc_after)

            elif val_acc_after < val_acc_before:
                # Add the node back if it accuracy wasn't improved
                node.children = _children_backup

        return nodecounts, accuracies

    def multi_path_attrs(self):
        return {
            attr: self._multi_path_attrs(self.root, attr)
            for attr in ATTRIBUTES_NUMERICAL
        }

    @staticmethod
    def _multi_path_attrs(dtree, attr):
        """Return thresholds of a numerical attribute that occurs multiple times on a path in tree."""

        if not dtree.children:
            return []
        else:
            mpa = partial(DecisionTree._multi_path_attrs, attr=attr)
            max_path_thresholds = max(map(mpa, dtree.children), key=len)

            if attr == ATTRIBUTES[dtree.split_attr] and dtree.median_value is not None:
                return [dtree.median_value] + max_path_thresholds
            else:
                return list(max_path_thresholds)
