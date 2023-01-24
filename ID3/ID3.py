import math

from DecisonTree import Leaf, Question, DecisionNode, class_counts, unique_vals
from utils import *

"""
Make the imports of python packages needed
"""


class ID3:
    def __init__(self, label_names: list, min_for_pruning=0, target_attribute='diagnosis'):
        self.label_names = label_names  # first col is class (sick/healthy), the rest are feature/column names
        self.target_attribute = target_attribute
        self.tree_root = None
        self.used_features = set()
        self.min_for_pruning = min_for_pruning

    @staticmethod
    def entropy(rows: np.ndarray, labels: np.ndarray):
        """
        Calculate the entropy of a distribution for the classes probability values.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: entropy value.
        """
        # TODO:
        #  Calculate the entropy of the data as shown in the class.
        #  - You can use counts as a helper dictionary of label -> count, or implement something else.

        counts = class_counts(rows, labels)
        impurity = 0.0

        # ====== YOUR CODE: ======
        tot_examples = np.sum(list(counts.values()))
        for c in counts.keys():
            pc = counts[c]/tot_examples
            impurity += pc * math.log(pc, 2)
        impurity = -1 * impurity
        # ========================

        return impurity

    def info_gain(self, left, left_labels, right, right_labels, current_uncertainty):
        """
        Calculate the information gain, as the uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        :param left: the left child rows.
        :param left_labels: the left child labels.
        :param right: the right child rows.
        :param right_labels: the right child labels.
        :param current_uncertainty: the current uncertainty of the current node
        :return: the info gain for splitting the current node into the two children left and right.
        """
        # TODO:
        #  - Calculate the entropy of the data of the left and the right child.
        #  - Calculate the info gain as shown in class.
        assert (len(left) == len(left_labels)) and (len(right) == len(right_labels)), \
            'The split of current node is not right, rows size should be equal to labels size.'

        info_gain_value = 0.0
        # ====== YOUR CODE: ======
        tot_examples = len(left) + len(right)
        left_elem = (len(left) / tot_examples) * self.entropy(left, left_labels)
        right_elem = (len(right) / tot_examples) * self.entropy(right, right_labels)
        info_gain_value = current_uncertainty - (left_elem + right_elem)
        # ========================

        return info_gain_value

    def partition(self, rows, labels, question: Question, current_uncertainty):
        """
        Partitions the rows by the question.
        :param rows: array of samples
        :param labels: rows data labels.
        :param question: an instance of the Question which we will use to partition the data.
        :param current_uncertainty: the current uncertainty of the current node
        :return: Tuple of (gain, true_rows, true_labels, false_rows, false_labels)
        """
        # TODO:
        #   - For each row in the dataset, check if it matches the question.
        #   - If so, add it to 'true rows', otherwise, add it to 'false rows'.
        #   - Calculate the info gain using the `info_gain` method.

        gain, true_rows, true_labels, false_rows, false_labels = None, None, None, None, None
        assert len(rows) == len(labels), 'Rows size should be equal to labels size.'

        # ====== YOUR CODE: ======
        true_rows = []
        true_labels = []
        false_rows = []
        false_labels = []
        for i, row in enumerate(rows):
            if question.match(row):
                true_rows.append(row)
                true_labels.append(labels[i])
            else:
                false_rows.append(row)
                false_labels.append(labels[i])

        gain = self.info_gain(true_rows, true_labels, false_rows, false_labels, current_uncertainty)
        # ========================

        return gain, true_rows, true_labels, false_rows, false_labels

    def find_best_split(self, rows, labels):
        """
        Find the best question to ask by iterating over every feature / value and calculating the information gain.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: Tuple of (best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels)
        """
        # TODO:
        #   - For each feature of the dataset, build a proper question to partition the dataset using this feature.
        #   - find the best feature to split the data. (using the `partition` method)
        best_gain = - math.inf  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        best_false_rows, best_false_labels = None, None
        best_true_rows, best_true_labels = None, None
        current_uncertainty = self.entropy(rows, labels)

        # ====== YOUR CODE: ======
        feature_names = np.copy(self.label_names[1:])
        for i, feature in enumerate(feature_names):
            # if feature not in self.used_features:
            values = list(unique_vals(rows, i))
            values.sort()
            thresholds = [(0.5 * (values[j] + values[j + 1])) for j in range(len(values) - 1)]
            for t in thresholds:
                question = Question(feature, i, t)
                gain, true_rows, true_labels, false_rows, false_labels = \
                    self.partition(rows, labels, question, current_uncertainty)
                if gain >= best_gain:
                    best_gain = gain
                    best_true_rows, best_true_labels, best_false_rows, best_false_labels = \
                        true_rows, true_labels, false_rows, false_labels
                    best_question = question
        # ========================

        return best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels

    def build_tree(self, rows, labels):
        """
        Build the decision Tree in recursion.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: a Question node, This records the best feature / value to ask at this point, depending on the answer.
                or leaf if we have to prune this branch (in which cases ?)

        """
        # TODO:
        #   - Try partitioning the dataset using the feature that produces the highest gain.
        #   - Recursively build the true, false branches.
        #   - Build the Question node which contains the best question with true_branch, false_branch as children
        best_question = None
        true_branch, false_branch = None, None

        # ====== YOUR CODE: ======


        # if all of the labels are the same
        if self.entropy(rows, labels) == 0:  # OR we can use np.all(labels == labels[0]):
            return Leaf(rows, labels)

        # Early pruning
        if len(labels) <= self.min_for_pruning:
            return Leaf(rows, labels)

        # if all the features have been used
        # we subtract 1 from label_names because the first is the class (sick/healthy)
        # if len(self.used_features) == (len(self.label_names)-1):
        #     return Leaf(rows, labels)

        best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels = \
            self.find_best_split(rows, labels)

        # self.used_features.add(best_question.column)
        true_branch = self.build_tree(best_true_rows, best_true_labels)
        false_branch = self.build_tree(best_false_rows, best_false_labels)
        # ========================

        return DecisionNode(best_question, true_branch, false_branch)

    def fit(self, x_train, y_train):
        """
        Trains the ID3 model. By building the tree.
        :param x_train: A labeled training data.
        :param y_train: training data labels.
        """
        # TODO: Build the tree that fits the input data and save the root to self.tree_root

        # ====== YOUR CODE: ======
        self.tree_root = self.build_tree(x_train, y_train)
        # ========================

    def predict_sample(self, row, node: DecisionNode or Leaf = None):
        """
        Predict the most likely class for single sample in subtree of the given node.
        :param row: vector of shape (1,D).
        :return: The row prediction.
        """
        # TODO: Implement ID3 class prediction for set of data.
        #   - Decide whether to follow the true-branch or the false-branch.
        #   - Compare the feature / value stored in the node, to the example we're considering.

        if node is None:
            node = self.tree_root
        prediction = None

        # ====== YOUR CODE: ======
        if type(node) is Leaf:
            # the leaf node can have examples with any classification
            # the classification of the leaf is determined by the majority
            # returns the label that appears the most out of the training examples from this node
            prediction = max(node.predictions, key=lambda x: node.predictions[x])
        else:  # node is of type DecisionNode
            if node.question.match(row):
                prediction = self.predict_sample(row, node.true_branch)
            else:
                prediction = self.predict_sample(row, node.false_branch)
        # ========================

        return prediction

    def predict(self, rows):
        """
        Predict the most likely class for each sample in a given vector.
        :param rows: vector of shape (N,D) where N is the number of samples.
        :return: A vector of shape (N,) containing the predicted classes.
        """
        # TODO:
        #  Implement ID3 class prediction for set of data.

        y_pred = None

        # ====== YOUR CODE: ======
        pred = []
        for row in rows:
            pred.append(self.predict_sample(row))

        y_pred = np.array(pred)
        # ========================

        return y_pred
