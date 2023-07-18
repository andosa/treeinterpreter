# -*- coding: utf-8 -*-
import numpy as np
import sklearn

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, _tree
from distutils.version import LooseVersion
if LooseVersion(sklearn.__version__) < LooseVersion("0.17"):
    raise Exception("treeinterpreter requires scikit-learn 0.17 or later")


def _get_tree_paths(tree, node_id, depth=0):
    """
    Returns all paths through the tree as list of node_ids
    """
    if node_id == _tree.TREE_LEAF:
        raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    if left_child != _tree.TREE_LEAF:
        left_paths = _get_tree_paths(tree, left_child, depth=depth + 1)
        right_paths = _get_tree_paths(tree, right_child, depth=depth + 1)

        for path in left_paths:
            path.append(node_id)
        for path in right_paths:
            path.append(node_id)
        paths = left_paths + right_paths
    else:
        paths = [[node_id]]
    return paths


class Treeinterpreter():

    def __init__(self, model, X, joint_contribution=False):
        """ Initialize the treeinterpreter object

        Parameters
        ----------
        model : DecisionTreeRegressor, DecisionTreeClassifier,
            ExtraTreeRegressor, ExtraTreeClassifier,
            RandomForestRegressor, RandomForestClassifier,
            ExtraTreesRegressor, ExtraTreesClassifier
        Scikit-learn model on which the prediction should be decomposed.

        X : array-like, shape = (n_samples, n_features)
        Test samples.

        joint_contribution : boolean
        Specifies if contributions are given individually from each feature,
        or jointly over them
        """
        self.model = model
        self.X = X
        self.joint_contribution = joint_contribution

    def _predict_tree(self, tree):
        """
        For a given DecisionTreeRegressor, DecisionTreeClassifier,
        ExtraTreeRegressor, or ExtraTreeClassifier,
        returns a triple of [prediction, bias and feature_contributions], such
        that prediction ≈ bias + feature_contributions.
        """
        leaves = tree.apply(self.X)
        paths = _get_tree_paths(tree.tree_, 0)

        for path in paths:
            path.reverse()

        leaf_to_path = {}
        # map leaves to paths
        for path in paths:
            leaf_to_path[path[-1]] = path

        # remove the single-dimensional inner arrays
        values = tree.tree_.value.squeeze(axis=1)
        # reshape if squeezed into a single float
        if len(values.shape) == 0:
            values = np.array([values])
        if isinstance(tree, DecisionTreeRegressor):
            # we require the values to be the same shape as the biases
            values = values.squeeze(axis=1)
            biases = np.full(self.X.shape[0], values[paths[0][0]])
            line_shape = self.X.shape[1]
        elif isinstance(tree, DecisionTreeClassifier):
            # scikit stores category counts, we turn them into probabilities
            normalizer = values.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            values /= normalizer

            biases = np.tile(values[paths[0][0]], (self.X.shape[0], 1))
            line_shape = (self.X.shape[1], tree.n_classes_)
        direct_prediction = values[leaves]

        # make into python list, accessing values will be faster
        values_list = list(values)
        feature_index = list(tree.tree_.feature)

        if self.joint_contribution:
            contributions = []
            for row, leaf in enumerate(leaves):
                path = leaf_to_path[leaf]

                path_features = set()
                contributions.append({})
                for i in range(len(path) - 1):
                    path_features.add(feature_index[path[i]])
                    contrib = values_list[path[i+1]] - \
                        values_list[path[i]]
                    # path_features.sort()
                    contributions[row][tuple(sorted(path_features))] = \
                        contributions[row].get(
                            tuple(sorted(path_features)), 0) + contrib
            return direct_prediction, biases, contributions

        else:
            unique_leaves = np.unique(leaves)
            contributions = np.zeros(
                [len(leaves), self.X.shape[1], values.shape[1]])

            for row, leaf in enumerate(unique_leaves):
                for path in paths:
                    if leaf == path[-1]:
                        break

                contribs = np.zeros(line_shape)
                for i in range(len(path) - 1):

                    contrib = values_list[path[i+1]] - \
                        values_list[path[i]]
                    contribs[feature_index[path[i]]] += contrib
                contributions[leaves == leaf, ...] = contribs

            return direct_prediction, biases, contributions

    def _predict_forest(self):
        """
        For a given RandomForestRegressor, RandomForestClassifier,
        ExtraTreesRegressor, or ExtraTreesClassifier returns a triple of
        [prediction, bias and feature_contributions], such that prediction ≈ bias +
        feature_contributions.
        """

        if self.joint_contribution:
            biases = []
            contributions = []
            predictions = []

            for tree in self.model.estimators_:
                pred, bias, contribution = self._predict_tree(tree)

                biases.append(bias)
                contributions.append(contribution)
                predictions.append(pred)

            total_contributions = []

            for i in range(len(self.X)):
                contr = {}
                for j, dct in enumerate(contributions):
                    for k in set(dct[i]).union(set(contr.keys())):
                        contr[k] = (contr.get(k, 0)*j +
                                    dct[i].get(k, 0)) / (j+1)

                total_contributions.append(contr)

            for i, item in enumerate(contribution):
                total_contributions[i]
                sm = sum([v for v in contribution[i].values()])

            return (np.mean(predictions, axis=0), np.mean(biases, axis=0),
                    total_contributions)
        else:
            mean_pred = 0.0
            mean_bias = 0.0
            mean_contribution = 0.0

            for i, tree in enumerate(self.model.estimators_):
                pred, bias, contribution = self._predict_tree(tree)

                mean_pred += pred
                mean_bias += bias
                mean_contribution += contribution

            mean_pred /= len(self.model.estimators_)
            mean_bias /= len(self.model.estimators_)
            mean_contribution /= len(self.model.estimators_)
            return mean_pred, mean_bias, mean_contribution

    def predict(self):
        """ Returns a triple (prediction, bias, feature_contributions), such
        that prediction ≈ bias + feature_contributions.

        Returns
        -------
        decomposed prediction : triple of
        * prediction, shape = (n_samples) for regression and (n_samples, n_classes)
            for classification
        * bias, shape = (n_samples) for regression and (n_samples, n_classes) for
            classification
        * contributions, If joint_contribution is False then returns and  array of 
            shape = (n_samples, n_features) for regression or
            shape = (n_samples, n_features, n_classes) for classification, denoting
            contribution from each feature.
            If joint_contribution is True, then shape is array of size n_samples,
            where each array element is a dict from a tuple of feature indices to
            to a value denoting the contribution from that feature tuple.
        """
        # Only single out response variable supported,
        if self.model.n_outputs_ > 1:
            raise ValueError("Multilabel classification trees not supported")

        if (isinstance(self.model, DecisionTreeClassifier) or
            isinstance(self.model, DecisionTreeRegressor)):
            return self._predict_tree()
        elif (isinstance(self.model, RandomForestClassifier) or
              isinstance(self.model, ExtraTreesClassifier) or
              isinstance(self.model, RandomForestRegressor) or
              isinstance(self.model, ExtraTreesRegressor)):
            return self._predict_forest()
        else:
            raise ValueError("Wrong model type. Base learner needs to be a "
                             "DecisionTreeClassifier or DecisionTreeRegressor.")
