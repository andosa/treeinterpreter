# -*- coding: utf-8 -*-
from distutils.version import LooseVersion

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _tree

if LooseVersion(sklearn.__version__) < LooseVersion("0.17"):
    raise Exception("treeinterpreter requires scikit-learn 0.17 or later")

# Local cache of the model paths for each tree
_tree_cache = {}

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


def _predict_tree(model, X):
    """
    For a given DecisionTreeRegressor or DecisionTreeClassifier,
    returns a triple of [prediction, bias and feature_contributions], such
    that prediction ≈ bias + feature_contributions.
    """
    leaves = model.apply(X)

    global _tree_cache
    # If we haven't cached this tree, then add it here
    if model.tree_ not in _tree_cache:
        paths = _get_tree_paths(model.tree_, 0)
        for path in paths:
            path.reverse()
        _tree_cache[model.tree_] = paths
    else:
        # Grab the paths for this tree out of our cache if it's present
        paths = _tree_cache[model.tree_]

    # remove the single-dimensional inner arrays
    values = model.tree_.value.squeeze()

    if type(model) == DecisionTreeRegressor:
        contributions = np.zeros(X.shape)
        biases = np.zeros(X.shape[0])
        line_shape = X.shape[1]
    elif type(model) == DecisionTreeClassifier:
        # scikit stores category counts, we turn them into probabilities
        normalizer = values.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        values /= normalizer

        biases = np.zeros((X.shape[0], model.n_classes_))
        contributions = np.zeros((X.shape[0],
                                  X.shape[1], model.n_classes_))
        line_shape = (X.shape[1], model.n_classes_)

    for row, leaf in enumerate(leaves):
        for path in paths:
            if leaf == path[-1]:
                break
        biases[row] = values[path[0]]
        contribs = np.zeros(line_shape)
        for i in range(len(path) - 1):
            contrib = values[path[i+1]] - \
                      values[path[i]]
            contribs[model.tree_.feature[path[i]]] += contrib
        contributions[row] = contribs
        direct_prediction = values[leaves]

    return direct_prediction, biases, contributions


def _predict_forest(model, X):
    """
    For a given RandomForestRegressor or RandomForestClassifier,
    returns a triple of [prediction, bias and feature_contributions], such
    that prediction ≈ bias + feature_contributions.
    """
    biases = []
    contributions = []
    predictions = []
    for tree in model.estimators_:
        pred, bias, contribution = _predict_tree(tree, X)
        biases.append(bias)
        contributions.append(contribution)
        predictions.append(pred)
    return (np.mean(predictions, axis=0), np.mean(biases, axis=0),
            np.mean(contributions, axis=0))


def predict(model, X):
    """ Returns a triple (prediction, bias, feature_contributions), such
    that prediction ≈ bias + feature_contributions.
    Parameters
    ----------
    model : DecisionTreeRegressor, DecisionTreeClassifier or
        RandomForestRegressor, RandomForestClassifier
    Scikit-learn model on which the prediction should be decomposed.

    X : array-like, shape = (n_samples, n_features)
    Test samples.

    Returns
    -------
    decomposed prediction : triple of
    * prediction, shape = (n_samples) for regression and (n_samples, n_classes)
        for classification
    * bias, shape = (n_samples) for regression and (n_samples, n_classes) for
        classification
    * contributions, shape = (n_samples, n_features) for regression or
        shape = (n_samples, n_features, n_classes) for classification
    """
    # Only single out response variable supported,
    if model.n_outputs_ > 1:
        raise ValueError("Multilabel classification trees not supported")

    if (type(model) == DecisionTreeRegressor or
        type(model) == DecisionTreeClassifier):
        return _predict_tree(model, X)
    elif (type(model) == RandomForestRegressor or
          type(model) == RandomForestClassifier):
        return _predict_forest(model, X)
    else:
        raise ValueError("Wrong model type. Base learner needs to be \
            DecisionTreeClassifier or DecisionTreeRegressor.")

if __name__ == "__main__":
    # test
    from sklearn.datasets import load_iris
    iris = load_iris()
    idx = range(len(iris.data))
    np.random.shuffle(idx)
    X = iris.data[idx]
    Y = iris.target[idx]
    dt = RandomForestClassifier(max_depth=3)
    dt.fit(X[:len(X)/2], Y[:len(X)/2])
    testX = X[len(X)/2:len(X)/2+5]
    base_prediction = dt.predict_proba(testX)
    pred, bias, contrib = _predict_forest(dt, testX)

    assert(np.allclose(base_prediction, pred))
    assert(np.allclose(pred, bias + np.sum(contrib, axis=1)))
