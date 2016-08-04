# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import expit

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, _tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from distutils.version import LooseVersion
import sklearn
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


def _predict_tree(model, X):
    """
    For a given DecisionTreeRegressor or DecisionTreeClassifier,
    returns a triple of [prediction, bias and feature_contributions], such
    that prediction ≈ bias + feature_contributions.
    """
    leaves = model.apply(X)
    paths = _get_tree_paths(model.tree_, 0)

    for path in paths:
        path.reverse()

    leaf_to_path = {}
    #map leaves to paths
    for path in paths:
        leaf_to_path[path[-1]] = path

    # remove the single-dimensional inner arrays
    values = model.tree_.value.squeeze()
    # reshape if squeezed into a single float
    if len(values.shape) == 0:
        values = np.array([values])
    if type(model) == DecisionTreeRegressor:
        biases = np.full(X.shape[0], values[paths[0][0]])
        line_shape = X.shape[1]
    elif type(model) == DecisionTreeClassifier:
        # scikit stores category counts, we turn them into probabilities
        normalizer = values.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        values /= normalizer

        biases = np.tile(values[paths[0][0]], (X.shape[0], 1))
        line_shape = (X.shape[1], model.n_classes_)
    direct_prediction = values[leaves]


    #make into python list, accessing values will be faster
    values_list = list(values)
    feature_index = list(model.tree_.feature)

    contributions = []
    for row, leaf in enumerate(leaves):
        for path in paths:
            if leaf == path[-1]:
                break

        contribs = np.zeros(line_shape)
        for i in range(len(path) - 1):

            contrib = values_list[path[i+1]] - \
                     values_list[path[i]]
            contribs[feature_index[path[i]]] += contrib
        contributions.append(contribs)

    return direct_prediction, biases, np.array(contributions)


def _predict_forest(model, X):
    """
    For a given regressor or classifier,
    returns a triple of [prediction, bias and feature_contributions], such
    that prediction ≈ bias + feature_contributions.
    """
    n_estimators = model.n_estimators
    n_classes = 1
    try:
        n_classes = model.n_classes_
    except:
        n_classes = 1

    predictions   = np.zeros((X.shape[0], n_classes))
    biases        = np.zeros((X.shape[0], n_classes))
    contributions = np.zeros((X.shape[0], X.shape[1], n_classes))

    if(type(model) == RandomForestClassifier):
        for tree in model.estimators_:
            pred, bias, contribution = _predict_tree(tree, X)
            predictions   = predictions + pred / n_estimators
            biases        = biases + bias / n_estimators
            contributions = contributions + contribution / n_estimators
    if(type(model) == RandomForestRegressor):
        for tree in model.estimators_:
            # No need for last dimension (n_classes == 1) here:
            contributions = np.squeeze(contributions)
            predictions = np.squeeze(predictions)
            biases = np.squeeze(biases)

            pred, bias, contribution = _predict_tree(tree, X)
            predictions   = predictions + pred / n_estimators
            biases        = biases + bias / n_estimators
            contributions = contributions + contribution / n_estimators

    elif(type(model) == GradientBoostingRegressor or
         type(model) == GradientBoostingClassifier):
	learning_rate = model.learning_rate

	# Base learner
        base_pred = model.init_.predict(X)
        if n_classes == 2:
            biases[:,1]      = np.squeeze(base_pred)
            predictions[:,1] = np.squeeze(base_pred)
            biases[:,0]      = -np.squeeze(base_pred)
            predictions[:,0] = -np.squeeze(base_pred)
        else:
            biases = biases + base_pred
            predictions = predictions + base_pred

	# Tree learners
	for trees in model.estimators_:
            for c, tree in enumerate(trees):
                pred, bias, contribution = _predict_tree(tree, X)
                biases[:,c]          = biases[:,c] + bias * learning_rate
                contributions[:,:,c] = contributions[:,:,c] + contribution * learning_rate
                predictions[:,c]     = predictions[:,c] + pred * learning_rate

        # If classification, need to output probabilities. So scaling score into [0-1]
        if(type(model) == GradientBoostingClassifier):
            # Special case when only 2 classes:
            if n_classes == 2:
                predictions = model.loss_._score_to_proba(predictions[:,0])
                contributions[:,:,1] = contributions[:,:,0]
                contributions[:,:,0] = -contributions[:,:,0]
                biases[:,1] = biases[:,0]
                biases[:,0] = -biases[:,0]
            else:
                predictions = model.loss_._score_to_proba(predictions)
        else:
            predictions   = np.squeeze(predictions)
            biases        = np.squeeze(biases)
            contributions = np.squeeze(contributions)

    return (predictions, biases, contributions)

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
    #if model.n_outputs_ > 1:
    #    raise ValueError("Multilabel classification trees not supported")

    if (type(model) == DecisionTreeRegressor or
        type(model) == DecisionTreeClassifier):
        return _predict_tree(model, X)
    elif (type(model) == RandomForestRegressor or
          type(model) == RandomForestClassifier or
          type(model) == GradientBoostingClassifier or
          type(model) == GradientBoostingRegressor):
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
