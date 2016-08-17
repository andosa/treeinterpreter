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

from sklearn.tree import export_graphviz

def _predict_tree(model, X):
    """
    For a given DecisionTreeRegressor or DecisionTreeClassifier,
    returns a triple of [prediction, bias and feature_contributions], such
    that prediction ≈ bias + feature_contributions.
    """
    tree = model.tree_

    if type(model) == DecisionTreeRegressor:
        b = tree.value[0,0,0]
        bias = np.array([b,] * X.shape[0]) # output shape = (n_samples)

        contribs = np.zeros(X.shape)
        preds    = np.zeros(X.shape[0])
    elif type(model) == DecisionTreeClassifier:
        b = tree.value[0,0]
        b = b / b.sum()
        bias = np.array([b,] * X.shape[0]) # output shape = (n_samples, n_classes)

        contribs = np.zeros((X.shape[0], X.shape[1], model.n_classes_))
        preds    = np.zeros((X.shape[0], model.n_classes_))
    else:
        raise('%s model is not a scikit decision tree model.' % type(model))

    # Contributions & predictions for each line
    for i,x in enumerate(np.array(X)):
        leaf = model.apply([x])
        if type(model) == DecisionTreeRegressor:
            preds[i] = tree.value[leaf].squeeze()
        elif type(model) == DecisionTreeClassifier:
            preds[i] = tree.value[leaf].squeeze()
            s = sum(preds[i])
            normalizer = s if s != 0 else 1
            preds[i] = preds[i] / normalizer

        node = 0 # Starting from root, going to the leaf
        while node != leaf:
            # Using node indices to find the leaf
            if leaf < tree.children_right[node]:
                prev = node
                node = tree.children_left[node]
            else:
                prev = node
                node = tree.children_right[node]

            if type(model) == DecisionTreeRegressor:
                contribs[i,tree.feature[node]] = contribs[i,tree.feature[node]] + tree.value[node,0] - tree.value[prev,0]
            elif type(model) == DecisionTreeClassifier:
                norm1 = sum(tree.value[node, 0])
                norm1 = norm1 if norm1 != 0 else 1
                norm2 = sum(tree.value[prev, 0])
                norm2 = norm2 if norm2 != 0 else 1
                contribs[i,tree.feature[node]] = contribs[i,tree.feature[node]] + tree.value[node,0] / norm1 - tree.value[prev,0] / norm2

            #print node, tree.threshold[node]
            if node == -1:
                export_graphviz(model)
                print x
                raise Exception('Can\'t find the following leaf: %d' % leaf)

    return preds, bias, contribs


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

    if(type(model) == RandomForestClassifier or
       type(model) == RandomForestRegressor):
        for tree in model.estimators_:
            # No need for last dimension when n_classes == 1:
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
