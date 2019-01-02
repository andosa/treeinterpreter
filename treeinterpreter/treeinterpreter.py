# -*- coding: utf-8 -*-
import numpy as np
import sklearn
from collections import Counter
from sklearn.ensemble.forest import ForestClassifier, ForestRegressor
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


def _predict_tree(model, X, joint_contribution=False):
    """
    For a given DecisionTreeRegressor, DecisionTreeClassifier,
    ExtraTreeRegressor, or ExtraTreeClassifier,
    returns a triple of [prediction, bias and feature_contributions], such
    that prediction â‰ˆ bias + feature_contributions.
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
    values = model.tree_.value.squeeze(axis=1)
    # reshape if squeezed into a single float
    if len(values.shape) == 0:
        values = np.array([values])
    if isinstance(model, DecisionTreeRegressor):
        biases = np.full(X.shape[0], values[paths[0][0]])
        line_shape = X.shape[1]
    elif isinstance(model, DecisionTreeClassifier):
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
    if joint_contribution:
        for row, leaf in enumerate(leaves):
            path = leaf_to_path[leaf]
            
            
            path_features = set()
            contributions.append({})
            for i in range(len(path) - 1):
                path_features.add(feature_index[path[i]])
                contrib = values_list[path[i+1]] - \
                         values_list[path[i]]
                #path_features.sort()
                contributions[row][tuple(sorted(path_features))] = \
                    contributions[row].get(tuple(sorted(path_features)), 0) + contrib
        return direct_prediction, biases, contributions
        
    else:
        unique_leaves = np.unique(leaves)
        unique_contributions = {}
        
        for row, leaf in enumerate(unique_leaves):
            for path in paths:
                if leaf == path[-1]:
                    break
            
            contribs = np.zeros(line_shape)
            for i in range(len(path) - 1):
                
                contrib = values_list[path[i+1]] - \
                         values_list[path[i]]
                contribs[feature_index[path[i]]] += contrib
            unique_contributions[leaf] = contribs
            
        for row, leaf in enumerate(leaves):
            contributions.append(unique_contributions[leaf])

        return direct_prediction, biases, np.array(contributions)


def _predict_forest(model, X, joint_contribution=False, aggregated_contributions=False):
    """
    For a given RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, or ExtraTreesClassifier returns a triple of
    [prediction, bias and feature_contributions], such that prediction â‰ˆ bias +
    feature_contributions.
    """
    biases = []
    contributions = []
    predictions = []

    
    if joint_contribution:
        
        # If user wants the contributions outputed to be already aggregated run this section. It uses Counter from 
        # collections to automatically sum all contributions and divide by number of trees.
        if aggregated_contributions:
            
            total_contributions = Counter()

            for tree in model.estimators_:
                pred, bias, contribution = _predict_tree(tree, X, joint_contribution=joint_contribution)

                biases.append(bias)
                predictions.append(pred)
                for dct in contribution: 
                    total_contributions.update(dct)

            # Total Contributions already aggregated. 
            total_contributions = {x: total_contributions[x]/len(X) for x in total_contributions.keys()}


            
            
        else:
            for tree in model.estimators_:
                pred, bias, contribution = _predict_tree(tree, X, joint_contribution=joint_contribution)

                biases.append(bias)
                contributions.append(contribution)
                predictions.append(pred)


            total_contributions = []

            for i in range(len(X)):
                contr = {}
                for j, dct in enumerate(contributions):
                    for k in set(dct[i]).union(set(contr.keys())):
                        contr[k] = (contr.get(k, 0)*j + dct[i].get(k,0) ) / (j+1)

                total_contributions.append(contr)    

            for i, item in enumerate(contribution):
                total_contributions[i]
                sm = sum([v for v in contribution[i].values()])
                

        
        return (np.mean(predictions, axis=0), np.mean(biases, axis=0),
            total_contributions)
    else:
        for tree in model.estimators_:
            pred, bias, contribution = _predict_tree(tree, X)

            biases.append(bias)
            contributions.append(contribution)
            predictions.append(pred)
        
        
        return (np.mean(predictions, axis=0), np.mean(biases, axis=0),
            np.mean(contributions, axis=0))


def predict(model, X, joint_contribution=False, aggregated_contributions=False):
    """ Returns a triple (prediction, bias, feature_contributions), such
    that prediction â‰ˆ bias + feature_contributions.
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
    
    aggregated_contributions : boolean
    Specifies if contributions are the aggregated contribution of all the 
    data samples. 

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
        If aggregated_contributions is False then nothing changes. 
        If aggregated_contributions is True then contributions is a dictionary 
        of the average contribution across all samples. 
    """
    # Only single out response variable supported,
    if model.n_outputs_ > 1:
        raise ValueError("Multilabel classification trees not supported")

    if (isinstance(model, DecisionTreeClassifier) or
        isinstance(model, DecisionTreeRegressor)):
        return _predict_tree(model, X, joint_contribution=joint_contribution)
    elif (isinstance(model, ForestClassifier) or
          isinstance(model, ForestRegressor)):
        return _predict_forest(model, X, joint_contribution=joint_contribution, aggregated_contributions=aggregated_contributions)
    else:
        raise ValueError("Wrong model type. Base learner needs to be a "
                         "DecisionTreeClassifier or DecisionTreeRegressor.")
