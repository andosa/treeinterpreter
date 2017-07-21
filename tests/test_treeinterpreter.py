#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_treeinterpreter
----------------------------------

Tests for `treeinterpreter` module.
"""

import numpy as np
import unittest

from sklearn.datasets import load_boston, load_iris
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              ExtraTreesClassifier, ExtraTreesRegressor,)
from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                          ExtraTreeClassifier, ExtraTreeRegressor,)

from treeinterpreter import treeinterpreter

class TestTreeinterpreter(unittest.TestCase):

    def setUp(self):
        self.boston = load_boston()
        self.iris = load_iris()

    def test_tree_regressor(self):
        for TreeRegressor in (DecisionTreeRegressor, ExtraTreeRegressor):
            X = self.boston.data
            Y = self.boston.target
            testX = X[int(len(X)/2):]
            
            #Predict for decision tree
            dt = TreeRegressor()
            dt.fit(X[:int(len(X)/2)], Y[:int(len(X)/2)])

            base_prediction = dt.predict(testX)
            pred, bias, contrib = treeinterpreter.predict(dt, testX)
            self.assertTrue(np.allclose(base_prediction, pred))
            self.assertTrue(np.allclose(pred, bias + np.sum(contrib, axis=1)))
            
            testX = X[-1:]
            base_prediction = dt.predict(testX)
            pred, bias, contrib = treeinterpreter.predict(dt, testX)
            self.assertTrue(np.allclose(base_prediction, pred))
            self.assertTrue(np.allclose(pred, bias + np.sum(contrib, axis=1)))
        
        
    def test_tree_classifier(self):
        for TreeClassifier in (DecisionTreeClassifier, ExtraTreeClassifier):
            X = self.iris.data
            Y = self.iris.target
            dt = TreeClassifier()
            dt.fit(X[:int(len(X)/2)], Y[:int(len(X)/2)])
            testX = X[int(len(X)/2):int(len(X)/2)+1]
            base_prediction = dt.predict_proba(testX)
            pred, bias, contrib = treeinterpreter.predict(dt, testX)
            
            self.assertTrue(np.allclose(base_prediction, pred))
            self.assertTrue(np.allclose(pred, bias + np.sum(contrib, axis=1)))

    def test_forest_regressor(self):
        for ForestRegressor in (RandomForestRegressor, ExtraTreesRegressor):
            X = self.boston.data
            Y = self.boston.target
            testX = X[int(len(X)/2):]
            
            #Predict for decision tree
            dt = ForestRegressor(n_estimators=10)
            dt.fit(X[:int(len(X)/2)], Y[:int(len(X)/2)])

            base_prediction = dt.predict(testX)
            pred, bias, contrib = treeinterpreter.predict(dt, testX)
            self.assertTrue(np.allclose(base_prediction, pred))
            self.assertTrue(np.allclose(pred, bias + np.sum(contrib, axis=1)))
        
    def test_forest_regressor_joint(self):
        for ForestRegressor in (RandomForestRegressor, ExtraTreesRegressor):
            X = self.boston.data
            Y = self.boston.target
            testX = X[int(len(X)/2):]
            
            #Predict for decision tree
            dt = ForestRegressor(n_estimators=10)
            dt.fit(X[:int(len(X)/2)], Y[:int(len(X)/2)])

            base_prediction = dt.predict(testX)
            pred, bias, contribs = treeinterpreter.predict(dt, testX, joint_contribution=True)
            self.assertTrue(np.allclose(base_prediction, pred))
            
            self.assertTrue(np.allclose(base_prediction, np.array([sum(contrib.values()) for contrib in contribs]) + bias))

    def test_forest_classifier(self):
        for ForestClassifier in (RandomForestClassifier, ExtraTreesClassifier):
            idx = np.arange(len(self.iris.data))
            np.random.shuffle(idx)
            X = self.iris.data[idx]
            Y = self.iris.target[idx]
            dt = ForestClassifier(max_depth=3)
            dt.fit(X[:int(len(X)/2)], Y[:int(len(X)/2)])
            testX = X[int(len(X)/2):]
            base_prediction = dt.predict_proba(testX)
            pred, bias, contrib = treeinterpreter.predict(dt, testX)
            self.assertTrue(np.allclose(base_prediction, pred))
            self.assertTrue(np.allclose(pred, bias + np.sum(contrib, axis=1)))


    def test_forest_classifier_joint(self):
        for ForestClassifier in (RandomForestClassifier, ExtraTreesClassifier):
            idx = np.arange(len(self.iris.data))
            np.random.shuffle(idx)
            X = self.iris.data[idx]
            Y = self.iris.target[idx]
            dt = ForestClassifier(max_depth=3)
            dt.fit(X[:int(len(X)/2)], Y[:int(len(X)/2)])
            testX = X[int(len(X)/2):]
            base_prediction = dt.predict_proba(testX)
            pred, bias, contribs = treeinterpreter.predict(dt, testX, joint_contribution=True)
            self.assertTrue(np.allclose(base_prediction, pred))
            self.assertTrue(np.allclose(base_prediction, np.array([sum(contrib.values()) for contrib in contribs]) + bias))
        
    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
