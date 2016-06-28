#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_treeinterpreter
----------------------------------

Tests for `treeinterpreter` module.
"""

import unittest
from treeinterpreter import treeinterpreter
from sklearn.datasets import load_boston, load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np

class TestTreeinterpreter(unittest.TestCase):

    def setUp(self):
        self.boston = load_boston()
        self.iris = load_iris()

    def test_tree_regressor(self):
        X = self.boston.data
        Y = self.boston.target
        testX = X[len(X)/2:]
        
        #Predict for decision tree
        dt = DecisionTreeRegressor()
        dt.fit(X[:len(X)/2], Y[:len(X)/2])

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
        X = self.iris.data
        Y = self.iris.target
        dt = DecisionTreeClassifier()
        dt.fit(X[:len(X)/2], Y[:len(X)/2])
        testX = X[len(X)/2:len(X)/2+1]
        base_prediction = dt.predict_proba(testX)
        pred, bias, contrib = treeinterpreter.predict(dt, testX)
        
        self.assertTrue(np.allclose(base_prediction, pred))
        self.assertTrue(np.allclose(pred, bias + np.sum(contrib, axis=1)))

    def test_forest_regressor(self):
        X = self.boston.data
        Y = self.boston.target
        testX = X[len(X)/2:]
        
        #Predict for decision tree
        dt = RandomForestRegressor(n_estimators=10)
        dt.fit(X[:len(X)/2], Y[:len(X)/2])

        base_prediction = dt.predict(testX)
        pred, bias, contrib = treeinterpreter.predict(dt, testX)
        self.assertTrue(np.allclose(base_prediction, pred))
        self.assertTrue(np.allclose(pred, bias + np.sum(contrib, axis=1)))
        
    def test_forest_regressor_joint(self):
        X = self.boston.data
        Y = self.boston.target
        testX = X[len(X)/2:]
        
        #Predict for decision tree
        dt = RandomForestRegressor(n_estimators=10)
        dt.fit(X[:len(X)/2], Y[:len(X)/2])

        base_prediction = dt.predict(testX)
        pred, bias, contribs = treeinterpreter.predict(dt, testX, joint_contribution=True)
        self.assertTrue(np.allclose(base_prediction, pred))
        
        self.assertTrue(np.allclose(base_prediction, np.array([sum(contrib.values()) for contrib in contribs]) + bias))

    def test_forest_classifier(self):
        idx = range(len(self.iris.data))
        np.random.shuffle(idx)
        X = self.iris.data[idx]
        Y = self.iris.target[idx]
        dt = RandomForestClassifier(max_depth=3)
        dt.fit(X[:len(X)/2], Y[:len(X)/2])
        testX = X[len(X)/2:]
        base_prediction = dt.predict_proba(testX)
        pred, bias, contrib = treeinterpreter.predict(dt, testX)
        self.assertTrue(np.allclose(base_prediction, pred))
        self.assertTrue(np.allclose(pred, bias + np.sum(contrib, axis=1)))


    def test_forest_classifier_joint(self):
        idx = range(len(self.iris.data))
        np.random.shuffle(idx)
        X = self.iris.data[idx]
        Y = self.iris.target[idx]
        dt = RandomForestClassifier(max_depth=3)
        dt.fit(X[:len(X)/2], Y[:len(X)/2])
        testX = X[len(X)/2:]
        base_prediction = dt.predict_proba(testX)
        pred, bias, contribs = treeinterpreter.predict(dt, testX, joint_contribution=True)
        self.assertTrue(np.allclose(base_prediction, pred))
        self.assertTrue(np.allclose(base_prediction, np.array([sum(contrib.values()) for contrib in contribs]) + bias))
        
    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
