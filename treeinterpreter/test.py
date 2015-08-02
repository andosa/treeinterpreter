import sys
sys.path.insert(0, r'c:\projects\scikit-learn-master\scikit-learn\build\lib.win32-2.7')
sys.path.insert(0, r'c:\projects\scikit-learn-master\build\lib.win-amd64-2.7')

from sklearn.tree import DecisionTreeClassifier as DTC

dt = DTC()
print type(dt)
print type(dt) == DTC