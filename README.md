# DecisionTree

The project includes implementation of Decision tree classifier without using any libraries. The Objective of this project is to make prediction and train the model over a dataset (Advertisement dataset, Breast Cancer dataset, Iris dataset). The dataset is split randomly between training and testing set in the ratio of 8:2 respectively. After constructing the decision tree with the training data and applying the appropriate pruning strategy following details are observed in two independent runs:

# Sample Outputs (Advertisement dataset)

    Dataset URL: https://www.superdatascience.com/pages/machine-learning

    1st Run, (test set_1 for a training set_1)
        Accuracy before pruning: 88.0%
        Accuracy after pruning: 90.0%
        Total Accuracy Increase: 2%

    2nd Run, (test set_2 for a training set_2)
        Accuracy before pruning on 86.0%
        Accuracy after pruning on the same set: 91.0%
        Total Accuracy Increase: 5%
        
# Sample Outputs (Breast Cancer dataset)

    Dataset URL: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/

    1st Run, (test set_1 for a training set_1)
        Accuracy before pruning: 66.0%
        Accuracy after pruning: 74.0%
        Total Accuracy Increase: 8%

    2nd Run, (test set_2 for a training set_2)
        Accuracy before pruning on 69.0%
        Accuracy after pruning on the same set: 74.0%
        Total Accuracy Increase: 5%

# Sample outputs (Iris Data Set)

    Dataset URL: https://archive.ics.uci.edu/ml/datasets/iris

    1st Run, (test set_1 for a training set_1)
        Accuracy before pruning: 83.0%
        Accuracy after pruning: 90.0%
        Total Accuracy Increase: 7%

    2nd Run, (test set_2 for a training set_2)
        Accuracy before pruning on 93.0%
        Accuracy after pruning on the same set: 100.0%
        Total Accuracy Increase: 7%

# Pruning Strategy

To prune each node one by one (except the root and the leaf nodes), and check weather pruning helps in increasing the accuracy, if the accuracy is increased, prune the node which gives the maximum accuracy at the end to construct the final tree (if the accuracy of 100% is achieved by pruning a node, stop the algorithm right there and do not check for further new nodes).

# How to configure

    1. If the system don't have python installed in it, first install any python version (version greater than v2.7).
        https://www.python.org/downloads/
    2. The code has following dependencies, which needs to be installed before running this code: - Pandas. More details at: https://pandas.pydata.org
        from command line: pip install pandas
        scikit-learn for only one method in the driver code - train test split
        from command line: pip install -U scikit-learn
    3. Open root directory (DecisionTree) of the project and run command
        from command line: python driver.py
