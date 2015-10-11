__author__ = 'yorko'

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import preprocessing
from load_data import load_train_and_test
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score

X_train, y, X_test, _, target_encoder  = load_train_and_test("../data/car_insurance_train.csv",
                                         "../data/car_insurance_test.csv")

# parameter combinations ot try
tree_params = {'criterion': ('gini', 'entropy'),
               'max_depth': list(range(1,11)),
               'min_samples_leaf': list(range(1,11))}

locally_best_tree = GridSearchCV(DecisionTreeClassifier(),
                                 tree_params,
                                 verbose=True, n_jobs=4, cv=5)
locally_best_tree.fit(X_train, y)

print("Best params:", locally_best_tree.best_params_)
print("Best cross validaton score", locally_best_tree.best_score_)

# export tree visualization
# after that $ dot -Tpng tree.dot -o tree.png    (PNG format)
# or open in Graphviz
export_graphviz(locally_best_tree.best_estimator_, out_file="../output/tree.dot")

# make predictions.
predicted_labels = locally_best_tree.best_estimator_.predict(X_test)

# turn predictions into data frame and save as csv file
predicted_df = pd.DataFrame(predicted_labels,
                            index = np.arange(1, X_test.shape[0] + 1),
                            columns=["compensated"])
predicted_df.to_csv("../data/tree_prediction.csv", index_label="id")
#
#
# # that's for me, you don't know the answers
# expected_labels_df = pd.read_csv("../data/car_insurance_test_labels.csv",
#                                  header=0, index_col=0)
# expected_labels = target_encoder.transform(expected_labels_df['compensated'])
# print(roc_auc_score(predicted_labels, expected_labels))

