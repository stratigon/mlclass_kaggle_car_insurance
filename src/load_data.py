import numpy as np
import pandas as pd
from sklearn import preprocessing


def load_train_and_test(path_to_train, path_to_test):
    # read data into pandas data frames
    train_df = pd.read_csv(path_to_train,
                           header=0, index_col=0)
    test_df = pd.read_csv(path_to_test,
                          header=0, index_col=0)

    # number of observations (objects) in train and test sets
    n_train, n_test = train_df.shape[0], test_df.shape[0]

    # auto brand and too_much are categorical so we encode these columns
    # ex: "Volvo" -> 1, "Audi" -> 2 etc
    auto_brand_encoder = preprocessing.LabelEncoder()
    auto_brand_encoder.fit(train_df['auto_brand'])
    target_encoder = preprocessing.LabelEncoder()
    target_encoder.fit(train_df['too_much'])

    # form a numpy array to fit as a train set
    # X will have encoded columns 'auto_brand' and 'compensated'
    X_train = np.hstack([auto_brand_encoder.transform(train_df['auto_brand'])
                  .reshape([n_train,1]),
                   train_df['compensated']
                  .reshape([n_train,1])])

    # form a numpy array to fit as train set labels
    y = target_encoder.transform(train_df['too_much'])

    # form a numpy array of a test set
    X_test = np.hstack([auto_brand_encoder.transform(test_df['auto_brand'])
                  .reshape([n_test, 1]),
                   test_df['compensated']
                  .reshape([n_test,1])])

    return X_train, y, X_test, auto_brand_encoder, target_encoder

if __name__ == "__main__":
    X_train, y, X_test, _, target_encoder  = load_train_and_test("../data/car_insurance_train.csv",
                                         "../data/car_insurance_test.csv")
    print(X_train.shape)