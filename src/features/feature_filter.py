import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

def drop_high_missing(X_train, X_test, threshold=0.5):
    missing_ratio = X_train.isnull().mean()

    high_missing_cols = missing_ratio[
        missing_ratio > threshold
    ].index.tolist()

    X_train_out = X_train.drop(columns=high_missing_cols)
    X_test_out = X_test.drop(columns=high_missing_cols)

    return X_train_out, X_test_out, high_missing_cols

def variance_filter(X_train, X_test, threshold=0.0):
    imputer = SimpleImputer(strategy="median")

    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_train_imputed)

    selected_cols = X_train.columns[
        selector.get_support()
    ].tolist()

    X_train_out = X_train[selected_cols].copy()
    X_test_out = X_test[selected_cols].copy()

    return X_train_out, X_test_out, selected_cols