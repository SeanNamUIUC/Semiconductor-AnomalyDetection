import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

def select_k_best(X_train, y_train, X_test, k=50):
    imputer = SimpleImputer(strategy="median")

    X_train_input = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_input = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    selector = SelectKBest(
        score_func=f_classif,
        k=k
    )

    selector.fit(X_train_input, y_train)

    selected_cols = X_train.columns[
        selector.get_support()
    ].tolist()

    X_train_out = X_train[selected_cols].copy()
    X_test_out = X_test[selected_cols].copy()

    score_df = pd.DataFrame({
        "feature": X_train.columns,
        "score": selector.scores_,
        "pvalue": selector.pvalues_
    }).sort_values(
        by="score",
        ascending=False
    )

    return X_train_out, X_test_out, selected_cols, score_df