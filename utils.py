"""
Imports
"""
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.compose import ColumnTransformer

def customStratifiedKFold(X, y, random_state=42):
    """
    Stratified K fold
    Default: K = 7 and shuffle = True
    """
    vetor_X_train, vetor_y_train, vetor_X_test, vetor_y_test = [], [], [], []
    split = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)
    for train_index, val_index in split.split(X, y):
        vetor_X_train.append(X[train_index])
        vetor_X_test.append(X[val_index])
        vetor_y_train.append(y[train_index])
        vetor_y_test.append(y[val_index])
    return vetor_X_train, vetor_y_train, vetor_X_test, vetor_y_test


def transform_data(X):
    number_categorical_columns = [i for i in range(len(X[0])) if type(X[0][i]) == str]
    labelencoder = LabelEncoder()
    for i in number_categorical_columns:
        X[:, i] = labelencoder.fit_transform(X[:, i])
    # transformer = ColumnTransformer(
    #     transformers=[
    #         (
    #             "Country",        # Just a name
    #             OneHotEncoder(), # The transformer class
    #             number_categorical_columns # The column(s) to be applied on.
    #         )
    #     ],
    #     remainder='passthrough' # donot apply anything to the remaining columns
    # )
    # X = transformer.fit_transform(X)
    return X


def transformYValues(predictions, encoder=None):
    """
    input = [[1, 1], [0, 1]]
    output = [1, 1, 0, 1]
    """
    y_pred = []
    for lista in predictions:
        if (encoder is not None):
            for elemento in encoder.inverse_transform(lista):
                y_pred.append(elemento)
        else:
            for elemento in lista:
                y_pred.append(elemento)
    return y_pred