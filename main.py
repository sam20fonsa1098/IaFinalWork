"""
Imports
"""
import pandas as pd
import shap
import matplotlib.pyplot as plt

from utils import *
from customKeras import *

from tensorflow.keras.utils import plot_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
"""
Declaring MinMaxScaler
"""
sc = MinMaxScaler()
"""
Read csv data
"""
df = pd.read_csv('./trains-transformed.csv')
columns = list(df.columns)[:-1]
"""
Get X and y values
"""
X = df.iloc[:, :-1].values
X = transform_data(X)
X = sc.fit_transform(X)

y = df.iloc[:,-1].values
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
"""
Using customStratifiedKfold
"""
vetor_X_train, vetor_y_train, vetor_X_test, vetor_y_test = customStratifiedKFold(X, y)
"""
Count: max = 5
"""
contador, i = 0, 1
"""
Y values
"""
y_pred_all, y_test_all = [], []
"""
Loop while to predict each folder
"""
while(contador < len(vetor_X_train)):
    X_train = vetor_X_train[contador]
    y_train = vetor_y_train[contador]
    X_test = vetor_X_test[contador]
    y_test = vetor_y_test[contador]
    y_test_all.append(y_test)
    
    estimator = baseline_model(input_value=X_train.shape[-1])
    estimator.fit(X_train, y_train, epochs=1000, batch_size=20)

    e = shap.DeepExplainer(estimator, X_train)

    shap_values_test = e.shap_values(X_test)
    shap_values_train = e.shap_values(X_train)

    for index_X_test in range(len(X_test)):
        # plot the feature attributions
        shap.force_plot(e.expected_value[0], 
                        shap_values_test[0][index_X_test,:], 
                        X_test[index_X_test], 
                        link="logit", 
                        matplotlib=True,
                        show=False,
                        feature_names=columns)

        plt.savefig(f'shapeImages/individual_force_plot{i}.pdf',
                    bbox_inches = 'tight')
        plt.close()
        
        i = i + 1

    y_pred = estimator.predict(X_test)
    y_pred_all.append(y_pred)

    del estimator

    contador = contador + 1

"""
Plot model
"""
plot_model(baseline_model(), 
           to_file='model_plot.png', 
           show_shapes=True, 
           show_layer_names=True)

y_pred_all, y_test_all = transformYValues(y_pred_all), transformYValues(y_test_all) 
y_pred_all = [round(y_pred_value[0]) for y_pred_value in y_pred_all]

print(metrics.classification_report(y_test_all, y_pred_all))

print(metrics.confusion_matrix(y_test_all, y_pred_all))

estimator = baseline_model(input_value=X.shape[-1])
estimator.fit(X, y, epochs=1000, batch_size=20)

e = shap.DeepExplainer(estimator, X)

shap_values = e.shap_values(X)

shap.decision_plot(e.expected_value[0], 
                   shap_values[0],
                   X_train,
                   link='logit',
                   show=False,
                   feature_names=columns)

plt.savefig(f'shapeImages/decision_plot.pdf',
            bbox_inches = 'tight')
plt.close()

for index_X in range(len(X)):
    # plot the feature attributions
    shap.decision_plot(e.expected_value[0], 
                       shap_values[0][index_X,:], 
                       X[index_X], 
                       link="logit", 
                       show=False,
                       feature_names=columns)

    plt.savefig(f'shapeImages/individual_decision_plot{index_X + 1}.pdf',
                bbox_inches = 'tight')
    plt.close()