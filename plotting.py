import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def plot_learning_curve(X, y, model_name, hyperparam_name, hyperparam_values):
    '''
    Plots learning curves for the selected model
    '''
    rmse_train_list = []
    rmse_test_list = []
    hyperparam = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for hyperparam_value in hyperparam_values:
        hyperparam[hyperparam_name] = hyperparam_value
        
        if model_name == 'DT':    
            model = DecisionTreeRegressor(**hyperparam, random_state=42)
        elif model_name == 'KNN':
            model = KNeighborsRegressor(**hyperparam)
        elif model_name == 'KNN_weights':
            model = KNeighborsRegressor(**hyperparam, weights='distance')
        elif model_name == 'LR':
            model = LinearRegression(**hyperparam)
        
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_train_list.append(rmse_train)

        y_test_pred = model.predict(X_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        rmse_test_list.append(rmse_test)

    # Prepare subplots
    plt.figure(figsize = (8,5))

    # Plot error distribution
    # plt.subplot(1,2,1)
    plt.plot(hyperparam_values, rmse_train_list, 'o-', label='train')
    plt.plot(hyperparam_values, rmse_test_list, 'o-', label='test')
    plt.legend()
    plt.xlabel(hyperparam_name)
    plt.ylabel('RMSE')
    plt.title(f'RMSE Learning curve ({model_name})')

    plt.show()