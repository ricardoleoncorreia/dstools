import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.model_selection import train_test_split


class RegressionModeling:
    '''
    Helper to standardize the Machine Learning model study.
    '''
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size=test_size
        self.random_state=random_state
        self.models_errors = {}
        self.benchmark=None
        self.best_hyperparams = {}
        
    def study_dataset(self, X, y, models_info, metric='rmse', plot_residuals=False, kde=True, plot_error_metrics=False):
        '''
        Executes the study for specified models.
        '''
        self.models_info = models_info
        self._train_test_split(X, y)
        self._get_models_results(plot_residuals=plot_residuals, metric=metric, kde=kde)
        
        if plot_error_metrics:
            self._plot_error_metrics()
    
    def _train_test_split(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
    
    def _get_models_results(self, metric, plot_residuals, kde):
        for model_name in self.models_info:

            # Select model
            print(f'Modelo: {model_name}')
            model = self.models_info[model_name]

            # Train model
            model.fit(self.X_train, self.y_train)

            # Predict results
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)

            # Calculate errors
            model_errors = self._calculate_model_errors(y_train_pred, y_test_pred, metric=metric)
            self.models_errors[model_name] = model_errors

            # Plot results if required
            if plot_residuals:
                self._plot_model_errors(y_train_pred, y_test_pred, kde)
            
            # If linear model, save coefficients
            if isinstance(model, LinearRegression):
                self._save_linear_model_coef(model.coef_)
            
            # If DT model, save feature importances
            if isinstance(model, DecisionTreeRegressor):
                self._save_dt_model_feature_importances(model.feature_importances_)

        self._convert_errors_to_dataframe(metric=metric)
    
    def _calculate_model_errors(self, y_train_pred, y_test_pred, metric):
        # Set values
        train_error = np.inf
        test_error = np.inf
        metric_name = 'Error'
        
        # Calculate errors according to selected metric
        if metric == 'rmse':
            train_error = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            test_error = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            metric_name = 'Raíz del error cuadrático medio'
        elif metric == 'mae':
            train_error = median_absolute_error(self.y_train, y_train_pred)
            test_error = median_absolute_error(self.y_test, y_test_pred)
            metric_name = 'Error absoluto medio'

        # Print calculated values
        print(f'{metric_name} en Train: {train_error}')
        print(f'{metric_name} en Test: {test_error}\n')
        return [train_error, test_error]
    
    def _plot_model_errors(self, y_train_pred, y_test_pred, kde):
        # Prepare subplots
        plt.figure(figsize = (13,5))

        # Plot error distribution
        plt.subplot(1,2,1)
        sns.distplot(self.y_train - y_train_pred, bins = 20, kde=kde, label = 'train')
        sns.distplot(self.y_test - y_test_pred, bins = 20, kde=kde, label = 'test')
        plt.xlabel('errores')
        plt.legend()

        # Plot real vs predictions values in test set
        ax = plt.subplot(1,2,2)
        ax.scatter(self.y_test,y_test_pred, s =2)

        lims_min = np.min([ax.get_xlim(), ax.get_ylim()])
        lims_max = np.max([ax.get_xlim(), ax.get_ylim()])
        lims = [lims_min, lims_max]

        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        plt.xlabel('y (test)')
        plt.ylabel('y_pred (test)')

        # Show plots
        plt.tight_layout()
        plt.show()
    
    def _convert_errors_to_dataframe(self, metric):
        indexes = [f'{metric}_train (USD)', f'{metric}_test (USD)']
        self.models_errors_dataframe = pd.DataFrame(self.models_errors, index=indexes)
    
    def _plot_error_metrics(self):
        ax = self.models_errors_dataframe.transpose().plot.bar(rot=0, figsize=(10,5))
        ax.legend(loc='lower right')
        if self.benchmark is not None:
            for error_value in self.benchmark.to_dict():
                color = 'r' if 'train' in error_value else 'g'
                plt.axhline(y=self.benchmark[error_value],linewidth=1, color=color, label=error_value)
                handles, _ = ax.get_legend_handles_labels()
                plt.legend(handles = handles)
        plt.show()
        
    def _save_linear_model_coef(self, linear_coef):
        self.linear_coef = pd.DataFrame({
            'predictors': self.X_train.columns,
            'coefficients': linear_coef
        })
    
    def _save_dt_model_feature_importances(self, dt_feature_importances):
        self.dt_feature_importances = pd.DataFrame({
            'predictors': self.X_train.columns,
            'feature_importances': dt_feature_importances
        })
        
    def set_benchmark(self, model_name, print_errors=True):
        '''
        Selects model as a benchmark. It helps to show the horizontal lines
        in model error comparison plots.
        '''
        self.benchmark = self.models_errors_dataframe.copy()[model_name]
        benchmark_names = [f'benchmark_{name}' for name in self.benchmark.index.copy().values]
        self.benchmark.index = benchmark_names
        if print_errors: print(self.benchmark)
