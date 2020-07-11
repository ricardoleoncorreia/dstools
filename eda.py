import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataHandler:
    '''
    Saves the original dataset and keeps a copy to be processed with required filters.
    
    Attributes
    ----------
    original_data: pd.DataFrame
        Contains original data.
    processed_data: pd.DataFrame
        Contains a copy of the original data with required filters applied.
    
    Methods
    -------
    filter_data(**kwargs)
        Resets and applies filters to the original data copy.
    '''
    
    def __init__(self, data):
        self.original_data = data.copy()
        self.processed_data = data.copy()

    def filter_data(self, **filters):
        '''
        Resets and applies filters to the original data copy.
        
        Parameters
        ----------
        filters: dict
            Dictionary with required filters. Keys should be provided as below:
            - reset_data: boolean.
                Resets processed data to original value.
            - fixed_values: dict.
                Each key is the column name and its value should be a list
                of coincidences.
            - columns: string list.
                Columns to keep.
            - drop_invalid: boolean.
                Drops invalid instances.
            - drop_missing: Boolean.
                Drops instances with missing values.
            - "max_" keys: int or float
                Applies a max filters to selected column. To be valid,
                the name of the column must be prefixed with "max_"
            - "min_" keys: int or float
                Applies a min filters to selected column. To be valid,
                the name of the column must be prefixed with "min_"
            
        Examples
        --------
        filters = {
            'reset_data': True,
            'fixed_values': {
                'property_type': ['Departamento', 'PH', 'Casa'],
                'l2': ['Capital Federal']
            },
            'columns': ['rooms', 'bedrooms', 'bathrooms', 'surface_total', 'surface_covered', 'price'],
            'max_surface_total' : 1000,
            'min_surface_total' : 15,
            'drop_invalid': True,
            'drop_missing': True,
            'max_price' : 4000000
        }
        data_handler.filter_data(**filters)
        '''
        if 'reset_data' in filters and filters['reset_data']:
            self._reset_data()
        if 'fixed_values' in filters:
            self._filter_rows(filters['fixed_values'])
        if 'columns' in filters:
            self._filter_columns(filters['columns'])
        if 'drop_invalid' in filters:
            self._drop_invalid()
        if 'drop_missing' in filters:
            self._drop_missing()
        self._apply_boundaries(**filters)
    
    def _reset_data(self):
        self.processed_data = self.original_data

    def _filter_rows(self, rows):
        for prop in rows:
            mask = self.processed_data[prop].isin(rows[prop])
            self.processed_data = self.processed_data[mask]

    def _filter_columns(self, columns):
        self.processed_data = self.processed_data[columns] 

    def _drop_invalid(self):
        mask = self.processed_data.surface_covered <= self.processed_data.surface_total
        self.processed_data = self.processed_data[mask]
    
    def _drop_missing(self):
        self.processed_data.dropna(inplace=True)
        
    def _apply_boundaries(self, **boundaries):
        self._apply_max_boundaries(**boundaries)
        self._apply_min_boundaries(**boundaries)
    
    def _apply_max_boundaries(self, **boundaries):
        max_boundaries = {prop.replace('max_', ''):boundaries[prop] for prop in boundaries if 'max_' in prop and prop.replace('max_', '') in self.processed_data.columns}
        for prop in max_boundaries:
            max_mask = self.processed_data[prop] <= max_boundaries[prop]
            self.processed_data = self.processed_data[max_mask]
    
    def _apply_min_boundaries(self, **boundaries):
        min_boundaries = {prop.replace('min_', ''):boundaries[prop] for prop in boundaries if 'min_' in prop and prop.replace('min_', '') in self.processed_data.columns}
        for prop in min_boundaries:
            min_mask = self.processed_data[prop] >= min_boundaries[prop]
            self.processed_data = self.processed_data[min_mask]
            
    def calculate_correlation(self, price_only=False):
        '''
        Plots a heatmap with correlations for all numeric variables.
        
        Parameters
        ----------
        price_only: boolean, optional.
            When true, only plots correlation between numeric values
            with the feature price.
        '''
        corr_summary = pd.DataFrame()
        if price_only:
            corr_summary['price_all'] = self.processed_data.corr().drop(['price'], axis=1).iloc[-1]
        else:
            corr_summary = self.processed_data.corr()
        sns.heatmap(corr_summary, annot=True)
        plt.show()
