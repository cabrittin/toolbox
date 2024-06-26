"""
@name: pandas_mod.py
@description:

Defines dataframe accessor class with added functionality

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import pandas as pd

@pd.api.extensions.register_dataframe_accessor("pdmod")
class PDMod:
    """ 
    Extend Pandas Dataframe by registing the accessor 'PDMod'

    Methods
    -------
    column_has_values: returns dataframe row indicies with specified value
    column_startswith: returns dataframe row indicies where column value starts with string
    column_contains: returns dataframe row indicies where column value contains string

    """
    def __init__(self, pandas_obj):
        self.dataset = pandas_obj
    
    def column_has_values(self,column,values):
        """
        Returns row indicies where column has specified value

        Parameters
        ----------
        column: str            Column name
        values: str, list
            Values to query

        Returns
        -------
        idx: list
            List of row indicies with specified value
        """
        idx = None
        if type(values) == str:
            idx = self.dataset.index[self.dataset[column] == values].tolist()
        elif type(values) == list:
            idx = self.dataset.index[self.dataset[column].isin(values)].tolist()
        else:
            raise ValueError("values are not str or list")
        return idx
    
    def column_startswith(self,column,value):
        """
        Returns row indicies where column starts with value

        Parameters
        ----------
        column: str
            Column name
        value: str, list
            Value to query

        Returns
        -------
        idx: list
            List of row indicies with specified value
        """
 
        idx = None 
        if type(value) == str:
            idx = self.dataset.index[getattr(self.dataset,column).str.startswith(value,na=False)]
            idx = idx.tolist()
        else:
            raise ValueError("value should be str")
        return idx

    def column_contains(self,column,value):
        """
        Returns row indicies where column starts with value

        Parameters
        ----------
        column: str
            Column name
        value: str, list
            Value to query

        Returns
        -------
        idx: list
            List of row indicies with specified value
        """
        idx = None 
        if type(value) == str:
            idx = self.dataset.index[getattr(self.dataset,column).str.contains(value,na=False)]
            idx = idx.tolist()
        else:
            raise ValueError("value should be str")
        return idx

    def value_index_map(self,column):
        """
        Determines the unique values of 'column' and then returns a dictionary of
        the row indicies of unique value.

        Parameters
        ----------
        column: str
            Column name

        Returns
        -------
        val_dict: dict
            Dictionary where key is the unique values of column and values are the
            list of row indicies where each unique value occurs.
        """
        unique_vals  = self.dataset[column].unique().tolist()
        val_dict = {}
        for (i,_val) in enumerate(unique_vals):
            jdx = self.dataset.index[self.dataset[column] == _val].tolist()
            val_dict[_val] = jdx
        return val_dict
    
    def set_target(self,value,col,key='target'):
        """
        Sets a binary target column in the dataframe where target = 1 if 'value'
        is in 'col' and target = 0 otherwise.

        Parameters:
        -----------
        value: str,int,float
            value that serves as the target. Type must match col
        col: str,list
            If str, name of column. If list, look that value is in at least one
            one of the columns in the list
        key: str, optional, default('target')
            Name of the new target column that will be added to the dataframe
        """
        if type(col) != list: col = [col]
        z = np.zeros(self.dataset.shape[0])
        for idx,row in self.dataset.iterrows():
            z[idx] = value in [row[c] for c in col]
        self.dataset[key] = z
 
    def get_unique_values(self,cols):
        """
        Gets unique values in columns 'cols'. Primarily used when searching
        multiple columns

        Parameters:
        -----------
        col: str,list
            If str, name of column. If list, multiple columns will be searched
            for unique values
       
        Returns:
        -------
        vals: list
            List of unique values in cols

        """
        vals = []
        if type(col) != list: col = [col]
        for c in cols: vals += df[c].unique().tolist()
        return list(set(vals))

