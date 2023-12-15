# dataloader.py

from os.path import join
from pandas import read_feather, DataFrame, Series, Index, Timestamp

class DataLoader:
    '''
    A class to organize and preprocess the data.
    Automatically calculate the adjusted close price, return and future return
    when initialized.
    '''
    def __init__(self, path: str) -> None:
        '''
        Parameters
        ----------
        path : str
            The path of the data.
        '''
        self.stk_daily = read_feather(join(path, 'stk_daily.feather'))
        self.stock_list = self.stk_daily['stk_id'].unique()
        self.date_range = self.stk_daily['date'].unique()
        self.stk_daily.set_index(['stk_id', 'date'], inplace=True)
        self.stk_daily.sort_index(inplace=True)
        self._load()
    
    def _load(self) -> None:
        '''
        Calculate the adjusted close price, return and future return.
        '''

        self._calculate_adj('close')
        self._calculate_return('adj')
        self.stk_daily['ret'].fillna(0, inplace=True)
        self.stk_daily['future_ret'] = self.stk_daily['ret'].shift(-1)
        self.stk_daily['future_ret'].fillna(0, inplace=True)

    def _calculate_adj(self, price: str, col_name='adj') -> Series:
        '''
        Calculate the adjusted price of a stock.

        Parameters
        ----------
        price : str
            The price to be adjusted.

        col_name : str, optional
            The name of the column to be added, by default 'adj'.
        
        Returns
        -------
        Series
            A series of adjusted price.
        '''
        self.stk_daily[col_name] = self.stk_daily[price] * self.stk_daily['cumadj']
        return self.stk_daily[col_name]
    
    def _calculate_return(self, price: str, col_name='ret') -> Series:
        '''
        Calculate the return of a stock.

        Parameters
        ----------
        price : str
            The price to be adjusted.

        col_name : str, optional
            The name of the column to be added, by default 'ret'.
        
        Returns
        -------
        Series
            A series of return.
        '''
        self.stk_daily[col_name] = self.timeseries[price].pct_change()
        return self.stk_daily[col_name]
   
    def snap(self,
             start_date: Timestamp = None,
             end_date: Timestamp = None
        ) -> DataFrame:
        '''
        Slice the time period for backtest.

        Parameters
        ----------
        start_date : Timestamp
            The start date of the backtest.
            By default None, which means the first date of the data.

        end_date : Timestamp
            The end date of the backtest.
            By default None, which means the last date of the data.

        Returns
        -------
        DataFrame
            A DataFrame of the data.
        '''
        if start_date is None:
            start_date = self.date_range.min()
        if end_date is None:
            end_date = self.date_range.max()

        res = self.stk_daily.copy()
        return res[(res.index.get_level_values('date') >= start_date) & \
            (res.index.get_level_values('date') <= end_date)].copy()
       
    def get_data(self) -> DataFrame:
        '''
        Return a DataFrame of the market data.

        Returns
        -------
        DataFrame
            A DataFrame of the market data.
        ''' 
        return self.stk_daily.copy()
    
    def get_stock_list(self) -> Index:
        '''
        Return a list of stocks.

        Returns
        -------
        list
            A series of index of stocks.
        '''
        return self.stock_list
    
    def get_stock(self, stk_id: str | list) -> DataFrame:
        '''
        Return a DataFrame of a stock or a list of stocks.

        Parameters
        ----------
        stk_id : str | list
            Stock id or a list of stock ids.
        
        Returns
        -------
        DataFrame
            A DataFrame of a stock or a list of stocks.
        '''
        if isinstance(stk_id, str):
            return self.stk_daily[self.stk_daily['stk_id'] == stk_id]
        else:
            return self.stk_daily[self.stk_daily['stk_id'].isin(stk_id)]
    
    def get_date_list(self) -> Series:
        return self.date_range
    
    def get_date(self, date) -> DataFrame:
        '''
        Return a DataFrame of a date.

        Parameters
        ----------
        date : str
            Date in format of 'YYYY-MM-DD'.
        
        Returns
        -------
        DataFrame
            A DataFrame of a date.
        '''
        return self.stk_daily[self.stk_daily['date'] == date]
    
    def get_fields(self) -> Index:
        '''
        Return a list of fields in the DataFrame.

        Returns
        -------
        list
            A series of Index of fields.
        '''
        return self.stk_daily.columns
     
    @property
    def index(self) -> Index:
        '''
        Return the index of the DataFrame.

        Returns
        -------
        Index
            The index of the DataFrame.
        '''
        return self.stk_daily.index

    @property
    def timeseries(self) -> DataFrame:
        '''
        Group the DataFrame by stock id for time series analysis.
        
        Returns
        -------
        DataFrame
            A DataFrame grouped by stock id.
        '''
        return self.stk_daily.groupby(level='stk_id', group_keys=False)

    @property
    def crosssection(self) -> DataFrame:
        '''
        Group the DataFrame by date for cross-sectional analysis.
        
        Returns
        -------
        DataFrame
            A DataFrame grouped by date.
        '''
        return self.stk_daily.groupby(level='date', group_keys=False)