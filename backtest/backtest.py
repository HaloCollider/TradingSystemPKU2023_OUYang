# backtest.py

from .dataloader import DataLoader
from .strategy import Strategy
from types import FunctionType
from pandas import DataFrame, Series, concat, \
    to_datetime, Grouper, Timedelta
from numpy import array
import matplotlib.pyplot as plt

class Backtest:
    '''
    A backtest class to backtest the strategy.
    '''
    def __init__(self,
                 strategy: Strategy,
                 dataloader: DataLoader,
                 start_date = None,
                 end_date = None,
                 freq: str | Grouper = 'D',
                 slippage: float = 0.005,
                 position: float = 1e6,
                 long_pct: float = 0.1,
                 short: bool = False,
                 short_pct: float = 0,
                 weight_func: FunctionType = None,
                 short_weight_func: FunctionType = None
        ) -> None:
        '''
        Parameters
        ----------
        strategy : Strategy
            A Strategy object.
        
        dataloader : DataLoader
            A DataLoader object.
        
        start_date : str, optional
            The start date of the backtest, by default None,
            which means the first date of the data.
        
        end_date : str, optional
            The end date of the backtest, by default None,
            which means the last date of the data.
        
        freq : str or Grouper, optional
            The frequency of the backtest, which is supported by pandas.
            By default 'D', which means daily.
            if type is str, it will be used as the freq parameter of a Grouper.
            If type is Grouper, it will be used to the groupby object.
        
        slippage : float, optional
            The slippage (transaction cost) of the backtest, by default 0.005.
        
        position : float, optional
            The initial position of the backtest, by default 1e7.
        
        long_pct : float, optional
            The percentage of stocks to be long, by default 0.1.
        
        short : bool, optional
            Whether to short, by default False.

        short_pct : float, optional
            The percentage of stocks to be short, by default 0
            (must set the argument short true).

        weight_func : function, optional
            The function to calculate the portfolio weight, by default None,
            which means equal weight for stocks in the portfolio.
            Otherwize, the function should take a DataFrame of the signals
            as input and return an array of the portfolio weight.
        
        short_weight_func : function, optional
            The function to calculate the portfolio weight for short positions,
            by default None, which means it is the same as weight_func.
        '''
        self.strategy = strategy
        self.dataloader = dataloader

        if isinstance(start_date, type(None)):
            start_date = self.dataloader.date_range.min()
        self.start_date = to_datetime(start_date)
        if isinstance(end_date, type(None)):
            self.end_date = self.dataloader.date_range.max()
        self.end_date = to_datetime(end_date)

        if isinstance(freq, str):
            self.freq = Grouper(freq=freq, level='date', label='left')
        else:
            assert isinstance(freq, Grouper), \
                'freq should be a str or a Grouper.'
            self.freq = freq
            self.freq.level = 'date'
            self.freq.label = 'left'
        
        self.slippage = slippage
        self.position = position
        self.long_pct = long_pct
        self.short = short
        self.short_pct = short_pct
        assert long_pct + short_pct <= 1, \
            'The sum of long_pct and short_pct should be less than 1.'
        assert long_pct >= 0 and short_pct >= 0, \
            'long_pct and short_pct should be greater than 0.'
        
        self.weight_func = weight_func
        self.short_weight_func = short_weight_func
    
    def run(self, verbose=True) -> DataFrame:
        '''
        Run the backtest.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print the log, by default True.
        '''
        self._test_data = self.dataloader.snap(self.start_date, self.end_date)
        self._signal = self.strategy.signal(self.dataloader)
        self._signal.index = self.dataloader.index
        self._signal.name = 'signal'
        self._backtest = self.backtest(
            self._signal, self._test_data, verbose=verbose
        )
        self._evaluation = self.evaluate(self._backtest, verbose=verbose)
        return self._evaluation
    
    def backtest(
            self,
            signal,
            test_data,
            verbose=True
        ) -> DataFrame:
        '''
        Evaluate the backtest result.

        Parameters
        ----------
        signal : DataFrame
            A DataFrame of the signals.

        test_data : DataFrame
            A DataFrame of the test data.
        
        verbose : bool, optional
            Whether to print the log, by default True.
        
        Returns
        -------
        DataFrame
            A DataFrame of the backtest result.
        '''

        if verbose:
            print('Building backtest data...')

        test_data = test_data.copy()
        test_data = test_data.join(signal, how='left')
        test_data_grouped = test_data.groupby(level='date', group_keys=False)

        if verbose:
            print('Calculating ranks...')
        
        test_data['rank'] = test_data_grouped['signal'].rank(ascending=False)
        test_data['top'] = test_data_grouped['rank'].transform(
            lambda x: x <= int(len(x) * self.long_pct)
        )
        if self.short:
            test_data['bottom'] = test_data_grouped['rank'].transform(
                lambda x: 1 - x <= int(len(x) * self.short_pct)
            )

        if verbose:
            print('Calculating weights...')
        
        test_data['weight'] = 0
        if isinstance(self.weight_func, type(None)):
            weight_func = lambda x: 1 / len(x)
        else:
            weight_func = self.weight_func
        long_weight = test_data[test_data['top']].groupby(
                level='date', group_keys=False).transform(
                weight_func
            )
        test_data.loc[test_data['top'], 'weight'] = long_weight
        
        if self.short:
            if isinstance(self.short_weight_func, type(None)):
                short_weight_func = weight_func
            else:
                short_weight_func = self.short_weight_func
            short_weight = test_data[test_data['bottom']].groupby(
                    level='date', group_keys=False).transform(
                    short_weight_func
                )
            test_data.loc[test_data['bottom'], 'weight'] = short_weight

        if verbose:
            print(f'Aggregating results with frequency being {self.freq.freq}...')
            print('(Note that a high frequency would take a long time...)')
        
        if self.freq.freq != 'D':
            test_data = test_data.groupby(
                [self.freq, 'stk_id'], group_keys=False
                ).agg(
                    {'weight': 'first', 'future_ret': lambda x: (1 + x).prod() - 1}
                )

        test_data['signal_ret'] = test_data['weight'] * test_data['future_ret']
        # slippage adjustment, calculated by the absolute change of weights
        delta_weight = test_data.groupby(level='stk_id')['weight'].diff()
        delta_weight[delta_weight.isna()] = test_data.groupby(
            level='stk_id'
            )['weight'].first()
        test_data['signal_ret'] -= self.slippage * abs(delta_weight)
        portfolio_ret = test_data.groupby(level='date')['signal_ret'].sum()
        net_val = (1 + portfolio_ret).cumprod() * self.position

        benchmark_ret = test_data.groupby(level='date')['future_ret'].mean()
        benchmark_ret -= self.slippage
        benchmark_val = (1 + benchmark_ret).cumprod() * self.position

        res = DataFrame({
            'portfolio_ret': portfolio_ret,
            'net_val': net_val,
            'benchmark_ret': benchmark_ret,
            'benchmark_val': benchmark_val
        })

        # if the end date is not the last date of the data,
        # merge the backtest result with the last date of the data
        # since we aggregate the future return to the present
        # all the values should be shifted one period forward
        if self.end_date != res.index[-1]:
            temp = DataFrame({
                'portfolio_ret': 0,
                'net_val': 0,
                'benchmark_ret': 0,
                'benchmark_val': 0
            }, index=[self.end_date]
            )
            res = concat([res, temp])
        
        res = res.shift(1)
        res.iloc[0] = [0, self.position, 0, self.position]
        # validate the start date
        res.rename(index={res.index[0]: self.start_date}, inplace=True)

        return res
    
    def evaluate(self, backtest, verbose=True) -> DataFrame:
        '''
        Evaluate the backtest result.

        Parameters
        ----------
        backtest : DataFrame
            A DataFrame of the backtest result.

        verbose : bool, optional
            Whether to print the log, by default True.
        
        Returns
        -------
        DataFrame
            A DataFrame of the evaluation result.
        '''
        if verbose:
            print('Evaluating the backtest result...')
        
        freq = Series(backtest.index).diff().mean()
        scale = Timedelta('365d') / freq
        timespan = freq * len(backtest)

        start_date = self.start_date
        end_date = self.end_date

        frequency = self.freq.freq

        initial_position = self.position
        ending_position = backtest['net_val'].iloc[-1]

        annual_return = (
            1 + backtest['portfolio_ret']
            ).prod() ** (Timedelta('365d') / timespan) - 1
        
        benchmark_annual_return = (
            1 + backtest['benchmark_ret']
            ).prod() ** (Timedelta('365d') / timespan) - 1

        excessive_annual_return = annual_return - benchmark_annual_return
        
        annual_volatility = backtest['portfolio_ret'].std() * scale ** 0.5

        sharpe_ratio = annual_return / annual_volatility

        max_drawdown = (backtest['net_val'] /
            backtest['net_val'].cummax() - 1).min()
        
        winning_rate = (backtest['portfolio_ret'] > 0).mean()

        evaluation = {
            'Start Date': start_date.strftime('%Y-%m-%d'),
            'End Date': end_date.strftime('%Y-%m-%d'),
            'Frequency': frequency,
            'Initial Position': initial_position,
            'Ending Position': ending_position,
            'Annual Return': annual_return,
            'Benchmark Annual Return': benchmark_annual_return,
            'Excessive Annual Return': excessive_annual_return,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Winning Rate': winning_rate,
        }
        index = list(evaluation.keys())
        vals = list(evaluation.values())

        return DataFrame(
            vals, index=index, columns=['Evaluation']
        )

    def plot(self,
             backtest: DataFrame = None,
             benchmark: bool = True,
             figsize: tuple | list = (10, 5),
             **kwargs) -> None:
        '''
        Plot the backtest result.

        Parameters
        ----------
        backtest : DataFrame, optional
            A DataFrame of the backtest result, by default None,
            which means the result has been saved within the class.
        
        benchmark : bool, optional
            Whether to plot the benchmark, by default True.

        **kwargs : dict
            The parameters for matplotlib.pyplot.plot.
        '''
        if isinstance(backtest, type(None)):
            assert self.__getattribute__('backtest') is not None, \
            'Run the backtest first.'
            backtest = self._backtest
        
        plt.figure(figsize=figsize)
        plt.plot(backtest['net_val'], **kwargs)
        if benchmark:
            plt.plot(backtest['benchmark_val'], **kwargs)
        plt.title('Backtest Result')
        plt.xlabel('Date')
        plt.ylabel('Net Value')
        plt.legend(['Strategy Net Value', 'Benchmark Net Value'])
        plt.grid()
        plt.show()
