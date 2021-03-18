from pandas import DataFrame
from typing import Tuple, Dict

from backtesting.metrics import get_metrics
from backtesting.base import BaseBacktesting
from backtesting.backtest_utility import (execute_simple_backtesting,
                                            execute_compound_backtesting,
                                          rolling_backtest_result_analysis)

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题,或者转换负号为字符串

class LongShortPeriodBacktesting(BaseBacktesting):
    """
    定期调仓、起始日平滑回测系统
    需要的数据:
        1.因子值: DataFrame, index为交易时间(datetime), columns为品种(underlying_symbol), data为因子值
        2.品种价格指数: DataFrame, index为交易时间(datetime), columns为品种(underlying_symbol), data为主力连续合约价格
    设置的参数:
        1.调仓周期period
        2.交易费用rate
        3.单利or复利interest
        4.用于计算收益的连续
    """

    def __init__(self,
                 rate: float = 0,
                 period: int = 1,
                 interest: str = 'simple',
                 contract: str = 'main',
                 price: str = 'close',
                 rebalance_num: int = 1,
                 **kwargs) -> None:

        """Constructor"""
        super().__init__(rate=rate,
                         period=period,
                         interest=interest,
                         contract=contract,
                         price=price,
                         rebalance_num=rebalance_num,
                         **kwargs)

        self.profit_series: DataFrame = None
        self.cum_profit_series: DataFrame = None

        self.rolled_sharpe: float = None

    def run_backtesting(self, tqdm_flag: bool = True, start: str = None, end: str = None) -> Tuple[Dict]:
        """
        运行回测

        Parameters
        ----------
        tqdm_flag: bool, default True

        Returns
        -------
        None
        """

        # 预先检查
        if not isinstance(self.weights, DataFrame):
            try:
                self.prepare_weights()
            except:
                raise ValueError("init weights first!")

        params = self.get_params()
        rate = params['rate']
        period = params['period']
        interest = params['interest']
        contract = params['contract']
        price = params['price']
        rebalance_num = params['rebalance_num']

        # 初始资金
        init_total_value = 100000000
        self.init_total_value = init_total_value

        # 获取权重
        weights = self.weights

        # 获取收盘价
        price_df = self.get_continuous_field(contract, price, rebalance_num, 'continuous_price')
        price_df = price_df['2010': ]
        if start != None:
            price_df = price_df[start:]
        if end != None:
            price_df = price_df[:end]

        # 品种行业对应表
        symbol_industry_map = self.symbol_industry_map.set_index('underlying_symbol')

        # 校正权重和收盘价
        common_index = weights.index.intersection(price_df.index)
        common_columns = weights.columns.intersection(price_df.columns)
        weights = weights.loc[common_index][common_columns]
        price_df = price_df.loc[common_index][common_columns]

        rolling_metrics_result = {}
        rolling_profit = {}
        rolling_cumsum_profit = {}
        rolling_value = {}

        for shift in range(period):

            # 生成hold_datetime_list
            hold_datetime_list = []
            for i in range(len(price_df)):
                if (i - shift) % period == 0:
                    date = price_df.index[i]
                    hold_datetime_list.append(date)

            profit_df: DataFrame = DataFrame()
            hold_profit_df: DataFrame = DataFrame()
            # 执行回测
            if interest == 'simple':
                value_df, hold_value_df, weight_df, hold_weight_df, turnover_df, \
                hold_turnover_df, profit_df, hold_profit_df = execute_simple_backtesting(weights,
                                                                                         price_df,
                                                                                        init_total_value,
                                                                                        hold_datetime_list,
                                                                                        rate,
                                                                                         tqdm_flag)
            elif interest == 'compound':
                value_df, hold_value_df, weight_df, hold_weight_df, turnover_df, \
                hold_turnover_df, profit_df, hold_profit_df = execute_compound_backtesting(weights,
                                                                                           price_df,
                                                                                           init_total_value,
                                                                                           hold_datetime_list,
                                                                                           rate,
                                                                                           tqdm_flag)

            else:
                raise ValueError("interest must be simple or compound!")

            # # 分离多头部分
            # long_weight_df = weight_df.copy()
            # long_weight_df[(weight_df < 0.0) and weight_df.notnull()] = 0.0
            #
            # long_hold_weight_df = hold_weight_df.copy()
            # long_hold_weight_df[(hold_weight_df < 0.0) and hold_weight_df.notnull()] = 0.0
            #
            # long_turnover_df = turnover_df.copy()
            # long_turnover_df[(weight_df < 0.0) and weight_df.notnull()] = 0.0
            #
            # long_hold_turnover_df = hold_turnover_df.copy()
            # long_hold_turnover_df[(hold_weight_df < 0.0) and weight_df.notnull()] = 0.0


            # 生成总体的指标
            metrics_result = get_metrics(weight_df,
                                         hold_weight_df,
                                         symbol_industry_map,
                                         turnover_df,
                                         hold_turnover_df,
                                         init_total_value,
                                         profit_df,
                                         hold_profit_df,
                                         interest)

            # 生成long leg的指标

            rolling_metrics_result[shift] = metrics_result
            rolling_value[shift] = value_df
            rolling_profit[shift] = profit_df.fillna(0.0).sum(axis=1)
            rolling_cumsum_profit[shift] = profit_df.sum(axis=1).fillna(0.0).cumsum()

        metrics_result = rolling_backtest_result_analysis(rolling_metrics_result)
        self.backtest_result = {}
        self.backtest_result['metrics'] = metrics_result


        for shift in rolling_cumsum_profit:
            if shift == 0:
                rolled_profit_series = rolling_profit[shift]
                rolled_cumsum_profit_series = rolling_cumsum_profit[shift]
            elif shift > 0:
                rolled_profit_series += rolling_profit[shift]
                rolled_cumsum_profit_series += rolling_cumsum_profit[shift]

        rolled_profit_series = rolled_profit_series / len(rolling_profit)

        self.rolling_value = rolling_value
        self.rolling_profit = rolling_profit
        self.profit_series = rolled_profit_series
        self.cum_profit_series = self.profit_series.cumsum()
        self.cum_return_series = self.cum_profit_series / 100000000
        self.rolling_profit = rolling_profit
        self.rolling_cumsum_profit = rolling_cumsum_profit

        return metrics_result

