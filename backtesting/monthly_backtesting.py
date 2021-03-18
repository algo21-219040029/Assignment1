import os
from pathlib import Path
from pandas import DataFrame
from typing import Tuple, Dict
import matplotlib.pyplot as plt

from backtesting.metrics import get_metrics
from backtesting.base import BaseBacktesting

from backtesting.backtest_utility import execute_simple_backtesting, \
    execute_compound_backtesting

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题,或者转换负号为字符串

class LongShortMonthlyBacktesting(BaseBacktesting):
    """
    多空组合，月末调仓回测系统
    """
    def __init__(self,
                 rate: float = 0,
                 period: str ='end',
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

    def run_backtesting(self) -> Tuple[Dict]:
        """
        执行回测

        Parameters
        ----------


        Returns
        -------
        backtest_result: Tuple[Dict]
                        回测结果
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

        # 获取收盘价,如果扩展开盘价或收盘价，则需要进一步扩展
        price_df = self.get_continuous_field(contract, price, rebalance_num, 'continuous_price')
        # price_df = price_df['2010': '2018-09-21']

        # 品种行业对应表
        symbol_industry_map = self.symbol_industry_map.set_index('underlying_symbol')

        # 校正权重和收盘价
        common_index = weights.index.intersection(price_df.index)
        common_columns = weights.columns.intersection(price_df.columns)
        weights = weights.loc[common_index][common_columns]
        price_df = price_df.loc[common_index][common_columns]

        dts = price_df.index.to_series(index=range(len(price_df))).to_frame('datetime')
        dts['year'] = dts['datetime'].dt.year
        dts['month'] = dts['datetime'].dt.month

        # 确定调仓日期，月初调仓或者月末调仓
        hold_datetime_list = []
        if period == 'end':
            hold_datetime_list = \
            dts.sort_values(by=['year', 'month'], ascending=True).groupby(['year', 'month'], as_index=False)[
                'datetime'].nth(-1).tolist()
        elif period == 'start':
            hold_datetime_list = \
                dts.sort_values(by=['year', 'month'], ascending=True).groupby(['year', 'month'], as_index=False)[
                    'datetime'].nth(0).tolist()

        if interest == 'simple':
            value_df, hold_value_df, weight_df, hold_weight_df, turnover_df, \
            hold_turnover_df, profit_df, hold_profit_df = execute_simple_backtesting(weights,
                                                                                     price_df,
                                                                                     init_total_value,
                                                                                     hold_datetime_list,
                                                                                     rate)
        elif interest == 'compound':
            value_df, hold_value_df, weight_df, hold_weight_df, turnover_df,\
            hold_turnover_df, profit_df, hold_profit_df = execute_compound_backtesting(weights,
                                                                                       price_df,
                                                                                       init_total_value,
                                                                                       hold_datetime_list,
                                                                                       rate)

        else:
            raise ValueError("interest must be simple or compound!")

        self.value_df = value_df
        self.hold_value_df = hold_value_df
        self.weight_df = weight_df
        self.hold_weight_df = hold_weight_df
        self.turnover_df = turnover_df
        self.hold_turnover_df = hold_turnover_df
        self.profit_df = profit_df
        self.hold_profit_df = hold_profit_df

        detail_result = {'value': value_df,
                         'hold_value': hold_value_df,
                         'weight': weight_df,
                         'hold_weight': hold_weight_df,
                         'turnover': turnover_df,
                         'hold_turnover': hold_turnover_df,
                         'profit': profit_df,
                         'hold_profit': hold_profit_df}

        metrics_result = get_metrics(
                                     weight_df,
                                     hold_weight_df,
                                     symbol_industry_map,
                                     turnover_df,
                                     hold_turnover_df,
                                     init_total_value,
                                     profit_df,
                                     hold_profit_df,
                                     interest)

        backtest_result = {'metrics': metrics_result,
                  'detail': detail_result}
        self.profit_series = profit_df.sum(axis=1)
        self.cum_profit_series = self.profit_series.cumsum()
        self.backtest_result = backtest_result

        return backtest_result

