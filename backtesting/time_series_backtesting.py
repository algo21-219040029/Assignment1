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

class TimeSeriesBacktesting(BaseBacktesting):
    """
    定期调仓，各品种独立时间序列回测系统
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
                 start: str = None,
                 end: str = None,
                 **kwargs) -> None:
        """Constructor"""
        super().__init__(rate=rate,
                         period=period,
                         interest=interest,
                         contract=contract,
                         price=price,
                         rebalance_num=rebalance_num,
                         start=start,
                         end=end,
                         **kwargs)

    def run_backtesting(self, tqdm_flag: bool = True) -> None:
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

        # 导入参数
        params = self.get_params()
        rate = params['rate']
        period = params['period']
        interest = params['interest']
        contract = params['contract']
        price = params['price']
        rebalance_num = params['rebalance_num']
        start = params['start']
        end = params['end']

        # 获取权重
        weights = self.weights

        # 获取收盘价
        price_df = self.get_continuous_field(contract, price, rebalance_num, 'continuous_price')
        price_df = price_df['2010': ]
        if start != None:
            price_df = price_df[start:]
        if end != None:
            price_df = price_df[:end]

        common_index = weights.index.intersection(price_df.index)
        common_columns = weights.columns.intersection(price_df.columns)
        weights = weights.loc[common_index][common_columns]
        price_df = price_df.loc[common_index][common_columns]

        if start == None:
            start = price_df.index[0].strftime("%Y%m%d")
            self.set_params(start=start)
        if end == None:
            end = price_df.index[-1].strftime("%Y%m%d")
            self.set_params(end=end)

        # 根据起始日平滑进行遍历
        for shift in range(period):

            # 生成hold_datetime_list
            hold_datetime_list = []
            for i in range(len(price_df)):
                if (i - shift) % period == 0:
                    date = price_df.index[i]
                    hold_datetime_list.append(date)




