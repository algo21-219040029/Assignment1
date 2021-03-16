import numpy as np
import pandas as pd
from pandas import DataFrame

from signals.GroupSignal.base import BaseGroupSignal

class GroupNLongSignal1(BaseGroupSignal):
    """
    分组做多品种信号生成器1(仅考虑因子值小于0的品种)

    Attributes
    __________
    group_num: int, default 5
                分组组数

    rank: str, default descending
            排序方法, descending表示因子值大的品种做多, ascending表示因子值小的品种做多

    """
    def __init__(self, group_num: int = 5, rank: str = 'descending') -> None:
        """Constructor"""
        super().__init__(group_num=group_num, rank=rank)

    def transform(self) -> DataFrame:
        """
        生成信号, 1表示做多, -1表示做空,
        Returns
        -------
        signal_df: DataFrame
                    信号DataFrame, index为交易时间, columns为品种代码, data为信号值-
        """

        if not isinstance(self.factor_data, DataFrame):
            raise ValueError("Please specify the factor_data first!")
        else:
            factor_data = self.factor_data

        if not isinstance(self.commodity_pool, DataFrame):
            raise ValueError("Please specify the commodity pool first!")
        else:
            commodity_pool = self.commodity_pool

        factor_data[~commodity_pool] = np.nan
        factor_data[factor_data >= 0.0] = np.nan
        params = self.get_params()
        group_num = params['group_num']
        rank = params['rank']

        if rank == 'ascending':
            factor_data = - factor_data

        def modified_qcut(series, q: int, labels):
            """
            修正的quct
            Parameters
            ----------
            series: 排序值

            Returns
            -------
            组别: Series
            """
            # 如果有效数据个数小于组别，则全部为0，即不持仓
            if series.count() < q:
                new_series = pd.Series([0.0] * len(series))
                new_series.index = series.index
                return new_series
            else:
                return pd.qcut(x=series, q=q, labels=labels)

        signal_df = factor_data.rank(axis=1, ascending=False, na_option='keep', method='first') \
            .apply(func=modified_qcut, axis=1,
                   q=group_num, labels=range(1, group_num + 1, 1)).astype(float)
        self.signal_df = signal_df
        return signal_df