
import numpy as np
import pandas as pd
from tqdm import tqdm
from pandas import DataFrame
from datetime import datetime
from typing import List

def get_hold_profit_df(pos_df: DataFrame, price_df: DataFrame, hold_datetime_list: List[datetime]) -> DataFrame:
    """
    根据每日持仓、每日收盘价和换仓日期列表获取每次持仓的收益（时间戳为持仓开始日期）

    Parameters
    ----------
    pos_df: DataFrame
            每日持仓，index为交易日期，columns为品种代码，data为每日持仓（对应开盘调整后的持仓）

    price_df: DataFrame
            每日价格，index为交易日期，columns为品种代码，data为价格

    hold_datetime_list: List[datetime]
                        换仓日期列表

    Returns
    -------
    hold_profit_df: DataFrame
                    每次持仓数据
    """
    hold_profit_list = []
    for i in range(len(hold_datetime_list)):
        if i != len(hold_datetime_list)-1:
            start_date = hold_datetime_list[i]
            end_date = hold_datetime_list[i+1]
            pos = pos_df.loc[start_date]
            start_price = price_df.loc[start_date]
            end_price = price_df.loc[end_date]
            hold_profit = pos * (end_price - start_price)
            hold_profit_list.append(hold_profit)
        else:
            start_date = hold_datetime_list[i]
            pos = pos_df.loc[start_date]
            start_price = price_df.loc[start_date]
            end_price = price_df.iloc[-1]
            hold_profit = pos * (end_price - start_price)
            hold_profit_list.append(hold_profit)
    hold_profit_df = pd.concat(hold_profit_list, axis=1).T
    hold_profit_df.index = hold_datetime_list
    return hold_profit_df

def execute_simple_time_series_backtesting(weight_df: DataFrame,
                                           price_df: DataFrame,
                                           init_total_value: int,
                                           hold_datetime_list: List[datetime],
                                           rate: float, tqdm_flag: bool = True):
    """
    时间序列单利回测执行
    通过输入每日持仓权重、每日收盘价、初始资金量、换仓日期列表、交易费用执行单利回测

    Parameters
    ----------
    weight_df: DataFrame
                每日持仓权重

    price_df: DataFrame
                每日价格

    init_total_value: int
                    初始持仓金额
    hold_datetime_list: List[datetime]
                    换仓日期列表
    rate: float
            交易费用

    tqdm_flag: bool, default True
            是否使用tqdm
    """
    flag = 0
    pos_list = []  # 每日持仓数量（换仓日为换仓之后的）
    value_list = []  # 每日持仓价值（换仓日为换仓之后的）
    weight_list = []
    turnover_list = []  # 每日成交额数量（除了换仓日都是0）
    profit_list = []  # 每日收益（换仓日为换仓之前的仓位的收益）

    if tqdm_flag:
        iters = tqdm(range(len(weight_df)))
    else:
        iters = range(len(weight_df))

    for i in iters:
        date = weight_df.index[i]
        # 第一天
        if i == 0:
            price = price_df.iloc[i]
            weight = weight_df.iloc[i]
            value = weight * init_total_value
            pos = value / price
            profit = pd.Series(data=0.0, index=price.index)
            turnover = value.copy()

            pos_list.append(pos)
            value_list.append(value)
            profit_list.append(profit)
            weight_list.append(weight)
            turnover_list.append(turnover)

        # 不是第一天
        else:
            # 如果是换仓日
            if date in hold_datetime_list:
                # 换仓前对前一组进行统计
                price = price_df.iloc[i]
                last_price = price_df.iloc[i - 1]
                profit = pos * (price - last_price)
                value = value + np.sign(value) * profit
                profit_list.append(profit)

                # 更新仓位
                weight = weight_df.iloc[i]
                price = price_df.iloc[i]
                new_value = weight * init_total_value
                turnover = new_value - value
                value = new_value.copy()
                pos = value / price
                pos_list.append(pos)
                value_list.append(value)
                profit_list.append(profit)
                weight_list.append(weight)
                turnover_list.append(turnover)
            # 如果不是换仓日
            else:
                price = price_df.iloc[i]
                last_price = price_df.iloc[i-1]
                profit = pos * (price - last_price)
                value = value + np.sign(value)*profit
                turnover = pd.Series(data=0.0, index=price.index)
                pos_list.append(pos)
                value_list.append(value)
                weight_list.append(weight)
                profit_list.append(profit)
                turnover_list.append(turnover)

    pos_df = pd.concat(pos_list, axis=1).T
    pos_df.index = price_df.index
    pos_df.columns = price_df.columns
    pos_df.index.names = ['datetime']
    pos_df.columns.names = ['underlying_symbol']

    value_df = pd.concat(value_list, axis=1).T
    value_df.index = price_df.index
    value_df.columns = price_df.columns
    value_df.index.names = ['datetime']
    value_df.columns.names = ['underlying_symbol']

    turnover_df = pd.concat(turnover_list, axis=1).T
    turnover_df.index = price_df.index
    turnover_df.columns = price_df.columns
    turnover_df.index.names = ['datetime']
    turnover_df.columns.names = ['underlying_symbol']
    turnover_df = np.abs(turnover_df)

    profit_df = pd.concat(profit_list, axis=1).T
    profit_df.index = price_df.index
    profit_df.columns = price_df.columns
    profit_df.index.names = ['datetime']
    profit_df.columns.names = ['underlying_symbol']

    profit_df = profit_df - turnover_df * rate

    weight_df = pd.concat(weight_list, axis=1).T
    weight_df.index = price_df.index
    weight_df.columns = price_df.columns
    weight_df.index.names = ['datetime']
    weight_df.columns.names = ['underlying_symbol']

    hold_pos_df = pos_df.loc[hold_datetime_list]
    hold_weight_df = weight_df.loc[hold_datetime_list]
    hold_turnover_df = turnover_df.loc[hold_datetime_list]
    hold_value_df = value_df.loc[hold_datetime_list]
    hold_profit_df = get_hold_profit_df(pos_df, price_df, hold_datetime_list)
    hold_profit_df.index.names = ['datetime']

    return value_df, hold_value_df, weight_df, hold_weight_df, turnover_df, hold_turnover_df, profit_df, hold_profit_df


def execute_compound_time_series_backtesting(weight_df: DataFrame,
                                           price_df: DataFrame,
                                           init_total_value: int,
                                           hold_datetime_list: List[datetime],
                                           rate: float, tqdm_flag: bool = True):
    """
    时间序列复利回测执行
    通过输入每日持仓权重、每日收盘价、初始资金量、换仓日期列表、交易费用执行复利回测

    Parameters
    ----------
    weight_df: DataFrame
                每日持仓权重

    price_df: DataFrame
                每日价格

    init_total_value: int
                    初始持仓金额
    hold_datetime_list: List[datetime]
                    换仓日期列表
    rate: float
            交易费用

    tqdm_flag: bool, default True
            是否使用tqdm
    """
    flag = 0
    pos_list = []  # 每日持仓数量（换仓日为换仓之后的）
    value_list = []  # 每日持仓价值（换仓日为换仓之后的）
    weight_list = []
    turnover_list = []  # 每日成交额数量（除了换仓日都是0）
    profit_list = []  # 每日收益（换仓日为换仓之前的仓位的收益）

    if tqdm_flag:
        iters = tqdm(range(len(weight_df)))
    else:
        iters = range(len(weight_df))

    for i in iters:
        date = weight_df.index[i]
        # 第一天
        if i == 0:
            price = price_df.iloc[i]
            weight = weight_df.iloc[i]
            value = weight * init_total_value
            pos = value / price
            profit = pd.Series(data=0.0, index=price.index)
            turnover = value.copy()

            pos_list.append(pos)
            value_list.append(value)
            profit_list.append(profit)
            weight_list.append(weight)
            turnover_list.append(turnover)

        # 不是第一天
        else:
            # 如果是换仓日
            if date in hold_datetime_list:
                # 换仓前对前一组进行统计
                price = price_df.iloc[i]
                last_price = price_df.iloc[i - 1]
                profit = pos * (price - last_price)
                value = value + np.sign(value) * profit
                profit_list.append(profit)

                # 更新仓位
                weight = weight_df.iloc[i]
                price = price_df.iloc[i]
                new_value = weight * value
                turnover = new_value - value
                value = new_value.copy()
                pos = value / price
                pos_list.append(pos)
                value_list.append(value)
                profit_list.append(profit)
                weight_list.append(weight)
                turnover_list.append(turnover)
            # 如果不是换仓日
            else:
                price = price_df.iloc[i]
                last_price = price_df.iloc[i - 1]
                profit = pos * (price - last_price)
                value = value + np.sign(value) * profit
                turnover = pd.Series(data=0.0, index=price.index)
                pos_list.append(pos)
                value_list.append(value)
                weight_list.append(weight)
                profit_list.append(profit)
                turnover_list.append(turnover)

    pos_df = pd.concat(pos_list, axis=1).T
    pos_df.index = price_df.index
    pos_df.columns = price_df.columns
    pos_df.index.names = ['datetime']
    pos_df.columns.names = ['underlying_symbol']

    value_df = pd.concat(value_list, axis=1).T
    value_df.index = price_df.index
    value_df.columns = price_df.columns
    value_df.index.names = ['datetime']
    value_df.columns.names = ['underlying_symbol']

    turnover_df = pd.concat(turnover_list, axis=1).T
    turnover_df.index = price_df.index
    turnover_df.columns = price_df.columns
    turnover_df.index.names = ['datetime']
    turnover_df.columns.names = ['underlying_symbol']
    turnover_df = np.abs(turnover_df)

    profit_df = pd.concat(profit_list, axis=1).T
    profit_df.index = price_df.index
    profit_df.columns = price_df.columns
    profit_df.index.names = ['datetime']
    profit_df.columns.names = ['underlying_symbol']

    profit_df = profit_df - turnover_df * rate

    weight_df = pd.concat(weight_list, axis=1).T
    weight_df.index = price_df.index
    weight_df.columns = price_df.columns
    weight_df.index.names = ['datetime']
    weight_df.columns.names = ['underlying_symbol']

    hold_pos_df = pos_df.loc[hold_datetime_list]
    hold_weight_df = weight_df.loc[hold_datetime_list]
    hold_turnover_df = turnover_df.loc[hold_datetime_list]
    hold_value_df = value_df.loc[hold_datetime_list]
    hold_profit_df = get_hold_profit_df(pos_df, price_df, hold_datetime_list)
    hold_profit_df.index.names = ['datetime']

    return value_df, hold_value_df, weight_df, hold_weight_df, turnover_df, hold_turnover_df, profit_df, hold_profit_df




