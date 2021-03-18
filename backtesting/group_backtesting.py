import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from typing import Any, Dict, Union

from backtesting.base import BaseBacktesting
from backtesting.period_backtesting import LongShortPeriodBacktesting
from backtesting.monthly_backtesting import LongShortMonthlyBacktesting

class GroupBacktesting(BaseBacktesting):
    """
    分组回测系统

    分组回测系统会根据因子值将期货品种等分成若干组，做多每一组的品种，得到每一组的回测结果

    Attributes
    __________
    group_df: DataFrame
                根据因子值大小得到的分组结果，index为交易时间，columns为品种代码，分组为1，2，3，4，5，0表示不属于任何一组（不可交易）
    weight_df_dict: Dict[str, DataFrame]
                    分组权重DataFrame的字典。为方便进行分组回测，将每组的权重DataFrame分成若干DataFrame，
                    每个DataFrame中仅有租内的品种权重不为0，其他品种均为0。
    backtesting: Union[LongShortPeriodBacktesting, LongShortPeriodBacktesting]
                    作为执行回测的回测引擎
    backtesting_result_dict: Dict[str, Any]
                                每组回测的结果字典
    profit_series_dict: Dict[str, Series]
                        每组回测的日收益时间序列
    return_series_dict: Dict[str, Series]
                        每组回测的日收益率序列
    cum_profit_series_dict: Dict[str, Series]
                            每组回测的累积收益序列
    cum_return_series_dict: Dict[str, Series]
                            每组回测的累积收益率序列

    See Also
    ________
    backtesting.monthly_backtesting.LongShortMonthlyBacktesting
    backtesting.period_backtesting.LongShortPeriodBacktesting
    """

    def __init__(self,
                 rate: float = 0,
                 period: Union[str,int] = 'end',
                 interest: str = 'simple',
                 contract: str = 'main',
                 price: str = 'close',
                 rebalance_num: int = 1,
                 group_num: int = 5,
                 **kwargs) -> None:
        """Constructor"""
        super().__init__(rate=rate,
                         period=period,
                         interest=interest,
                         contract=contract,
                         price=price,
                         rebalance_num=rebalance_num,
                         group_num=group_num,
                         **kwargs)

        self.group_df: DataFrame = None  # 分组的DataFrame,index为交易时间,columns为品种代码，分组为1，2，3，4，5.0表示不属于任何一组
        # 分组权重DataFrame的字典，key为组label，如1，2，3，4，5.value为df
        # index为交易时间，columns为品种代码，data为权重
        self.weight_df_dict: Dict[str, DataFrame] = None

        self.backtesting: Union[LongShortPeriodBacktesting, LongShortPeriodBacktesting] = None
        self.backtest_result_dict: Dict[str, Any] = {}

        self.profit_series_dict: Dict[str, Series] = {}
        self.cum_profit_series_dict: Dict[str, Series] = {}

    def run_backtesting(self) -> None:
        """
        运行分组回测
        :param rate: 交易费用
        :param period: monthly或者整数
        :return: None
        """
        # 检查是否有weight
        self.prepare_weights()
        if not isinstance(self.weights, dict):
            raise ValueError("Init weight first!")

        params = self.get_params()
        rate = params['rate']
        period = params['period']
        interest = params['interest']
        contract = params['contract']
        price = params['price']
        rebalance_num = params['rebalance_num']
        group_num = params['group_num']

        # 如果是月末调仓
        if isinstance(period, str):
            backtesting = LongShortMonthlyBacktesting(rate=rate,
                                                      period=period,
                                                      interest=interest,
                                                      contract=contract,
                                                      price=price,
                                                      rebalance_num=rebalance_num,
                                                      )

        # 如果是固定天数调仓
        elif isinstance(period, int):
            backtesting = LongShortPeriodBacktesting(rate=rate,
                                                    period=period,
                                                    interest=interest,
                                                    contract=contract,
                                                     price=price,
                                                    rebalance_num=rebalance_num)

        else:
            raise TypeError("period must be an integer or a string")

        for i in self.weights:
            weight_df = self.weights[i]
            backtesting.set_weight_df(weight_df)
            backtesting.run_backtesting()

            self.profit_series_dict[i] = backtesting.profit_series
            self.cum_profit_series_dict[i] = backtesting.cum_profit_series
            self.backtest_result_dict[i] = backtesting.backtest_result['metrics']

            self.backtesting = backtesting

    def output_backtest_result(self, overwrite: bool = True) -> None:
        """
        输出回测结果

        Parameters
        ----------
        overwrite: bool, default True
                    是否覆盖已有的

        Returns
        -------
        None
        """
        # 因子信息
        factor_info = self.factor_info
        factor_group, factor_name = factor_info['group'], factor_info['name']
        factor_folder_path = self.backtest_result_path.joinpath(factor_group).joinpath(factor_name)
        if not os.path.exists(factor_folder_path):
            os.makedirs(factor_folder_path)

        # 商品池信息
        commodity_pool_info = self.commodity_pool_info
        commodity_pool_group, commodity_pool_name = commodity_pool_info['group'], commodity_pool_info['name']

        # 信号信息
        signal_info = self.signal_info
        signal_group, signal_name = signal_info['group'], signal_info['name']

        # 权重名称
        weight_info = self.weight_info
        weight_group, weight_name = weight_info['group'], weight_info['name']

        # 回测参数
        backtest_params = self.get_params()

        info_dict = {}
        info_dict['factor_group'] = factor_group
        info_dict['factor_name'] = factor_name
        info_dict['factor_params'] = self.factor_params
        info_dict['commodity_pool_group'] = commodity_pool_group
        info_dict['commodity_pool_name'] = commodity_pool_name
        info_dict['commodity_pool_params'] = self.commodity_pool_params
        info_dict['signal_group'] = signal_group
        info_dict['signal_name'] = signal_name
        info_dict['signal_params'] = self.signal_params
        info_dict['weight_group'] = weight_group
        info_dict['weight_name'] = weight_name
        info_dict['weight_params'] = self.weight_params
        info_dict['backtest_params'] = backtest_params
        str_info_dict = str(info_dict)

        factor_group_analysis_folder_path = factor_folder_path.joinpath("group_analysis")
        if not os.path.exists(factor_group_analysis_folder_path):
            os.makedirs(factor_group_analysis_folder_path)

        settings = self.load_setting()
        if str_info_dict in settings:
            if not overwrite:
                return
            else:
                group_analysis_id = settings[str_info_dict]
        else:
            group_analysis_id = len(os.listdir(factor_group_analysis_folder_path)) +1
            settings[str_info_dict] = group_analysis_id
            self.save_setting(settings)

        single_factor_group_analysis_folder_path = factor_group_analysis_folder_path.joinpath(f"{group_analysis_id}")
        if not os.path.exists(single_factor_group_analysis_folder_path):
            os.makedirs(single_factor_group_analysis_folder_path)
        with open(single_factor_group_analysis_folder_path.joinpath("setting.json"), "w") as f:
            json_info_dict = json.dumps(info_dict)
            f.write(json_info_dict)

        # 指标保存
        symbol_result_list = []
        industry_result_list = []
        all_result_list = []
        for i in self.backtest_result_dict:
            backtest_result = self.backtest_result_dict[i]

            symbol_result = backtest_result['symbol']
            symbol_result = symbol_result.stack().to_frame("value")
            symbol_result['group'] = i
            symbol_result.index.names = ['underlying_symbol', 'metrics']
            symbol_result.reset_index(inplace=True)
            symbol_result_list.append(symbol_result)

            industry_result = backtest_result['industry']
            industry_result = industry_result.stack().to_frame("value")
            industry_result['group'] = i
            industry_result.index.names = ['underlying_symbol', 'metrics']
            industry_result.reset_index(inplace=True)
            industry_result_list.append(industry_result)

            all_result = backtest_result['all'].to_frame("value")
            all_result['group'] = i
            all_result.index.names = ['metrics']
            all_result.reset_index(inplace=True)
            all_result_list.append(all_result)

        symbol_result_df = pd.concat(symbol_result_list, axis=0)
        symbol_result_df = symbol_result_df.set_index(['underlying_symbol', 'metrics', 'group']).\
            unstack(level=[2, 1])
        symbol_result_df.columns = symbol_result_df.columns.droplevel(level=0)
        industry_result_df = pd.concat(industry_result_list, axis=0)
        industry_result_df = industry_result_df.set_index(['underlying_symbol', 'metrics', 'group']).\
            unstack(level=[2, 1])
        industry_result_df.columns = industry_result_df.columns.droplevel(level=0)
        all_result_df = pd.concat(all_result_list, axis=0)
        all_result_df = all_result_df.set_index(['metrics', 'group']).unstack(level=-1)
        all_result_df.columns = all_result_df.columns.droplevel(level=0)
        all_result_df.columns.names = [None]
        all_result_df.index.names = [None]

        all_result_df.to_csv(single_factor_group_analysis_folder_path.joinpath("all.csv"))
        industry_result_df.to_csv(single_factor_group_analysis_folder_path.joinpath("industry.csv"))
        symbol_result_df.to_csv(single_factor_group_analysis_folder_path.joinpath("symbol.csv"))

        title = ''
        # 添加因子
        title += f"{self.factor.get_string()}\n"
        # 添加商品池
        title += f"{self.commodity_pool.get_string()}\n"
        # 添加信号
        title += f"{self.signal.get_string()}\n"
        # 添加权重
        title += f"{self.weight.get_string()}\n"
        # 添加回测
        title += f"{self.get_string()}"

        init_total_value = 100000000
        cum_return_df = pd.DataFrame(self.cum_profit_series_dict)/ init_total_value
        plt.figure(figsize=(20, 8))
        cum_return_df.plot(figsize=(20, 8))
        plt.title(title, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center'})
        plt.grid()
        plt.savefig(single_factor_group_analysis_folder_path.joinpath("curve.png"))
        plt.show()

    def save_setting(self, settings):
        setting_file_path = self.backtest_result_path.joinpath("group_analysis_setting.json")

        with open(setting_file_path, "w") as f:
            json_settings = json.dumps(settings)
            f.write(json_settings)

    def load_setting(self):
        setting_file_path = self.backtest_result_path.joinpath("group_analysis_setting.json")
        if not os.path.exists(setting_file_path):
            with open(setting_file_path, "w") as f:
                json_settings = json.dumps({})
                f.write(json_settings)
            return {}
        else:
            with open(setting_file_path, "rb") as f:
                settings = json.load(f)
        return settings



