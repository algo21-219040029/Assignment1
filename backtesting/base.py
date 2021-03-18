import os
import json
import importlib
from pathlib import Path
from pandas import DataFrame,\
                    Series
from abc import abstractmethod
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any

from bases.base import BaseClass
from data_manager.BasicsDataManager import BasicsDataManager
from data_manager.FactorDataManager import FactorDataManager
from data_manager.CommodityPoolManager import CommodityPoolManager
from data_manager.ContinuousContractDataManager import ContinuousContractDataManager

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

class BaseBacktesting(BaseClass):
    """
    回测系统的基类

    每一个回测系统均需要继承自BaseBacktesting，需要实现run_backtesting

    Attributes
    __________
    weights: DataFrame
            权重DataFrame, index为交易时间, columns为品种代码, data为权重, 做多为正, 做空为负, 空仓为0

    all_instruments: DataFrame
                    所有期货合约基础信息

    symbol_industry_map: DataFrame
                        各期货品种和行业对应表, 两列数据, 第一列为品种代码, 第二列为行业

    basics_data_manager: 基础数据管理器

    factor_data_manager: 因子管理器

    commodity_pool_manager: 商品池管理器

    continuous_contract_data_manager: 连续合约数据管理器

    backtest_result_path: 回测结果路径
    """
    def __init__(self, **params) -> None:
        """Constructor"""
        super().__init__(**params)

        self.weights: DataFrame = None

        self.all_instruments: DataFrame = None
        self.symbol_industry_map: DataFrame = None

        self.basics_data_manager: BasicsDataManager = BasicsDataManager()
        self.factor_data_manager: FactorDataManager = FactorDataManager()
        self.commodity_pool_manager: CommodityPoolManager = CommodityPoolManager()
        self.continuous_contract_data_manager: ContinuousContractDataManager = ContinuousContractDataManager()

        self.backtest_result_path: Path = Path(__file__).parent.parent.joinpath("output_result")

        if not os.path.exists(self.backtest_result_path):
            os.makedirs(self.backtest_result_path)

        self.init_basics_data()

        self.backtest_result: Dict[str, Any] = {}
        self.cum_profit_series: Series = None


    def init_basics_data(self) -> None:
        """
        初始化基础数据
        """
        if not isinstance(self.all_instruments, DataFrame):
            all_instruments = self.basics_data_manager.get_all_instruments()
            symbol_industry_map = all_instruments[['underlying_symbol', 'industry_name']].drop_duplicates()
            self.all_instruments = all_instruments
            self.symbol_industry_map = symbol_industry_map

    def get_continuous_field(self, contract: str, price: str = 'close', rebalance_num: int = 1, field: str = 'continuous_price') -> DataFrame:
        """
        获取连续合约指定字段的数据

        Parameters
        ----------
        contract: str
                合约种类，目前可选有main和active_near, main表示主力合约, active_near表示活跃近月

        price: str
                连续合约价格, close or settlement

        rebalance_num: int, default = 1
                换仓天数, 可选天数1,3,5
        field: str, default = 'continuous_price'
                字典，continuous_price

        Returns
        -------
        df: DataFrame
            连续合约field字段数据, 一般是开盘价或收盘价
        """
        return self.continuous_contract_data_manager.get_field(contract=contract,
                                                               price=price,
                                                               rebalance_num=rebalance_num,
                                                               field=field)

    def get_file_name(self, params):
        string = ''
        for param in params:
            string += f"{param}_{params[param]}"
            string += ' '
        if string:
            string = string[:-1]
        return string

    def import_component_class(self, component: str, group: str, name: str) -> Any:
        """
        根据component类别, group和name导入类(信号类和权重类)

        Parameters
        ----------
        component: str
                    回测的部件, signals or weight

        group: str
                所属类别

        name: 名称

        Returns
        -------
        component_class: Any
                        部件类，一般是信号类或权重类
        """
        component_file_path = f"{component}.{group}.{name}"
        component_module = importlib.import_module(component_file_path)
        component_class = getattr(component_module, name)
        return component_class

    def set_commodity_pool(self, group: str, name: str, **params) -> None:
        """
        根据group, name, str设置商品池

        Parameters
        ----------
        group: str
                商品池类别

        name: str
                商品池名称

        params: 商品池参数

        Returns
        -------
        None
        """
        self.commodity_pool_info = {'group': group, 'name': name}
        self.commodity_pool = self.commodity_pool_manager.get_commodity_pool(group=group, name=name, **params)
        self.commodity_pool_params = self.commodity_pool.get_params()

    def set_factor(self, group: str, name: str, **params: Dict[str, Any]) -> None:
        """
        设置因子

        Parameters
        ----------
        group: str
                因子类别

        name: str
                因子名称

        params: str
                因子参数

        Returns
        -------
        None
        """
        self.factor_info = {'group': group, 'name': name}
        self.factor = self.factor_data_manager.get_factor(group=group, name=name, **params)
        self.factor_params = self.factor.get_params()

    def set_signal(self, group: str, name: str, **params: Dict[str, Any]) -> None:
        """
        设置信号

        Parameters
        ----------
        group: str
                信号类别

        params: Dict[str, Any]
                信号参数

        Returns
        -------
        None
        """
        self.signal_info = {'group': group, 'name': name}
        self.signal= self.import_component_class(component='signals', group=group, name=name)(**params)
        self.signal_params = self.signal.get_params()

    def set_weight(self, group: str, name: str, **params: Dict[str, Any]) -> None:
        """
        设置权重

        Parameters
        ----------
        group: str
                权重类别

        name: str
                权重名称

        params: str
                权重参数

        Returns
        -------
        None
        """
        self.weight_info = {'group': group, 'name': name}
        self.weight = self.import_component_class(component='weight', group=group, name=name)(**params)
        self.weight_params = self.weight.get_params()

    def set_weight_df(self, weight_df: DataFrame) -> None:
        """
        导入权重DataFrame

        Parameters
        ----------
        weight_df: DataFrame
                    权重DataFrame

        Returns
        -------
        None
        """
        self.weights = weight_df

    def prepare_weights(self) -> None:
        """
        回测准备工作，生成回测所需要的权重

        Returns
        -------
        None
        """
        self.signal.set_commodity_pool(self.commodity_pool.commodity_pool_value)
        self.signal.set_factor_data(self.factor.factor_value)
        signal_df = self.signal.transform()
        self.weight.set_signal(signal_df)
        weights = self.weight.get_weight()
        self.weights = weights

    @abstractmethod
    def run_backtesting(self, *args, **kwargs) -> Tuple[Dict]:
        """
        执行回测，每个回测类需要具体实现的地方
        """
        raise NotImplementedError

    def save_setting(self, settings):
        """
        保存回测信息

        参数字典字符串: backtest_id

        Parameters
        ----------
        settings: 参数字典

        Returns
        -------
        None
        """
        setting_file_path = self.backtest_result_path.joinpath("backtest_setting.json")

        with open(setting_file_path, "w") as f:
            json_settings = json.dumps(settings)
            f.write(json_settings)

    def load_setting(self) -> Dict[str, Any]:
        """
        加载回测信息

        Returns
        -------
        None
        """
        setting_file_path = self.backtest_result_path.joinpath("backtest_setting.json")
        if not os.path.exists(setting_file_path):
            with open(setting_file_path, "w") as f:
                json_settings = json.dumps({})
                f.write(json_settings)
            return {}
        else:
            with open(setting_file_path, "rb") as f:
                settings = json.load(f)
        return settings

    def get_info(self) -> Dict[str, Any]:
        # 因子信息
        factor_info = self.factor_info
        factor_group, factor_name = factor_info['group'], factor_info['name']
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
        return info_dict

    def get_metrics(self) -> Tuple[Any, Any, Any]:
        all_metrics = self.backtest_result['metrics']['all']
        industry_metrics = self.backtest_result['metrics']['industry']
        symbol_metrics = self.backtest_result['metrics']['symbol']
        return all_metrics, industry_metrics, symbol_metrics

    def plot(self) -> None:
        backtest_result = self.backtest_result
        metrics_result = backtest_result['metrics']
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
        title += f"{self.get_string()}         "
        # 添加指标
        title += f"sharpe={round(metrics_result['all']['sharpe'],2)} turnover rate={round(metrics_result['all']['turnover_rate'],2)} annual return={round(metrics_result['all']['annual_return'],2)}"

        # 保存回测曲线结果
        plt.figure()
        init_total_value = 100000000
        (self.cum_profit_series / init_total_value).plot(figsize=(20, 8))
        # plt.title(title, fontdict={'horizontalalignment': 'center', 'verticalalignment':'left'}, bbox=dict(ec='black', fc='w'))
        plt.title(title, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center'})
        plt.grid()
        plt.show()

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

        # 初始化单因子回测路径
        factor_backtest_folder_path = factor_folder_path.joinpath("backtest_result")
        if not os.path.exists(factor_backtest_folder_path):
            os.makedirs(factor_backtest_folder_path)

        # 添加settings文件的回测信息
        settings = self.load_setting()
        if str_info_dict in settings:
            if not overwrite:
                return
            else:
                backtest_id = settings[str_info_dict]
        else:
            backtest_id = len(os.listdir(factor_backtest_folder_path))+1
            settings[str_info_dict] = backtest_id
            self.save_setting(settings)

        if not os.path.exists(factor_backtest_folder_path.joinpath(f"{backtest_id}")):
            os.makedirs(factor_backtest_folder_path.joinpath(f"{backtest_id}"))

        backtest_result = self.backtest_result
        metrics_result = backtest_result['metrics']
        for i in metrics_result:
            metrics_result[i].to_csv(str(factor_backtest_folder_path.joinpath(f"{backtest_id}").joinpath(f"{i}.csv")))
        (self.cum_profit_series/100000000).to_frame("cum_profit").to_csv(str(factor_backtest_folder_path.joinpath(f"{backtest_id}").joinpath("cum_profit.csv")))
        with open(str(factor_backtest_folder_path.joinpath(f"{backtest_id}").joinpath("setting.json")), "w") as f:
            json_info_dict = json.dumps(info_dict)
            f.write(json_info_dict)

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
        title += f"{self.get_string()}         "
        # 添加指标
        title += f"sharpe={round(metrics_result['all']['sharpe'],2)} turnover rate={round(metrics_result['all']['turnover_rate'],2)} annual return={round(metrics_result['all']['annual_return'],2)}"

        # 保存回测曲线结果
        plt.figure()
        init_total_value = 100000000
        (self.cum_profit_series / init_total_value).plot(figsize=(20, 8))
        # plt.title(title, fontdict={'horizontalalignment': 'center', 'verticalalignment':'left'}, bbox=dict(ec='black', fc='w'))
        plt.title(title, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center'})
        plt.grid()
        plt.savefig(factor_backtest_folder_path.joinpath(f"{backtest_id}").joinpath("curve.png"))
        plt.show()

    def __repr__(self) -> str:
        string = 'backtesting('
        for key, value in self.get_params().items():
            string += f"{key}={value}, "
        string = string[:-2]
        string += ')'
        return string

    def get_string(self) -> str:
        return self.__repr__()












