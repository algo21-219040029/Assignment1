import os
import importlib
import pandas as pd
from pathlib import Path
from pandas import Series, \
                    DataFrame
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

from commodity_pool.base import BaseCommodityPool
from collections import defaultdict


class CommodityPoolManager:
    """
    商品池处理器

    1.检查是否存在某个商品池
    2.获取商品池数据
    3.保存商品池数据
    4.保存商品池信息

    Attributes
    __________
    commodity_pool_file_folder_path: pathlib.Path
                                    商品池代码文件路径

    commodity_pool_data_folder_path: pathlib.Path
                                    商品池数据文件夹路径

    commodity_pool_dict: Dict[str, Dict[str, BaseCommodityPool]]
                        商品池字典，双层字典，外层为商品池名称: 该商品池的不同参数的具体商品池的字典，内层为商品池参数: 商品池DataFrame

    Notes
    _____
    CommodityPoolManager不涉及商品池构建的代码，商品池构建的代码详见commodity_pool.CommodityPool

    See Also
    ________
    commodity_pool.CommodityPool
    """

    def __init__(self) -> None:
        """Constructor"""
        self.commodity_pool_file_folder_path: Path = Path(__file__).parent.parent.joinpath("commodity_pool")
        self.commodity_pool_data_folder_path: Path = Path(__file__).parent.parent.joinpath("data").joinpath("commodity_pool")

        self.commodity_pool_dict: Dict[str, Dict[str, BaseCommodityPool]] = defaultdict(dict)

    def get_file_name(self, params: Dict[str, Any]) -> str:
        """
        将参数字典转化为文件名，即参数1_参数值1 参数2_参数值2 参数3_参数值3 xxx

        Parameters
        ----------
        params: Dict[str, Any]
                参数字典

        Returns
        -------
        string: str
                参数1_参数值1 参数2_参数值2 参数3_参数值3 xxx
        """
        string = ''
        for param in params:
            string += f"{param}_{params[param]}"
            string += ' '
        if string:
            string = string[:-1]
        return string

    def import_commodity_pool_class(self, group: str, name: str) -> BaseCommodityPool:
        """
        根据group和name导入商品池类

        Parameters
        ----------
        group: str
                商品池类别，即FixedPool或者DynamicPool

        name: str
                商品池名称

        Returns
        -------
        commodity_pool_class: 商品池类
        """
        commodity_pool_file_path = "commodity_pool"+"."+group+"."+name
        commodity_pool_module = importlib.import_module(commodity_pool_file_path)
        commodity_pool_class = getattr(commodity_pool_module, name)
        return commodity_pool_class

    def get_commodity_pool_in_out(self, group: str, name: str, **params) -> Dict[str, Any]:
        """
        获取商品池中品种每日进出及每个品种的进出次数
        Parameters
        ----------
        group
        name
        params

        Returns
        -------

        """
        commodity_pool_instance = self.get_commodity_pool(group=group, name=name, **params)
        commodity_pool_value: DataFrame = commodity_pool_instance.get_commodity_pool_value()
        commodity_pool_value.fillna(False, inplace=True)
        commodity_pool_value = commodity_pool_value.astype(int)
        commodity_pool_value_diff = commodity_pool_value.diff(1)
        commodity_pool_value_diff.iloc[0] = commodity_pool_value.iloc[0]
        commodity_pool_value_diff = commodity_pool_value_diff.astype(int)
        # 获取每日进入情况
        commodity_daily_in = {}
        commodity_daily_out = {}
        for date in commodity_pool_value_diff.index:
            commodity_pool_diff_series = commodity_pool_value_diff.loc[date]
            in_list = commodity_pool_diff_series[commodity_pool_diff_series == 1].index.tolist()
            out_list = commodity_pool_diff_series[commodity_pool_diff_series == -1].index.tolist()
            commodity_daily_in[date] = in_list
            commodity_daily_out[date] = out_list

        # 获取每个品种进入商品池几次
        commodity_in_num = {}
        commodity_out_num = {}
        for symbol in commodity_pool_value_diff.columns:
            value_count = commodity_pool_value_diff[symbol].value_counts().to_dict()
            if -1 not in value_count.keys():
                value_count[-1] = 0
            if 1 not in value_count.keys():
                value_count[1] = 0
            value_count.pop(0)
            commodity_in_num[symbol] = value_count[1]
            commodity_out_num[symbol] = value_count[-1]

        result = {'daily_in': commodity_daily_in,
                  'daily_out': commodity_daily_out,
                  'symbol_in_num': commodity_in_num,
                  'symbol_out_num': commodity_out_num}
        return result

    def get_commodity_pool_time_series_plot(self, group: str, name: str, **params) -> Series:
        """
        获取商品池中的品种数目变化图

        Parameters
        ----------
        group: str
                商品池类别，即FixedPool或者DynamicPool

        name: str
                商品池名称

        params: 可变参数
                商品池参数

        Returns
        -------
        None
        """
        commodity_pool_instance = self.get_commodity_pool(group=group, name=name, **params)
        commodity_pool_value: DataFrame = commodity_pool_instance.get_commodity_pool_value()
        commodity_pool_value.fillna(False, inplace=True)
        commodity_pool_num_series:Series = commodity_pool_value.sum(axis=1)
        plt.figure(figsize=(20, 8))
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.step(commodity_pool_num_series.index, commodity_pool_num_series.values)
        # for i, j in zip(commodity_pool_num_series.index, commodity_pool_num_series.values):
        #     ax.text(x=i, y=j+1, s=str(int(j)))
        fig.tight_layout()
        plt.grid()
        plt.title(commodity_pool_instance.__repr__())
        plt.show()
        return commodity_pool_num_series

    def get_commodity_pool(self, group: str, name: str, **params) -> BaseCommodityPool:
        """
        获取商品池

        Parameters
        ----------
        group: str
                商品池类别，即FixedPool或者DynamicPool

        name: str
                商品池名称

        params: 可变参数
                商品池参数

        Returns
        -------
        commodity_pool_instance: BaseCommodityPool
                                商品池实例，可获取商品池参数，商品池数值DataFrame
        """
        # 参数检测和生成文件名
        commodity_pool_class = self.import_commodity_pool_class(group=group, name=name)
        commodity_pool_instance = commodity_pool_class(**params)
        string = self.get_file_name(commodity_pool_instance.get_params())

        if f"{group}_{name}" in self.commodity_pool_dict:
            if string in self.commodity_pool_dict:
                return self.commodity_pool_dict[f"{group}_{name}"][string]
            else:
                commodity_pool_class = self.import_commodity_pool_class(group, name)
                commodity_pool_instance = commodity_pool_class(**params)

                try:
                    commodity_pool_value = pd.read_pickle(self.commodity_pool_data_folder_path.joinpath(group).
                                                    joinpath(name).joinpath(f"{string}.pkl"))
                    commodity_pool_instance.set_commodity_pool_value(commodity_pool_value)
                except:
                    commodity_pool_value = commodity_pool_instance.compute_commodity_pool()

                self.commodity_pool_dict[f"{group}_{name}"][string] = commodity_pool_instance
                return commodity_pool_instance

        else:
            commodity_pool_class = self.import_commodity_pool_class(group, name)
            commodity_pool_instance = commodity_pool_class(**params)

            try:
                commodity_pool_value = pd.read_pickle(self.commodity_pool_data_folder_path.joinpath(group).
                                                      joinpath(name).joinpath(f"{string}.pkl"))
                commodity_pool_instance.set_commodity_pool_value(commodity_pool_value)
            except:
                commodity_pool_value = commodity_pool_instance.compute_commodity_pool()

            self.commodity_pool_dict[f"{group}_{name}"][string] = commodity_pool_instance
            return commodity_pool_instance

    def compute_commodity_pool(self, group: str, name: str, **params) -> BaseCommodityPool:
        """
        计算商品池数据DataFrame

        Parameters
        ----------
        group: str
                商品池类别, FixedPool或者DynamicPool

        name: str
                商品池名称

        params: Dict[str, Any]
                商品池参数

        Returns
        -------
        commodity_pool_instance: BaseCommodityPool
                                商品池实例，可获取商品池参数，商品池数值DataFrame
        """
        commodity_pool_class = self.import_commodity_pool_class(group=group, name=name)
        commodity_pool_instance = commodity_pool_class(**params)
        string = self.get_file_name(commodity_pool_instance.get_params())

        if f"{group}_{name}" in self.commodity_pool_dict:
            if string in self.commodity_pool_dict[f"{group}_{name}"]:
                return self.commodity_pool_dict[f"{group}_{name}"][string]
            else:
                commodity_pool_class = self.import_commodity_pool_class(group, name)
                commodity_pool_instance = commodity_pool_class(**params)
                commodity_pool_instance.compute_commodity_pool()
                return commodity_pool_instance

        else:
            commodity_pool_class = self.import_commodity_pool_class(group, name)
            commodity_pool_instance = commodity_pool_class(**params)
            commodity_pool_instance.compute_commodity_pool()
            return commodity_pool_instance

    def save_commodity_pool(self, group: str, name: str, overwrite: bool = False, **params) -> None:
        """
        保存商品池数据

        Parameters
        ----------
        group: str
                商品池类别, FixedPool或者DynamicPool

        name: str
                商品池名称

        overwrite: bool, default False
                    是否覆盖本地已有的商品池数据

        params: Dict[str, Any]
                商品池参数

        Returns
        -------
        None
        """
        commodity_pool_class = self.import_commodity_pool_class(group=group, name=name)
        commodity_pool_instance = commodity_pool_class(**params)
        string = self.get_file_name(commodity_pool_instance.get_params())

        commodity_pool_data_folder_path = self.commodity_pool_data_folder_path.joinpath(group).joinpath(name)

        if not os.path.exists(commodity_pool_data_folder_path):
            os.makedirs(commodity_pool_data_folder_path)

        if not overwrite and os.path.exists(commodity_pool_data_folder_path.joinpath(f"{string}.pkl")):
            return

        if f"{group}_{name}" in self.commodity_pool_dict:
            if string in self.commodity_pool_dict[f"{name}_{group}"]:
                commodity_pool_value = self.commodity_pool_dict[f"{name}_{group}"][string].get_commodity_pool_value()
                commodity_pool_value.to_pickle(str(commodity_pool_data_folder_path.joinpath(f"{string}.pkl")))
            else:
                commodity_pool_instance = self.get_commodity_pool(group, name, **params)
                commodity_pool_value = commodity_pool_instance.get_commodity_pool_value()
                commodity_pool_value.to_pickle(str(commodity_pool_data_folder_path.joinpath(f"{string}.pkl")))
        else:
            commodity_pool_instance = self.get_commodity_pool(group, name, **params)
            commodity_pool_value = commodity_pool_instance.get_commodity_pool_value()
            commodity_pool_value.to_pickle(str(commodity_pool_data_folder_path.joinpath(f"{string}.pkl")))

if __name__ == "__main__":
    self = CommodityPoolManager()
    for group in ['FixedPool']:
        for name in ['FixedPool1']:
            self.save_commodity_pool(group=group, name=name)











#     def __init__(self) -> None:
#         """Constructor"""
#         self.commodity_pool_data_folder_path: Path = Path(__file__).parent.parent.\
#             joinpath("data").joinpath("commodity_pool")
#
#         self.commodity_pool_list: List[str] = None
#         self.commodity_pool_dict: Dict[str, DataFrame] = {}
#
#         self.init_commodity_pool_list()
#
#     def init_commodity_pool_list(self) -> None:
#         """
#         初始化查询商品池列表
#
#         该商品池列表通过遍历data/commodity_pool文件夹中的pkl文件汇总得到
#         """
#         self.commodity_pool_list = [file.replace(".pkl", "") for file in
#                                     os.listdir(self.commodity_pool_data_folder_path)]
#
#     def check_commodity_pool_name(self, pool_name: str) -> bool:
#         """
#         检查是否存在该商品池
#
#         Parameters
#         __________
#         pool_name: string
#                     商品池名称
#
#         Returns
#         _______
#         bool, 是否存在名称为pool_name的商品池
#         """
#         if not isinstance(self.commodity_pool_list, list):
#             self.init_commodity_pool_list()
#         return pool_name in self.commodity_pool_list
#
#     def get_commodity_pool(self, pool_name: str) -> DataFrame:
#         """
#         获取商品池数据
#
#         Parameters
#         __________
#         pool_name: string
#                     商品池名称
#
#         Returns
#         _______
#         commodity_pool: DataFrame
#                         商品池数据，index为交易时间，columns为品种代码，data为True or False
#
#         Examples
#         ________
#         >>> self = CommodityPoolManager()
#         >>> self.get_commodity_pool("dynamic_pool3")
#         underlying_symbol     A     AG    AL     AP  ...     WT     Y     ZC    ZN
#         datetime                                     ...
#         2009-01-05         True  False  True  False  ...  False  True  False  True
#         2009-01-06         True  False  True  False  ...  False  True  False  True
#         2009-01-07         True  False  True  False  ...  False  True  False  True
#         2009-01-08         True  False  True  False  ...  False  True  False  True
#         2009-01-09         True  False  True  False  ...  False  True  False  True
#                         ...    ...   ...    ...  ...    ...   ...    ...   ...
#         2020-12-16         True   True  True   True  ...  False  True   True  True
#         2020-12-17         True   True  True   True  ...  False  True   True  True
#         2020-12-18         True   True  True   True  ...  False  True   True  True
#         2020-12-21         True   True  True   True  ...  False  True   True  True
#         2020-12-22         True   True  True   True  ...  False  True   True  True
#         [2911 rows x 69 columns]
#         """
#         if not isinstance(self.commodity_pool_list, list):
#             self.init_commodity_pool_list()
#         if pool_name not in self.commodity_pool_list:
#             raise ValueError(f"No commodity pool named {pool_name}!")
#         if pool_name in self.commodity_pool_dict:
#             return self.commodity_pool_dict[pool_name]
#         else:
#             commodity_pool = pd.read_pickle(self.commodity_pool_data_folder_path.joinpath(f"{pool_name}.pkl"))
#             self.commodity_pool_dict[pool_name] = commodity_pool
#             return commodity_pool
#
#     def save_commodity_pool(self, pool_name: str, pool_df: DataFrame) -> None:
#         """
#         保存商品池数据
#
#         保存商品池数据的路径为data/commodity_pool
#
#         Parameters
#         ----------
#         pool_name: string
#                    商品池名称
#
#         pool_df: DataFrame
#                  商品池数据，index为交易时间，columns为品种代码，data为True or False
#
#         Returns
#         -------
#         None
#         """
#         self.commodity_pool_dict[pool_name] = pool_df
#         pool_df.to_pickle(str(self.commodity_pool_data_folder_path.joinpath(f"{pool_name}.pkl")))
#         self.init_commodity_pool_list()
#
#     def set_commodity_pool_info(self, pool_name: str, pool_info: str) -> None:
#         """
#         设置商品池信息
#
#         Parameters
#         __________
#         pool_name: string
#                     商品池名称
#         pool_info: string
#                     商品池信息
#
#         Returns
#         _______
#         None
#         """
#         info_file_path = self.commodity_pool_data_folder_path.joinpath("info.json")
#         if not os.path.exists(info_file_path):
#             json_pool_info = json.dumps({pool_name: pool_info})
#             with open(info_file_path, "w") as f:
#                 f.wirte(json_pool_info)
#         else:
#             with open(info_file_path, "rb") as f:
#                 pool_info = json.load(f)
#             pool_info[pool_name] = pool_info
#             json_pool_info = json.dumps(pool_info)
#             with open(info_file_path, "w") as f:
#                 f.write(json_pool_info)
#
#     def get_commodity_pool_info(self, pool_name: str) -> str:
#         """
#         获取商品池信息
#
#         Parameters
#         __________
#         pool_name: string
#                    商品池名称
#
#         Returns
#         _______
#         商品池信息
#         """
#         info_file_path = self.commodity_pool_data_folder_path.joinpath("info.json")
#         if not os.path.exists(info_file_path):
#             raise FileNotFoundError("No such file :info file path")
#         else:
#             with open(info_file_path, "rb") as f:
#                 pool_info = json.load(f)
#             if pool_name not in pool_info:
#                 raise KeyError(pool_name)
#             else:
#                 return pool_info[pool_name]
#
# if __name__ == "__main__":
#     self = CommodityPoolManager()
#     print(self.get_commodity_pool("dynamic_pool3"))






