import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
from collections import defaultdict

from bases.base import BaseClass
from factor.base import BaseFactor
from data_manager.FactorDataManager import FactorDataManager

class BaseFactorTest(BaseClass):

    def __init__(self) -> None:
        """Constructor"""
        super().__init__()
        self.factor_data_manager: FactorDataManager = FactorDataManager()
        self.factor_test_result_path: Path = Path(__file__).parent.parent.joinpath("output_result")

        self.factor_info: Dict[str, str] = {}
        self.factor: BaseFactor = None
        self.factor_params: Dict[str, Any] = {}

    def set_factor(self, group: str, name: str, **params) -> None:
        """
        设置因子

        Parameters
        ----------
        group: str
                因子类别

        name: str
                因子名称

        params: 因子参数

        Returns
        -------
        None
        """
        self.factor_info = {'group': group, 'name': name}
        self.factor = self.factor_data_manager.get_factor(group=group, name=name, **params)
        self.factor_params = self.factor.get_params()

    def get_factor_autocorrelation(self, **params) -> None:
        """
        获取因子自相关性图

        Parameters
        ----------
        params: 参数, 包括输出图的figsize, 滞后期数lags

        Returns
        -------
        None
        """

        # 预先检查, 如果没有因子, 则报错
        if not isinstance(self.factor, BaseFactor):
            raise ValueError("Please specify factor first!")
        else:
            factor = self.factor
        factor_value = factor.factor_value

        # 确定滞后阶数参数
        if 'lags' in params:
            lags = params['lags']
        else:
            lags = 100

        # 生成自相关性数据
        autocorr_dict = defaultdict(list)
        for symbol in factor_value.columns:
            factor_series = factor_value[symbol]
            for lag in range(1, lags+1, 1):
                autocorr_dict[symbol].append(factor_series.autocorr(lag=lag))
        autocorr_df = pd.DataFrame(autocorr_dict, index=range(1, lags+1, 1))
        autocorr_series = autocorr_df.mean(axis=1)

        # 输出的路径
        factor_folder_path = self.factor_test_result_path.joinpath(factor.group).joinpath(factor.name)
        if not os.path.exists(factor_folder_path):
            os.makedirs(factor_folder_path)
        autocorrelation_file_path = factor_folder_path.joinpath("autocorrelation.png")

        # 图片大小
        if 'figsize' in params:
            plt.figure(figsize=params['figsize'])
        else:
            plt.figure(figsize=(20, 8))

        autocorr_series.plot()
        picture_name = f"{factor.group} {factor.name}"
        for key, value in self.factor_params.items():
            picture_name += f" {str(key)}={str(value)}"
        plt.title(picture_name)
        plt.grid()
        plt.savefig(str(autocorrelation_file_path))