# Carry Factor for Commodity Futures

## The article

I choose one article in zhihu, one popular Chinese website where people can ask questions and connect with people who contribute unique insights :

如何构建稳健的商品期货carry组合？: https://zhuanlan.zhihu.com/p/46197447

## Carry Factor

Carry factor is a very useful but simple one for commodity futures.  As we all know, there are several contracts with different maturity date for every kind of commodity futures. For example, in March 1st, 2021, there are IF2103, IF2104, IF2106, IF2106 for IF. However, although they are all IFs, but these contracts have different prices due to their different maturity dates, which is the reason why the carry factor is useful.

Normally, we define carry factor as S-F/(F*t2), where S is the price of the spot and F is the price of futures, t2 is the time between now and the maturity date of the futures. But this formula is just a theoretical one, since normally we cannot get the price of the spot.  As a consequence, we find different ways to compute the value of the carry factor:

1. the first way is to select two contracts for one kind of futures.  In this case, we have three choices obviously:

   a. main contract and sub main contract (the near contract as S, the far contract as F)

   b. near contract and sub near contract

   c. main contract and near contract(the nearest contract except the main contract)

2. the slope of y (the natrual logarithm of the price of different contracts) and x(the natural dates between the maturity dates of different contracts and the maturity date of the nearest contract).



## Construction of the backtesting framework

The backtesting framework is composed of the following parts:

1. bases
2. factor
3. commodity pool
4. signals
5. weight
6. backtesting
7. factor_test



## bases

As we all know,  factors, commodity pools, signals, weight, backtesting are all with params. One question we need to tackle is to deal with senarios with many different params. I learn a very sueful way from the way of managing params of BaseEstimator in sklearn. we can make as keywrod argument of the contruction method of every class involving params. For further reference, please visit the website:  https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html



For example, if we have a factor named factor1 with params R=5, H=10, then we just need to instantiate:

```python
self = factor1(R=5, H=10)
self.get_params()  # get_params
self.set_params(R=10, H=20)  # change params
```

Therefore, I construct a **BaseClass in the base.py in the bases module**. All factors, commodity pools, signals, weights can inherit from this class. Then all these classes can use the same useful method to manage params.

The code of the BaseClass class:

```python
import inspect
from abc import ABC
from collections import defaultdict

class BaseClass(ABC):
    """
    所有含参类的基类（包括因子类, 信号类, 权重类和回测类)
    """
    def __init__(self, **params) -> None:
        """Constructor"""
        for key, value in params.items():
            setattr(self, key, value)

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self
```



## Factor

All factors are child classes of one base class for factors: BaseFactor, which is a child class of BaseClass, which has been introduced in bases.

The code of the BaseFactor class:

```python
# 导入库
import inspect
from pathlib import Path
from pandas import DataFrame
from typing import Dict, List
from abc import  abstractmethod

from bases.base import BaseClass
from data_manager.DailyDataManager import DailyDataManager
from data_manager.BasicsDataManager import BasicsDataManager
from data_manager.ContinuousContractDataManager import ContinuousContractDataManager

class BaseFactor(BaseClass):
    """
    因子基类

    Parameters
    __________
    **params: 因子参数，以关键字参数的形式输入，可通过set_params方法设置参数，可通过get_params方法获取参数

    Attributes
    __________
    all_instruments: DataFrame
                    所有期货合约基础信息，共contract, underlying_symbol, market_tplus, symbol, margin_rate,maturity_date, type,
                    trading_code, exchange, product, contract_multiplier, round_lot, trading_hours, listed_date, industry_name,
                    de_listed_date, underlying_order_book_id这些field。

    symbol_industry_map: DataFrame
                        所有期货合约对应的行业（采用all_instruments中的行业）。两列DataFrame，第一列为underlying_symbol，第二列为industry_name

    daily_data_manager: DailyDataManager
                        日线数据管理器

    basics_data_manager: BasicsDataManager
                        基础数据管理器

    continuous_contract_data_manager: ContinuousContractDataManager
                                        连续合约数据管理器

    See Also
    ________
    bases.bases.BaseClass
    data_manager.DailyDataManager.DailyDataManager
    data_manager.BasicsDataManager.BasicsDataManager
    data_manager.ContinuousContractDataManager.ContinuousContractDataManager

    """
    def __init__(self, **params) -> None:
        """Constructor"""
        super().__init__(**params)

        self.group: str = Path(inspect.getfile(self.__class__)).parent.name
        self.name: str = self.__class__.__name__

        self.all_instruments: DataFrame = None
        self.symbol_industry_map: DataFrame = None

        self.daily_data_manager: DailyDataManager = DailyDataManager()
        self.basics_data_manager: BasicsDataManager = BasicsDataManager()
        self.continuous_contract_data_manager: ContinuousContractDataManager = ContinuousContractDataManager()

        self.maturity_date_dict: Dict[str, DataFrame] = {}

        self.factor_value: DataFrame = None
        self.init_basics_data()

    def init_basics_data(self) -> None:
        """
        初始化基础数据,获取所有期货合约基础信息all_instruments和品种行业对应表symbol_industry_map
        """
        if not isinstance(self.all_instruments, DataFrame):
            all_instruments = self.basics_data_manager.get_all_instruments()
            symbol_industry_map = all_instruments[['underlying_symbol', 'industry_name']].drop_duplicates()
            self.all_instruments = all_instruments
            self.symbol_industry_map = symbol_industry_map

    def set_factor_value(self, factor_value: DataFrame) -> None:
        """
        设置因子值

        Parameters
        ----------
        factor_value: DataFrame
                      因子值DataFrame,index为交易时间, columns为品种代码, data为因子值

        Returns
        -------
        None
        """
        self.factor_value = factor_value

    def get_factor_value(self) -> DataFrame:
        """
        获取因子值

        Returns
        -------
        factor_value: DataFrame
                      因子值DataFrame,index为交易时间, columns为品种代码, data为因子值
        """
        return self.factor_value

    def get_symbol_list(self) -> List[str]:
        """
        获取期货品种列表

        Returns
        -------
        symbol_list: 期货品种列表: List[str]
        """
        symbol_list: List[str] = self.basics_data_manager.get_symbol_list()
        return symbol_list

    def get_continuous_contract_data(self, symbol: str,
                                     contract: str = 'main',
                                     price: str = 'close',
                                     rebalance_num: int = 1) -> DataFrame:
        """
        获取连续合约数据, 按品种代码取, 不做stack

        Parameters
        ----------
        symbol: str,
                品种代码

        contract: str, default main
                选择以何种合约为基础的连续数据, main主力, active_near活跃近月

        price: str, default close
                选择以什么价格为基础的连续数据, close为收盘价, settlement结算价

        rebalance_num: str, default 1
                        换仓天数, 1或3或5

        Returns
        -------
        continuous_contract_data: DataFrame
                                    连续合约数据，字典包括datetime, contract_before_shift, contract, flag, old_contract,
                                    new_contract, old_weight, new_weight, old_return, new_return, return, cum_return,
                                    continuous_price, underlying_symbol
        """
        continuous_contract_data = self.continuous_contract_data_manager.get_continuous_contract_data(contract=contract,
                                                                                                price=price,
                                                                                                rebalance_num=rebalance_num)
        continuous_contract_data = continuous_contract_data[continuous_contract_data['underlying_symbol']==symbol]
        return continuous_contract_data

    def get_continuous_field(self, contract: str = 'main', price: str = 'close', rebalance_num: int = 1, field: str = 'continuous_price') -> DataFrame:
        """
        获取连续合约指定字段的数据

        Parameters
        ----------
        contract: str
                合约种类，目前可选有main和active_near, main表示主力合约, active_near表示活跃近月

        price: str
                选择以什么价格为基础的连续数据, close为收盘价, settlement结算价

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
                                                               field=field
                                                               )

    def get_field(self, symbol: str, field: str) -> DataFrame:
        """
        获取单品种所有合约在整个交易时间某数值字段的DataFrame

        Parameters
        ________
        symbol: string
                品种代码

        field:  string
                要获取的字段，如open, high, low, close, settlement, prev_settlement, open_interest, volume, upper_limit, lower_limit

        Returns
        -------
        data: DataFrame
                单品种所有合约在整个交易时间某数值字段的DataFrame，index是交易日期，columns是合约代码，data是字段值
        """
        # 预先检查
        data = self.daily_data_manager.get_field(symbol, field)
        return data

    def get_maturity_date(self, symbol: str) -> DataFrame:
        """
        获取某个品种所有合约的到期日

        Parameters
        __________
        symbol: string
                品种代码

        Returns
        __________
        品种到期日期，DataFrame，一共两列，第一列为合约(contract)，第二列为到期日(maturity_date)

        Examples
        ________
        >>> print(self.get_maturity_date('A'))
            contract maturity_date
        0      A0303    2003-03-14
        1      A0305    2003-05-23
        2      A0307    2003-07-14
        3      A0309    2003-09-12
        4      A0311    2003-11-14
        ..       ...           ...
        """
        maturity_date = self.basics_data_manager.get_maturity_date(symbol)
        return maturity_date

    @abstractmethod
    def compute_factor(self, *args, **kwargs) -> DataFrame:
        """
        计算因子抽象方法，需要在所有因子中重写

        Parameters
        ----------
        args: 可变位置参数

        kwargs: 可变关键字参数

        Returns
        -------
        因子值，index为交易时间, columns为品种代码, data为因子值
        """
        raise NotImplementedError

    def __repr__(self):
        group = self.group
        name = self.name
        title = ''
        title += f"factor(group={group}, name={name}, "
        # 添加因子参数
        for key, value in self.get_params().items():
            title += f"{key}={value}, "
        title = title[:-2]
        title += ")"
        return title

    def get_string(self) -> str:
        return self.__repr__()
```

## CommodityPool

## Signals

## Weight

## Backtesting

