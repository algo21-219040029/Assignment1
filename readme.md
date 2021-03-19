# Carry Factor for Commodity Futures

## Reference Article

I choose one article in zhihu, one popular Chinese website where people can ask questions and connect with people who contribute unique insights: \
**如何构建稳健的商品期货carry组合？**: https://zhuanlan.zhihu.com/p/46197447

## Carry Factor

Carry factor is a very useful but simple one for commodity futures.  As we all know, there are several contracts with different maturity date for every kind of commodity futures. For example, in March 1st, 2021, there are IF2103, IF2104, IF2106, IF2106 for IF. However, although they are all IFs, but these contracts have different prices due to their different maturity dates, which is the reason why the carry factor is useful.

Normally, we define carry factor as (S-F)/(F*t2), where S is the price of the spot and F is the price of futures, t2 is the time between now and the maturity date of the futures. But this formula is just a theoretical one, since normally we cannot get the price of the spot.  As a consequence, we find different ways to compute the value of the carry factor:

1. the first way is to select two contracts for one kind of futures.  In this case, we have three choices obviously:

   a. main contract and sub main contract (the near contract as S, the far contract as F)

   b. near contract and sub near contract

   c. main contract and near contract(the nearest contract except the main contract)

2. the slope of y (the natrual logarithm of the price of different contracts) and x(the natural dates between the maturity dates of different contracts and the maturity date of the nearest contract).

# Construction of the backtesting framework

The backtesting framework is composed of the following parts:

1. bases
2. factor
3. commodity pool
4. signals
5. weight
6. backtesting
7. factor_test

# bases

As we all know,  factors, commodity pools, signals, weight, backtesting are all with params. One question we need to tackle is to deal with senarios with many different params. I learn a very sueful way from the way of managing params of BaseEstimator in sklearn. we can make as keywrod argument of the contruction method of every class involving params. For further reference, please visit the website:  
https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html

For example, if we have a factor named A with params R=5, H=10, then we just need to instantiate:


```python
from bases.base import BaseClass

class A(BaseClass):
    def __init__(self, R: int, H: int = 10) -> None:
        super().__init__(R=R, H=H)
```


```python
a = A(R=5, H=10)
```


```python
a.get_params()
```




    {'H': 10, 'R': 5}




```python
a.set_params(R=20, H=5)
a.get_params()
```




    {'H': 5, 'R': 20}



From the example above, we can know that one class inheriting from BaseClass can have several parameters. Once an instance of this class is instantiated, we need to specify the parameters. Besides, we can use the method **get_params** to get the parameters and we can also use the method **set_params** to set or change the parameters.


**In my framework, factor class, signal class, weight class and backtesting class are all inherited from BaseClass.** As a consequence, we apply this method of managine parameters in the same way.

# factor

## factor group and name

In the factor package, I define a class named BaseFactor, which is the base class of all factors. And every factor has its own group and name. Sometimes, we may also define a base class for every factor group. For example, every carry factor is inherited from the BaseCarryFactor, while the BaseCarryFactor is inherited from the BaseCalss.

For example, in the factor package, there is one folder named CarryFactor and in the folder there are several .py file such as FarNearFactor.py, MainNearFactor1.py, MainNearFactor2.py, etc. And in every .py file, there is one class with the same name as the .py file, which is how the factor is defined. For the MainNearFactor3, its group is CarryFactor and its name is MainNearFactor3. For the factor MainNearFactor3, it is a kind of CarryFactor and its name is MainNearFactor3.

## How to define a new factor

To define a new factor, we need to specfy its group and name. If it is included in a new group, you can create a new python package in the factor package. Otherwise, the .py file of the new factor can be in an existing python package, which means it is in this group.

### common method: compute_factor

The method compute_factor is an abstractmethod in the BaseClass. Therefore, every factor must redefine this method. This method is the place where we define how to compute the factor.

The following is the source code of MainNearFactor3.


```python
import numpy as np
import pandas as pd
from tqdm import tqdm
from pandas import DataFrame

from factor.CarryFactor.base import BaseCarryFactor
from utils.utility import stack_dataframe_by_fields
from factor.contract_utility import get_near_contract_except_main

class MainNearFactor3(BaseCarryFactor):
    """
    利用主力合约和去除主力合约后最近合约计算的期限结构因子

    合约: 主力合约和去除主力合约后最近合约，两者中近的合约定义为近月合约，远的合约定义为远月合约. 若合约只有一个，则因子值为缺失值

    时间差: 月份差

    计算方法: (近月合约价格-远月合约价格)/(远月合约价格*时间差)

    Attributes
    __________
    price: str
            用于代表合约价格的字段, close或settlement

    window: int
            因子平滑参数，所有因子都具有

    See Also
    ________
    factor.CarryFactor.bases.BaseFactor
    """

    def __init__(self, price: str = 'close', window: int = 1):
        """
        Constructor

        Parameters
        ----------
        price: str
                用于代表合约价格的字段，close或settlement

        window: int
                因子平滑参数，所有因子都具有
        """
        super().__init__(price=price, window=window)

    def compute_single_factor(self, symbol: str) -> DataFrame:
        """
        计算单品种的因子值

        Parameters
        ----------
        symbol: str
                品种代码

        Returns
        -------
        因子值: DataFrame
                默认的index, 三列, datetime, underlying_symbol(均为symbol), factor
        """
        params = self.get_params()
        price = params['price']
        main_contract_df = self.get_continuous_contract_data(symbol=symbol, price=price)
        main_contract_df = main_contract_df[['datetime', 'contract_before_shift']]\
            .rename(columns={'contract_before_shift': 'main_contract'})

        # 获取除主力合约以外的近月合约
        price_df: DataFrame = self.get_field(symbol=symbol, field=price)
        stack_price_df: DataFrame = price_df.stack().to_frame("price").reset_index()
        near_contract_df = get_near_contract_except_main(price_df, main_contract_df)

        # 获取每个合约的到期日
        maturity_date = self.get_maturity_date(symbol)

        # 预先拼接工作
        main_contract_df = pd.merge(left=main_contract_df,
                                    right=maturity_date.rename(columns={'contract': 'main_contract',
                                                                        'maturity_date':
                                                                            'main_contract_maturity_date'}),
                                    on='main_contract', how='left')

        main_contract_df = pd.merge(left=main_contract_df,
                                    right=stack_price_df.rename(
                                        columns={'contract': 'main_contract', 'price': 'main_price'}),
                                    on=['datetime', 'main_contract'], how='left')

        near_contract_df = pd.merge(left=near_contract_df,
                                    right=maturity_date.rename(columns={'contract': 'near_contract',
                                                                        'maturity_date':
                                                                            'near_contract_maturity_date'}),
                                    on='near_contract', how='left')

        near_contract_df = pd.merge(left=near_contract_df,
                                    right=stack_price_df.rename(
                                        columns={'contract': 'near_contract', 'price': 'near_price'}),
                                    on=['datetime', 'near_contract'], how='left')

        # factor: 因子值
        main_near_df = pd.concat([main_contract_df.set_index('datetime'), near_contract_df.set_index('datetime')],
                                 axis=1).reset_index()
        main_near_df.index = range(len(main_near_df))
        # 远月合约价格
        main_near_df['far_price'] = pd.Series(np.where(main_near_df['main_contract'] > main_near_df['near_contract'],
                                                       main_near_df['main_price'],
                                                       main_near_df['near_price']))

        main_near_df['near_close'] = pd.Series(np.where(main_near_df['main_contract'] < main_near_df['near_contract'],
                                                        main_near_df['main_price'],
                                                        main_near_df['near_price']))

        main_near_df['far_contract'] = pd.Series(np.where(main_near_df['main_contract'] > main_near_df['near_contract'],
                                                          main_near_df['main_contract'],
                                                          main_near_df['near_contract']))

        main_near_df['near_contract'] = pd.Series(
            np.where(main_near_df['main_contract'] < main_near_df['near_contract'],
                     main_near_df['main_contract'],
                     main_near_df['near_contract']))

        main_near_df['near_maturity_date'] = pd.Series(
            np.where(main_near_df['main_contract_maturity_date'] < main_near_df['near_contract_maturity_date'],
                     main_near_df['main_contract_maturity_date'],
                     main_near_df['near_contract_maturity_date']))

        main_near_df['far_maturity_date'] = pd.Series(
            np.where(main_near_df['main_contract_maturity_date'] > main_near_df['near_contract_maturity_date'],
                     main_near_df['main_contract_maturity_date'],
                     main_near_df['near_contract_maturity_date']))

        # main_near_df['date_delta'] = (main_near_df['near_maturity_date'] -
        #                               main_near_df['far_maturity_date']).dt.days

        main_near_df['far_year_month'] = pd.to_datetime('20' + main_near_df['far_contract'].str[-4:] + '01')
        main_near_df['far_year'] = main_near_df['far_year_month'].dt.year
        main_near_df['far_month'] = main_near_df['far_year_month'].dt.month

        main_near_df['near_year_month'] = pd.to_datetime('20' + main_near_df['near_contract'].str[-4:] + '01')
        main_near_df['near_year'] = main_near_df['near_year_month'].dt.year
        main_near_df['near_month'] = main_near_df['near_year_month'].dt.month

        main_near_df['month_delta'] = -((main_near_df['far_year'] - main_near_df['near_year']) * 12 +
                                        (main_near_df['far_month'] - main_near_df['near_month']))

        main_near_df['factor'] = (main_near_df['near_price'] - main_near_df['far_price']) / (main_near_df['far_price'] *
                                                                                             main_near_df[
                                                                                                 'month_delta'])

        main_near_df = main_near_df[['datetime', 'near_contract', 'far_contract',
                                    'near_price', 'far_price',
                                    'month_delta', 'factor']]

        main_near_df['underlying_symbol'] = symbol
        factor =main_near_df[['datetime', 'underlying_symbol', 'factor']]
        return factor

    def compute_factor(self) -> DataFrame:
        """
        计算因子，通过对compute_single_factor实现

        Returns
        -------
        因子值: DataFrame
                index为datetime, columns为underlying_symbol, data为factor
        """

        symbol_list = self.get_symbol_list()
        factor_list = []
        for symbol in tqdm(symbol_list):
            factor = self.compute_single_factor(symbol)
            factor_list.append(factor)
        factor = pd.concat(factor_list, axis=0)
        factor = stack_dataframe_by_fields(data=factor,
                                           index_field='datetime',
                                           column_field='underlying_symbol',
                                           data_field='factor')
        factor = factor.rolling(window=self.window, min_periods=1).mean()
        self.factor_value = factor
        return factor
```

## FactorDataManager

FactorDataManager is the class where we can get the factor, save the factor locally.


```python
from data_manager.FactorDataManager import FactorDataManager
self = FactorDataManager()
factor = self.get_factor(group='CarryFactor', name='MainNearFactor3', price='close')
```


```python
factor
```




    factor(group=CarryFactor, name=MainNearFactor3, price=close, window=1)




```python
factor.factor_value
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>underlying_symbol</th>
      <th>A</th>
      <th>AG</th>
      <th>AL</th>
      <th>AP</th>
      <th>AU</th>
      <th>B</th>
      <th>BB</th>
      <th>BC</th>
      <th>BU</th>
      <th>C</th>
      <th>...</th>
      <th>TF</th>
      <th>TS</th>
      <th>UR</th>
      <th>V</th>
      <th>WH</th>
      <th>WR</th>
      <th>WT</th>
      <th>Y</th>
      <th>ZC</th>
      <th>ZN</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2009-01-05</th>
      <td>-0.018093</td>
      <td>NaN</td>
      <td>-0.002926</td>
      <td>NaN</td>
      <td>0.001314</td>
      <td>-0.042694</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.018340</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.010199</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>-0.038462</td>
      <td>NaN</td>
      <td>-0.003555</td>
    </tr>
    <tr>
      <th>2009-01-06</th>
      <td>-0.015901</td>
      <td>NaN</td>
      <td>-0.007438</td>
      <td>NaN</td>
      <td>-0.004756</td>
      <td>-0.035408</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.017197</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.009801</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>-0.028207</td>
      <td>NaN</td>
      <td>-0.004711</td>
    </tr>
    <tr>
      <th>2009-01-07</th>
      <td>-0.010948</td>
      <td>NaN</td>
      <td>-0.006574</td>
      <td>NaN</td>
      <td>-0.001460</td>
      <td>-0.037962</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.015370</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.008337</td>
      <td>NaN</td>
      <td>0.004963</td>
      <td>-0.023498</td>
      <td>NaN</td>
      <td>-0.011883</td>
    </tr>
    <tr>
      <th>2009-01-08</th>
      <td>-0.013158</td>
      <td>NaN</td>
      <td>-0.007882</td>
      <td>NaN</td>
      <td>-0.004557</td>
      <td>-0.044682</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.015575</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.007583</td>
      <td>NaN</td>
      <td>0.006484</td>
      <td>-0.034365</td>
      <td>NaN</td>
      <td>-0.013233</td>
    </tr>
    <tr>
      <th>2009-01-09</th>
      <td>-0.009965</td>
      <td>NaN</td>
      <td>-0.009544</td>
      <td>NaN</td>
      <td>-0.002204</td>
      <td>-0.038712</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.016624</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.009563</td>
      <td>NaN</td>
      <td>0.006632</td>
      <td>-0.026808</td>
      <td>NaN</td>
      <td>-0.011866</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-12-16</th>
      <td>0.001099</td>
      <td>0.004384</td>
      <td>0.000000</td>
      <td>0.008941</td>
      <td>0.001942</td>
      <td>0.014957</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.015407</td>
      <td>0.006334</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.008255</td>
      <td>-0.004639</td>
      <td>0.000000</td>
      <td>-0.008814</td>
      <td>NaN</td>
      <td>-0.017134</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2020-12-17</th>
      <td>0.000804</td>
      <td>0.003908</td>
      <td>0.000000</td>
      <td>0.014377</td>
      <td>0.001758</td>
      <td>0.007893</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.015143</td>
      <td>0.006984</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.007174</td>
      <td>-0.006861</td>
      <td>0.000000</td>
      <td>-0.008774</td>
      <td>NaN</td>
      <td>-0.019160</td>
      <td>-0.014135</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2020-12-18</th>
      <td>0.002490</td>
      <td>0.004064</td>
      <td>0.000000</td>
      <td>0.020771</td>
      <td>0.001978</td>
      <td>0.012164</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.014482</td>
      <td>0.006692</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.008034</td>
      <td>-0.004820</td>
      <td>0.000000</td>
      <td>-0.006454</td>
      <td>NaN</td>
      <td>-0.022668</td>
      <td>-0.015479</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2020-12-21</th>
      <td>0.004269</td>
      <td>0.001749</td>
      <td>0.000000</td>
      <td>0.021848</td>
      <td>0.002432</td>
      <td>-0.033062</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.014565</td>
      <td>0.008507</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.011454</td>
      <td>-0.003117</td>
      <td>0.000000</td>
      <td>0.001016</td>
      <td>NaN</td>
      <td>-0.022597</td>
      <td>-0.014420</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2020-12-22</th>
      <td>0.002366</td>
      <td>0.002779</td>
      <td>0.000000</td>
      <td>0.019291</td>
      <td>0.003236</td>
      <td>-0.022762</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.014098</td>
      <td>0.008220</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.007692</td>
      <td>-0.004521</td>
      <td>0.000000</td>
      <td>0.006200</td>
      <td>NaN</td>
      <td>-0.023841</td>
      <td>-0.022606</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>2911 rows × 69 columns</p>
</div>



# commodity_pool

Commodity pool is used to filter commodity futures beferoe we run the backtest. There are many different commodity pools. In this project, I define several commodity pools, such as DynamicPool1, DynamicPool2, DynamicPool3, etc.


```python

```
