import pandas as pd
import numpy as np
import os
from mytools import backtest
import time
from matplotlib import pyplot as plt

if not os.path.exists("../cal_data/"):
    os.mkdir("../cal_data") # 存储计算结果的路径
plt.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文
plt.rcParams['axes.unicode_minus'] = False # 显示负号


def match_rpt_date(date):
    """
    将日期转化为对应的报告期；
    基于：一季报最晚4/30公布，半年报8/30，三季报10/30，年报来年4/30（因此不用）
    """
    y = date.year
    m = date.month
    if m in (5, 6, 7, 8): return f"{y}0331"
    elif m in (9, 10): return f"{y}0630"
    elif m in (11, 12): return f"{y}0930"
    elif m in (1, 2, 3, 4): return f"{y-1}0930"

def dataloader():
    """
    导入并初步处理数据
    """
    # 数据导入
    print("正在导入市场相关数据...")
    start = time.time()

    stk_data = pd.read_csv("../data/stk_data.csv")
    stk_data['close_date'] = pd.to_datetime(stk_data['close_date'])
    stk_data['open_date'] = pd.to_datetime(stk_data['open_date'])

    open_days_data = pd.read_csv("../data/open_days_data.csv")
    open_days_data['date'] = pd.to_datetime(open_days_data['date'])

    equity = pd.read_csv("../data/eqy_belongto_parcomsh.csv")
    equity['rpt_date'] = pd.to_datetime(equity['rpt_date'])
    print(f"市场相关数据导入完成，用时{round(time.time()-start, 2)}秒\n")
    # 数据展示
    print("stk_data.csv: ", stk_data.shape)
    print("""
    沪深两市股票20060101-20230928周度股票数据 
    stock_code:股票代码
    open_date:开盘时间
    close_date:收盘时间
    open:后复权开盘价
    close:后复权收盘价
    uadj_close:未复权收盘价
    total_shares:总股本数""")
    print(stk_data.head())

    print("""
    沪深两市上市公司20050930-20230630报告期内归属母公司的股东权益数据 
    stock_code:股票代码
    rpt_date:报告期日期
    eqy_belongto_parcomsh:归属母公司的股东权益""")
    print("eqy_belongto_parcomsh.csv: ", equity.shape)
    print(equity.head())

    print("""
    沪深两市股票20060101-20230928，每周开盘日的高开低收量
    stock_code:股票代码
    date:交易日期
    high:最高价
    open:开盘价
    low:最低价
    close:收盘价
    volume:交易量""")
    print("open_days_data.csv: ", open_days_data.shape)
    print(open_days_data.head())
    print("")
    return stk_data, equity, open_days_data

def data_calculater(stk_data, equity, open_days_data):
    """
    进一步处理、计算数据
    """
    print("\n市场相关数据处理中...")
    start = time.time()
    # 计算市值
    stk_data['mkt_cap'] = stk_data['TOTAL_SHARES'] * stk_data['uadj_close'] 

    # 利用match_rpt_date函数计算每个交易周对应的报告期（匹配所有者权益）
    stk_data['rpt_date'] = pd.to_datetime(stk_data['close_date'] \
        .apply(lambda x: match_rpt_date(x))) 
    all_data = pd.merge(stk_data, equity, on=['stock_code', 'rpt_date'], how='left') 

    odd = {}
    for key in ['HIGH', 'OPEN', 'LOW', 'CLOSE', 'VOLUME']:
        odd[key] = pd.pivot(open_days_data, index='date', columns='stock_code', values=key)

    odd['pred_rtn'] = (odd['OPEN'].shift(-2)-odd['OPEN'].shift(-1))/odd['OPEN'].shift(-1)
    pred_rtn_na = odd['pred_rtn'].isna() # 不要把空值变成0

    # 下周停牌的股票只能获得0的收益
    vol0 = odd['VOLUME'].shift(-1)==0 
    volna = odd['VOLUME'].shift(-1).isna()
    odd['pred_rtn'][vol0 | volna & (~pred_rtn_na)] = 0 

    # 下周一字涨停的股票无法买入，只能获得0的收益
    yz = odd['HIGH'].shift(-1)==odd['LOW'].shift(-1) # “一字”，价格没有变化
    zt = ~(odd['CLOSE'].shift(-1) <= odd['CLOSE']) # “涨停”，价格不比上周高
    odd['pred_rtn'][yz & zt & (~pred_rtn_na)] = 0 

    pred_rtn = odd['pred_rtn'].stack().reset_index().rename(columns={0: 'pred_rtn', 'date': 'open_date'})
    all_data = pd.merge(all_data, pred_rtn, on=['open_date', 'stock_code'], how='left')
    all_data = all_data[~all_data['pred_rtn'].isna()]
    print(f"市场相关数据处理完成，用时{round(time.time()-start, 2)}秒")
    return all_data

def factor_calculater(all_data):
    print("\n正在计算相关因子...")
    start = time.time()
    # 计算周收益率因子
    close = pd.pivot(all_data, index='close_date', columns='stock_code', values='CLOSE')
    fac_ret = (close-close.shift(1))/close.shift(1)
    fac_ret = fac_ret.stack().reset_index().rename(columns={0: 'fac_ret', 'date': 'close_date'})
    all_data = pd.merge(all_data, fac_ret, on=['close_date', 'stock_code'], how='left')

    # 计算规模因子
    all_data['fac_size'] = np.log(all_data['mkt_cap']/1000000)

    # 账面市值比因子
    all_data['fac_bm'] = all_data['EQY_BELONGTO_PARCOMSH'] / all_data['mkt_cap']

    # 因子数据整理
    factors = all_data[['stock_code', 'close_date', 'pred_rtn', 'fac_ret', 'fac_size', 'fac_bm']].reset_index(drop=True)
    factors = factors[~factors['pred_rtn'].isna()]
    print(f"因子数据计算完成，用时{round(time.time()-start, 2)}秒")
    print("factors数据展示: ")
    print(factors.head())
    factors.to_csv("../cal_data/factors.csv", index=False)
    return factors


def winsorize_factors(factors):
    fac_name = 'fac_size'
    factors[factors['close_date']=='2019-10-18'][fac_name].plot.kde(title="2019-10-18 Size因子分布情况（截尾前）")
    plt.savefig("../fig/20191018_Size因子分布情况（截尾前）.png")
    factors = backtest.winsorize_factor(factors, 'fac_size')
    factors = backtest.winsorize_factor(factors, 'fac_ret')
    factors = backtest.winsorize_factor(factors, 'fac_bm')
    factors.to_csv("../cal_data/winsorize_factors.csv", index=False)
    plt.clf()
    factors[factors['close_date']=='2019-10-18'][fac_name].plot.kde(title="2019-10-18 Size因子分布情况（截尾后）")
    plt.savefig("../fig/20191018_Size因子分布情况（截尾后）.png")
    factors.to_csv("../cal_data/winsorize_factors.csv", index=False)
    return factors

def main():
    stk_data, equity, open_days_data = dataloader()
    all_data = data_calculater(stk_data, equity, open_days_data)
    factors = factor_calculater(all_data)
    factors = winsorize_factors(factors)

if __name__ == "__main__":
    main()