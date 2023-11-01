import pandas as pd
from mytools import backtest
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文
plt.rcParams['axes.unicode_minus'] = False # 显示负号


def fama_macbeth_test(factors):
    res_list = []
    for fac_name in ['fac_size', 'fac_ret', 'fac_bm']:
        res_list.append(backtest.fama_macbeth(factors, fac_name))
    fama_macbeth_res = pd.DataFrame(res_list)
    print(fama_macbeth_res)


def main():
    factors = pd.read_csv("../cal_data/winsorize_factors.csv")
    factors['close_date'] = pd.to_datetime(factors['close_date'])

    print("对因子进行Fama-MacBeth回归检验，评价因子效果：")
    fama_macbeth_test(factors)
    print("""
    # 针对这一分析结果，三个因子t检验显著区别于0，是比较有效的因子；而其中账面市值比显著为正，其他两个显著为负数，也符合日常学术研究中对其的认知。
    # 其中，账面市值比因子回归后斜率分别为正负的数量基本相同，区分效应较差，因此从这一维度来说，他的效果并不是很好。\n\n
    """)

    print("绘图查看单因子分组收益情况：")
    for fac_name in ['fac_size', 'fac_ret', 'fac_bm']:
        group_rtns, group_cum_rtns = backtest.group_return_analysis(factors, fac_name)
        plt.savefig(f"../fig/因子{fac_name}分组收益率.png")
        plt.clf()
    print("回测后看出，三个因子都有一定的分组效果，其中账面市值比与市值因子分组效果最好，收益率因子分组效果相对差一些。\n\n")
    
    print("对单个因子进行百股策略回测：")
    for fac_name in ['fac_size', 'fac_ret', 'fac_bm']:
        rtn, evaluate_result = backtest.backtest_1week_nstock(factors, fac_name, True)
        plt.savefig(f"../fig/因子{fac_name}百股策略收益率.png")
        plt.clf()
        print(f"因子{fac_name}策略评价：")
        print(evaluate_result)

if __name__ == "__main__":
    main()