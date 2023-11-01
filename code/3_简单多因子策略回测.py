import pandas as pd
from mytools import backtest
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文
plt.rcParams['axes.unicode_minus'] = False # 显示负号

def main():
    factors = pd.read_csv("../cal_data/winsorize_factors.csv")
    factors['close_date'] = pd.to_datetime(factors['close_date'])

    print("测试简单分组打分法：")
    rtn, evaluate_result = backtest.mutifactor_score(factors, ['-fac_ret', '-fac_size', 'fac_bm'], group_num=10)
    plt.savefig(f"../fig/基于简单打分法的多因子百股策略收益率.png")
    plt.clf()
    print(f"简单打分法策略评价：")
    print(evaluate_result)
    print("相比于单个”市值“因子，因子组合后效果变差了。\n\n")

    print("测试多元回归选股法：")
    rtn, evaluate_result = backtest.mutifactor_regression(factors, ['fac_ret', 'fac_size', 'fac_bm'], stock_num=100, plot=True)
    plt.savefig(f"../fig/基于多元回归选股法的多因子百股策略收益率.png")
    plt.clf()
    print(f"多元回归选股法策略评价：")
    print(evaluate_result)
    print("""
    对比前面的诸多策略，该策略的收益率并不算高（尤其和市值因子相比），这也是因为我们回归后的系数滞后了两周才进行预测的结果。
    但是，整体来看该策略的效果是比Ret,B/M因子的效果好的，而且相比于Size因子，该策略可以很好的消除市场风格的影响。
    在2017年的大盘股行情中，该策略的最大回撤只有25%，比单纯的Size因子好很多。""")

if __name__ == "__main__":
    main()