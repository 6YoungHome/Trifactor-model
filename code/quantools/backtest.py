from tqdm import tqdm, trange
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

from . import evaluate


def winsorize_factor(factor, fac_name):
    '''
    为了保证因子的稳定性,对因子进行截尾（将<0.01 >0.99的因子调整为1/99分位数）
    '''
    def winsorize(col, fac_name):
        if col[fac_name] > col['p99']:
            return col['p99']
        elif col[fac_name] < col['p1']:
            return col['p1']
        else:
            return col[fac_name]
    factors = factor.copy()
    p = factors.groupby('close_date')[fac_name].quantile(0.99).reset_index().rename(columns={fac_name: 'p99'})
    p['p1'] = factors.groupby('close_date')[fac_name].quantile(0.01).reset_index()[fac_name]
    factors = factors.merge(p, on='close_date')

    tqdm.pandas(desc=f"Winsorizing the factor {fac_name}: ")
    factors[fac_name] = factors.progress_apply(lambda x: winsorize(x, fac_name), axis=1)
    factors = factors.drop(['p1', 'p99'], axis=1)
    return factors


def fama_macbeth(factor, fac_name):
    """
    返回fama_macbeth回归检验参数
    """
    def cal_beta(data, fac_name):
        """线性回归获得斜率参数"""
        x = data[fac_name].values.reshape((-1, 1))
        y = data['pred_rtn'].values
        model = LinearRegression()
        model.fit(x, y)
        return model.coef_[0]
    
    factors = factor.copy()
    factors = factors[~factors[fac_name].isna()]
    tqdm.pandas(desc=f"Fama-MacBeth Regression by factor {fac_name}: ")
    fg = factors.groupby('close_date').progress_apply(lambda x: cal_beta(x, fac_name)).reset_index()
    t, p = stats.ttest_1samp(fg[0], 0)
    pos_count, neg_count = (fg[0]>0).sum(),(fg[0]<0).sum()
    res_dict = {
        'fac_name': fac_name, 't': t, 'p': p,
        'pos_count': pos_count, 'neg_count': neg_count
    }
    return res_dict


def group_return_analysis(factor, fac_name, group_num=10, plot=True):
    factors = factor.copy()
    factors = factors[~factors[fac_name].isna()]
    factors['rank'] = factors.groupby('close_date')[fac_name].rank(method='first')
    factors['group'] = factors.groupby('close_date')['rank'].apply(lambda x: pd.qcut(x, group_num, range(group_num)))

    group_cum_rtns = pd.DataFrame()
    group_rtns = pd.DataFrame()
    for i in trange(group_num, desc="Calculating groups"):
        rtn = factors[factors['group']==i].groupby('close_date').mean()[['pred_rtn']]
        rtn['cum_rtn'] = (rtn['pred_rtn'] + 1).cumprod()
        group_cum_rtns[i] = rtn['cum_rtn']
        group_rtns[i] = rtn['pred_rtn']
    
    if plot:
        fig = plt.figure(figsize=(16, 5))
        ax1, ax2 = fig.subplots(1, 2)
        for i in range(group_num):
            group_cum_rtns[i].plot(ax=ax1, label=i)
        ax1.set_title(f"{group_num} group return of factor {fac_name}")
        ax1.legend()
        
        annual_rtns = [evaluate.annual_info(group_rtns[i])['annual_return'] for i in range(group_num)]
        ax2.bar(range(group_num), annual_rtns)
        ax2.set_title(f"Annual return of {group_num} groups")
    return group_rtns, group_cum_rtns


def get_strategy_rtn(factor, fac_name, reverse=False, n=100):
    """
    利用因子计算策略的收益率
    """
    factors = factor.copy()
    factors = factors[~factors[fac_name].isna()]
    factors['rank'] = factors.groupby('close_date')[fac_name].rank(method='first', ascending=reverse)
    factors = factors[factors['rank']<=n]
    rtn = factors.groupby('close_date')['pred_rtn'].mean().reset_index()
    rtn['cum_rtn'] = (1+rtn['pred_rtn']).cumprod()
    rtn = rtn.set_index('close_date')
    return rtn


def evaluate_strategy(return_df, yearly_evaluate=True):
    """
    (按年)规范化计算评价策略的部分指标,返回一个DataFrame
    """
    def _get_index(return_df):
        """
        计算评价策略的部分指标
        """
        res_dict = {}
        res_dict.update(evaluate.sharpe_ratio(return_df))
        res_dict.update(evaluate.maximum_drawdown(return_df))
        res_dict.update(evaluate.sortino_ratio(return_df))
        res_dict.update(evaluate.annual_info(return_df))
        return res_dict
    
    res_dict = _get_index(return_df['pred_rtn'])
    res_dict['section'] = 'Sum'
    res_dict_ls = [res_dict]
    if yearly_evaluate:
        years = [i for i in range(return_df.index[0].year, return_df.index[-1].year+1)]
        for year in years:
            res_dict = _get_index(return_df.loc[str(year), 'pred_rtn'])
            res_dict['section'] = str(year)
            res_dict_ls.append(res_dict)
    evaluate_result = pd.DataFrame(res_dict_ls)
    return evaluate_result


def backtest_1week_nstock(factor, fac_name, reverse=False, n=100, plot=True, yearly_evaluate=True):
    """
    利用因子计算策略的收益率，绘图，并计算相关指标
    Args:
        factor (_pd.DataFrame_): 因子数据框
        fac_name (_str_): 因子名称
        reverse (_bool_, optional): 正向因子or负向因子
        n (_int_, optional): 选取股票的数量（默认100只）
        plot (_bool_, optional): 是否绘图（默认绘图）
        yearly_evaluate (_bool_, optional): 是否评价策略每年情况（默认评价）

    Returns:
        _type_: 策略收益率, 回测指标信息
    """
    rtn = get_strategy_rtn(factor, fac_name, reverse, n)
    if plot:
        rtn['cum_rtn'].plot(figsize=(10,5), title=f'Cummulate return of factor {fac_name}')
    evaluate_result = evaluate_strategy(rtn[['pred_rtn']], yearly_evaluate)
    return rtn, evaluate_result


def mutifactor_score(factor, fac_names, group_num=10, stock_num=100, plot=True):
    """
    一个简单的通过打分的多因子选股方法
    Args:
        factor (_pd.DataFrame_): 因子数据框
        fac_names (_list_): 准备使用的多因子列表（反向因子前加"-"号即可）
        group_num (_int_, optional): _description_. 分组组数，默认为10
        stock_num (_int_, optional): _description_. 打分选股股票数，默认为100
        plot (_bool_, optional): _description_. 是否画图，默认为是
    """
    def fac_name_parse(fac_name):
        """解析因子名，判断是否为反向因子"""
        if fac_name[0] == '-':
            return fac_name[1::], False
        else:
            return fac_name, True
    
    factors = factor.copy()
    factors = factors.dropna()
    for fac_name in fac_names:
        fac_name, reverse = fac_name_parse(fac_name)
        factors['rank'] = factors.groupby('close_date')[fac_name] \
            .rank(method='first', ascending=reverse)
        factors[fac_name+'_score'] = factors.groupby('close_date')['rank'] \
            .apply(lambda x: pd.qcut(x, group_num, range(group_num)))
        
    factors['score'] = factors.loc[:, factors.columns.str.contains('score')].sum(axis=1)
    
    rtn, evaluate_result = backtest_1week_nstock(factors, 'score', plot=plot, n=stock_num)
    return rtn, evaluate_result


def mutifactor_regression(factor, fac_names, stock_num=100, plot=True):
    """
    一个简单的通过打分的多因子选股方法
    Args:
        factor (_pd.DataFrame_): 因子数据框
        fac_names (_list_): 准备使用的多因子列表（反向因子前加"-"号即可）
        stock_num (_int_, optional): _description_. 打分选股股票数，默认为100
        plot (_bool_, optional): _description_. 是否画图，默认为是
    """
    def cal_beta(data, fac_names):
        """
        通过线性回归获得斜率参数
        按次序依此为alpha与beta_i
        """
        data = data.dropna()
        x = data[fac_names].values.reshape((-1, len(fac_names)))
        y = data['pred_rtn'].values
        model = LinearRegression()
        model.fit(x, y)
        return tuple([model.intercept_]+[i for i in model.coef_])
        # return model.intercept_, *model.coef_
    
    def cal_reg_pred_rtn(col, fac_names, beta_fac_names):
        """
        根据系数计算预测后的回归收益
        """
        factor_nums = len(fac_names)
        reg_pred_rtn = col['alpha']
        for i in range(factor_nums):
            reg_pred_rtn += col[fac_names[i]] * col[beta_fac_names[i+1]]
        return reg_pred_rtn
    
    factors = factor.copy().dropna()
    
    # 计算回归系数
    beta_fac_names = ["alpha"] + [fac_name+"_beta" for fac_name in fac_names] # 各个因子回归系数的列表
    alpha_beta = factors.groupby('close_date').apply(lambda x: cal_beta(x, fac_names)).to_frame()
    alpha_beta[beta_fac_names] = alpha_beta[[0]].apply(lambda x: x[0], axis=1, result_type="expand")
    alpha_beta = alpha_beta.drop(0, axis=1)
    alpha_beta = alpha_beta.shift(2).reset_index() # 回归得到的系数事实上只能预测两周后的收益
    
    factors = pd.merge(factors, alpha_beta, on="close_date", how="left")
    
    tqdm.pandas(desc="Calculating regression return: ")
    factors['reg_pred_rtn'] = factors.progress_apply(lambda x:cal_reg_pred_rtn(x, fac_names, beta_fac_names), axis=1)

    rtn, evaluate_result = backtest_1week_nstock(factors, 'reg_pred_rtn', plot=plot, n=stock_num)
    return rtn, evaluate_result