import numpy as np

# 在我的个人网站www.6young.site中，分享过更多相关回测指标的实现方式
# 如果感兴趣可以进行查看（基于GitHub pages搭建，使用科学上网访问速度更快）
# 计算年(季/月/周)化收益的相关常数
BDAYS_PER_YEAR = 252
BDAYS_PER_QTRS = 63
BDAYS_PER_MONTH = 21
BDAYS_PER_WEEK = 5

DAYS_PER_YEAR = 365
DAYS_PER_QTRS = 90
DAYS_PER_MONTH = 30
DAYS_PER_WEEK = 7

MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52
QTRS_PER_YEAR = 4


def get_period_days(period):
    '''不同时期指标转化'''
    period_days = {
        'yearly': BDAYS_PER_YEAR, 'quarterly': BDAYS_PER_QTRS,
        'monthly': BDAYS_PER_MONTH, 'weekly': BDAYS_PER_WEEK, 'daily': 1, 
        
        'monthly2yearly': MONTHS_PER_YEAR,
        'quarterly2yearly': QTRS_PER_YEAR, 
        'weekly2yearly': WEEKS_PER_YEAR, 
    }
    return period_days[period]

def annual_info(returns_df, period='yearly'):
    """
    年化收益与波动率
    """
    period_days = get_period_days(period)
    total_return = (returns_df + 1).prod(axis=0)
    annual_ret = total_return ** (period_days / returns_df.shape[0]) - 1
    annual_vol = returns_df.std() * (period_days ** 0.5)
    res_dict = {
        'annual_return': annual_ret,
        'annual_volatility': annual_vol,
    }
    return res_dict


def sharpe_ratio(returns_df, risk_free=0, period='yearly'):
    """
    计算（年化）夏普比率
    """
    period_days = get_period_days(period)
    sr = (returns_df.mean() - risk_free) * (period_days ** 0.5) / returns_df.std()
    res_dict = {'sharpe_ratio': sr}
    return res_dict


def maximum_drawdown(returns_df):
    """
    计算最大回撤
    """
    cum_returns = (returns_df + 1).cumprod(axis=0)
    peak = cum_returns.expanding().max()
    dd = ((peak - cum_returns)/peak)
    mdd = dd.max()
    end = dd[dd==mdd].dropna().index[0]
    start = peak[peak==peak.loc[end]].index[0]
    res_dict = {
        'max_drawdown': mdd, 'max_drawdown_start': start,
        'max_drawdown_end': end, 
    }
    return res_dict

def sortino_ratio(returns_df, minimum_acceptable_return=0, period='yearly'):
    """
    计算年化sortino比率
    """
    period_days = get_period_days(period)
    downside_returns = returns_df[returns_df < 0]  # 筛选出负收益
    downside_volatility = np.std(downside_returns, axis=0)
    excess_return = returns_df.mean() - minimum_acceptable_return
    sr = excess_return*(period_days**0.5) / downside_volatility
    res_dict = {'sortino_ratio': sr}
    return res_dict
