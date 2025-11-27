import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm  # 进度条
from pandas import Interval
import pyarrow.parquet as pq
from scipy import stats  
import statsmodels.api as sm
import glob
from utils.path_manager import PathManager

# ==============================================================================
#                                 数学与统计辅助函数
# ==============================================================================

def get_newey_west_tstat(ic_series):
    """计算IC序列的Newey-West T-statistic和p-value"""
    if ic_series.empty or len(ic_series) < 2:
        return np.nan, np.nan
    
    N = len(ic_series)
    lags = int(4 * (N / 100)**(2/9))
    
    # 转换为 numpy array 避免索引对齐开销
    X = sm.add_constant(np.ones(N))
    y = ic_series.values
    
    try:
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lags, 'use_correction': True})
        return model.tvalues[0], model.pvalues[0]
    except Exception:
        if ic_series.std() == 0 or np.isnan(ic_series.std()):
            return np.nan, np.nan
        t_stat, p_value = stats.ttest_1samp(ic_series, 0.0, nan_policy='omit')
        return t_stat, p_value

def get_valid_pool(start_date, end_date, stk_range, input_dir):
    '''获取股票池中每日股票代码 (IO优化版)'''
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    pool_file_map = {
        'all': 'stock_index_weight.parquet',
        '300': 'stock_index_weight.parquet',
        '500': 'stock_index_weight.parquet',
        '800': 'stock_index_weight.parquet',
        '1000': 'stock_index_weight.parquet',
        '1800': 'stock_index_weight.parquet',
        '2200': 'stock_index_weight.parquet',
        '3000': 'stock_index_weight.parquet',
        'HighBeta800': 'beta_pool.parquet', 'LowBeta800': 'beta_pool.parquet',
        'HighBeta1000': 'beta_pool.parquet', 'LowBeta1000': 'beta_pool.parquet',
        'HighBeta1800': 'beta_pool.parquet', 'LowBeta1800': 'beta_pool.parquet',
        'HighBeta3000': 'beta_pool.parquet', 'LowBeta3000': 'beta_pool.parquet',
        'BigValue': 'bmsgv_pool.parquet', 'MedValue': 'bmsgv_pool.parquet', 'SmallValue': 'bmsgv_pool.parquet',
        'BigGrowth': 'bmsgv_pool.parquet', 'MedGrowth': 'bmsgv_pool.parquet', 'SmallGrowth': 'bmsgv_pool.parquet',
        'Ind1': 'ind_pool.parquet', 'Ind2': 'ind_pool.parquet', 'Ind3': 'ind_pool.parquet',
        'Ind4': 'ind_pool.parquet', 'Ind5': 'ind_pool.parquet', 'Ind6': 'ind_pool.parquet'
    }

    if stk_range in pool_file_map:
        path = os.path.join(input_dir, pool_file_map[stk_range])
        
        cols = ['TradingDay', 'SecuCode']
        if 'stock_index_weight' in path:
            if stk_range == '300': cols.append('IndexW300')
            elif stk_range == '500': cols.append('IndexW500')
            elif stk_range == '800': cols.extend(['IndexW300', 'IndexW500'])
            elif stk_range == '1800': cols.extend(['IndexW300', 'IndexW500', 'IndexW1000'])
            elif stk_range == '2200': cols.extend(['IndexW300', 'IndexW500', 'ID3000'])
            elif stk_range == '3000': cols.append('IndexW300')
        else:
            if 'ind_pool' in path:
                cols.append('Industry')
            else:
                cols.append(stk_range)

        df = pd.read_parquet(path, columns=cols)
        df = df[(df['TradingDay'] >= start_date) & (df['TradingDay'] <= end_date)]

        if stk_range == 'all':
            return df[['TradingDay', 'SecuCode']]
        elif stk_range == '300':
            return df.loc[df['IndexW300'] > 0, ['TradingDay', 'SecuCode']]
        elif stk_range == '500':
            return df.loc[df['IndexW500'] > 0, ['TradingDay', 'SecuCode']]
        elif stk_range == '800':
            return df.loc[(df['IndexW300'] > 0) | (df['IndexW500'] > 0), ['TradingDay', 'SecuCode']]
        elif stk_range == '1800':
            return df.loc[(df['IndexW300'] > 0) | (df['IndexW500'] > 0) | (df['IndexW1000'] > 0), ['TradingDay', 'SecuCode']]
        elif stk_range == '2200':
            return df.loc[(df['IndexW300'] == 0) & (df['IndexW500'] == 0) & (df['ID3000'] > 0), ['TradingDay', 'SecuCode']]
        elif stk_range.startswith('Ind'):
            ind_num = int(stk_range[-1])
            return df.loc[df['Industry'] == ind_num, ['TradingDay', 'SecuCode']]
        else:
            return df.loc[df[stk_range] > 0, ['TradingDay', 'SecuCode']]

    else:
        pool_path = f'./{stk_range}.csv'
        if os.path.exists(pool_path):
            pool = pd.read_csv(pool_path, header=None)
            pool.columns = ['TradingDay', 'SecuCode', stk_range]
            pool['TradingDay'] = pd.to_datetime(pool['TradingDay'])
            pool = pool.loc[(pool['TradingDay'] >= start_date) & (pool['TradingDay'] <= end_date)]
            return pool.loc[pool[stk_range] > 0, ['TradingDay', 'SecuCode']]
        else:
            raise ValueError(f'无法识别的股票池参数: {stk_range}')

def df2dict(valid_pool):
    valid_pool_dict = {}
    years = valid_pool['TradingDay'].dt.year.unique()
    for year in years:
        valid_pool_dict[year] = valid_pool[valid_pool['TradingDay'].dt.year == year]
    return valid_pool_dict

def filter_period(df, abn_dates_test):
    if abn_dates_test is None:
        return df
    
    if isinstance(abn_dates_test, str):
        if abn_dates_test == 'rise':
            intervals = [
                Interval(pd.Timestamp('2010-09-30'), pd.Timestamp('2010-10-15'), closed='both'),
                Interval(pd.Timestamp('2012-12-04'), pd.Timestamp('2012-12-14'), closed='both'),
                Interval(pd.Timestamp('2015-03-10'), pd.Timestamp('2015-06-12'), closed='both'),
                Interval(pd.Timestamp('2018-02-12'), pd.Timestamp('2018-02-26'), closed='both'),
                Interval(pd.Timestamp('2019-02-01'), pd.Timestamp('2019-02-25'), closed='both'),
                Interval(pd.Timestamp('2020-02-04'), pd.Timestamp('2020-02-20'), closed='both'),
                Interval(pd.Timestamp('2020-06-30'), pd.Timestamp('2020-07-09'), closed='both'),
                Interval(pd.Timestamp('2021-02-08'), pd.Timestamp('2021-02-10'), closed='both'),
                Interval(pd.Timestamp('2021-07-29'), pd.Timestamp('2021-08-04'), closed='both'),
                Interval(pd.Timestamp('2024-09-24'), pd.Timestamp('2024-10-08'), closed='both'),
            ]
        elif abn_dates_test == 'V':
            intervals = [
                Interval(pd.Timestamp('2011-12-02'), pd.Timestamp('2012-01-10'), closed='both'),
                Interval(pd.Timestamp('2012-03-28'), pd.Timestamp('2012-04-20'), closed='both'),
                Interval(pd.Timestamp('2015-07-01'), pd.Timestamp('2015-07-23'), closed='both'),
                Interval(pd.Timestamp('2016-01-20'), pd.Timestamp('2016-02-22'), closed='both'),
                Interval(pd.Timestamp('2018-02-06'), pd.Timestamp('2018-02-26'), closed='both'),
                Interval(pd.Timestamp('2020-01-21'), pd.Timestamp('2020-02-20'), closed='both'),
                Interval(pd.Timestamp('2021-07-23'), pd.Timestamp('2021-08-04'), closed='both'),
                Interval(pd.Timestamp('2022-03-02'), pd.Timestamp('2022-03-18'), closed='both'),
                Interval(pd.Timestamp('2022-04-20'), pd.Timestamp('2022-04-29'), closed='both'),
                Interval(pd.Timestamp('2023-10-13'), pd.Timestamp('2023-10-30'), closed='both'),
                Interval(pd.Timestamp('2024-01-26'), pd.Timestamp('2024-02-08'), closed='both'),
                Interval(pd.Timestamp('2024-10-09'), pd.Timestamp('2024-11-07'), closed='both')
            ]
        else:
            return df
            
        mask = False
        for interval in intervals:
            mask = mask | ((df['TradingDay'] >= interval.left) & (df['TradingDay'] <= interval.right))
        return df[mask]

    elif isinstance(abn_dates_test, dict):
        intervals = next(iter(abn_dates_test.values()))
        if not pd.api.types.is_datetime64_any_dtype(intervals['TradingDay']):
            if isinstance(intervals['TradingDay'].iloc[0], (int, np.int64)):
                intervals['TradingDay'] = pd.to_datetime(intervals['TradingDay'].astype(str), format='%Y%m%d')
            else:
                intervals['TradingDay'] = pd.to_datetime(intervals['TradingDay'])
        
        merge_df = pd.merge(df, intervals, on=['TradingDay'], how='inner')
        if 'valid' in merge_df.columns:
            merge_df = merge_df[merge_df['valid']==1].drop(columns=['valid'])
        return merge_df
    
    return df

def filter_by_month_status(df, month_status, dir_path):
    if month_status is None:
        return df 
    if not os.path.exists(dir_path):
        return df
    status_df = pd.read_excel(dir_path)
    selected_months = status_df[status_df['state'].isin(month_status)]['month'].astype(str).tolist()
    df['month'] = df['TradingDay'].dt.strftime('%Y%m')
    filtered_df = df[df['month'].isin(selected_months)].copy()
    filtered_df.drop('month', axis=1, inplace=True)
    return filtered_df

def cal_icir_turnover(df, factor_cols, rankic, output_ic):
    df = df.sort_values(by=['TradingDay', 'SecuCode']).copy()
    icir_dict = {}
    icts_dict = {}
    
    method = 'spearman' if rankic else 'pearson'
    
    for col in factor_cols:
        ic = df.groupby('TradingDay').apply(lambda g: g[col].corr(g['next_return'], method=method))
        icts_dict[col] = ic
        valid_ic = ic.dropna()
        
        df_wide = df.pivot(index='TradingDay', columns='SecuCode', values=col)
        corr = df_wide.corrwith(df_wide.shift(1), axis=1, method='spearman').mean()
        turnover = (1 - corr) / 2
        
        if len(valid_ic) < 2:
            icir_dict[col] = {'IC_Mean': np.nan, 'IC_Std': np.nan, 'ICIR': np.nan, 'TO': turnover,
                              'T-stat': np.nan, 'P-value': np.nan, 'NW_T-stat': np.nan, 'NW_P-value': np.nan}
        else:
            mean = valid_ic.mean()
            std = valid_ic.std()
            icir = mean / std if std != 0 else np.nan
            t_stat, p_value = stats.ttest_1samp(valid_ic, 0.0, nan_policy='omit')
            nw_t_stat, nw_p_value = get_newey_west_tstat(valid_ic)
            icir_dict[col] = {
                'IC_Mean': mean, 'IC_Std': std, 'ICIR': icir, 'TO': turnover,
                'T-stat': t_stat, 'P-value': p_value, 
                'NW_T-stat': nw_t_stat, 'NW_P-value': nw_p_value
            }
            
    results = [icir_dict, icts_dict] if output_ic else [icir_dict]
    return results

def cal_groupret_dict(data, factors, abn_dates_test):
    groupret_dict = {}
    groupret_ts_dict = {}
    for factor_name in factors:
        try:
            df_subset = data[['TradingDay', factor_name, 'next_return']].dropna(subset=[factor_name, 'next_return'])
            if df_subset.empty: continue

            factor_values = df_subset.groupby('TradingDay')[factor_name].transform(
                lambda x: pd.qcut(x, q=10, labels=False, duplicates='drop')
            )
            
            group_returns = df_subset.groupby(['TradingDay', factor_values])['next_return'].mean().unstack()
            
            if abn_dates_test == 'rise':
                daily_means = df_subset.groupby('TradingDay')['next_return'].mean()
                group_returns = group_returns.sub(daily_means, axis=0) 
            
            group_returns.columns = [f"{factor_name}_G{int(c)+1}" for c in group_returns.columns]
            groupret_ts_dict[factor_name] = group_returns

            groupret = group_returns.mean()
            groupret.index = [int(c.split('_G')[-1]) for c in groupret.index]
            groupret_dict[factor_name] = groupret
        except Exception:
            continue
    return groupret_dict, groupret_ts_dict

# ==============================================================================
#                                 核心计算流程
# ==============================================================================

def process_single_factor_group(name, input_dir, start_date_str, end_date, ret_df, abn_dates, 
                                month_status, status_path, group_ret, valid_pool_dict, sw1_dict, 
                                icir_path, rankic, output_ic, pm: PathManager):
    
    start_date = f"{start_date_str[:4]}-{start_date_str[4:6]}-{start_date_str[6:]}"
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    
    first_file = os.path.join(input_dir, str(start_year), f'Factors_{name}_all.parquet')
    if not os.path.exists(first_file): return

    try:
        columns = pq.ParquetFile(first_file).schema.names
    except: return

    to_remove = ['TradingDay', 'SecuCode', 'ClosePrice']
    factor_columns = [col for col in columns if col not in to_remove]
    if 'ClosePrice' in columns: factor_columns.append('ClosePriceRaw')
    
    batch_size = 10
    total_batches = (len(factor_columns) + batch_size - 1) // batch_size
    
    all_icir = {}
    all_groupret = {}
    all_icts = {}
    all_groupret_ts = []

    # [已恢复 tqdm] 方便单线程下监控进度
    for batch_idx in tqdm(range(total_batches), desc=f"计算因子 {name}", leave=False):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(factor_columns))
        batch_cols = factor_columns[batch_start:batch_end]
        read_cols = ['TradingDay', 'SecuCode'] + batch_cols
        
        df_list = []
        for year in range(start_year, end_year + 1):
            f_path = os.path.join(input_dir, str(year), f'Factors_{name}_all.parquet')
            if not os.path.exists(f_path): continue
            
            # 读取因子
            df = pd.read_parquet(f_path, columns=read_cols)
            
            # 股票池过滤
            if year in valid_pool_dict:
                df = pd.merge(valid_pool_dict[year], df, on=['TradingDay', 'SecuCode'], how='inner')
            else: continue

            # 行业中性化
            if sw1_dict and year in sw1_dict:
                df = pd.merge(df, sw1_dict[year], on=['TradingDay', 'SecuCode'], how='inner')
                mean_grouped = df.groupby(['TradingDay', 'SW1Code'])[batch_cols].transform('mean')
                std_grouped = df.groupby(['TradingDay', 'SW1Code'])[batch_cols].transform('std')
                mean_day = df.groupby('TradingDay')[batch_cols].transform('mean')
                std_day = df.groupby('TradingDay')[batch_cols].transform('std')
                mask = df.groupby(['TradingDay', 'SW1Code'])[batch_cols[0]].transform('size') == 1
                mean_combined = mean_grouped.where(~mask, mean_day)
                std_combined = std_grouped.where(~mask, std_day)
                df[batch_cols] = (df[batch_cols] - mean_combined) / std_combined
                df[batch_cols] = df[batch_cols].clip(-3, 3)
                df.drop(columns=['SW1Code'], inplace=True)
            
            df_list.append(df)
            
        if not df_list: continue
        df = pd.concat(df_list, ignore_index=True)
        del df_list
        
        # 筛选
        df = df[(df['TradingDay'] >= start_date) & (df['TradingDay'] <= end_date)].sort_values(["SecuCode", "TradingDay"])
        df = filter_period(df, abn_dates)
        df = filter_by_month_status(df, month_status, status_path)
        
        # [性能关键] 这里 ret_df 已经被裁切过，所以 right join 不会引入多余日期
        df = pd.merge(df, ret_df, on=['TradingDay', 'SecuCode'], how='right')
        
        if group_ret:
            g_dict, g_ts_dict = cal_groupret_dict(df, batch_cols, abn_dates)
            all_groupret.update(g_dict)
            if g_ts_dict:
                try: all_groupret_ts.append(pd.concat(g_ts_dict.values(), axis=1))
                except: pass

        results = cal_icir_turnover(df, batch_cols, rankic, output_ic)
        all_icir.update(results[0])
        if output_ic: all_icts.update(results[1])
        
        del df
        gc.collect()

    icir_df = pd.DataFrame.from_dict(all_icir, orient='index')
    if not icir_df.empty:
        icir_df = icir_df.round(4)
        icir_df.index.name = 'Column'
        icir_df.to_csv(icir_path, encoding='gbk')

    if group_ret and all_groupret:
        g_path = pm.get_groupret_path(icir_path)
        g_df = pd.DataFrame(all_groupret)
        g_df.index.name = 'Group'
        g_df.to_csv(g_path, encoding='gbk')
        
        if all_groupret_ts:
            g_ts_path = pm.get_groupret_ts_path(g_path)
            pd.concat(all_groupret_ts, axis=1).to_csv(g_ts_path, encoding='gbk')

    if output_ic and all_icts:
        icts_path = pm.get_icts_path(icir_path, rankic)
        pd.DataFrame(all_icts).to_csv(icts_path, encoding='gbk')


def run_calculation_task(pm: PathManager, name_list, input_dir, end_date, ret_idx, 
                         abn_dates, month_status, status_path, group_ret, ind_neu, rankic, output_ic):
    
    print("  [1/4] 加载股票池数据...")
    s_date_fmt = f"{pm.start_date_str[:4]}-{pm.start_date_str[4:6]}-{pm.start_date_str[6:]}"
    try:
        valid_pool = get_valid_pool(s_date_fmt, end_date, pm.pool, input_dir)
        valid_pool_dict = df2dict(valid_pool)
    except Exception as e:
        print(f"  [错误] 加载股票池失败: {e}")
        return
        
    print("  [2/4] 加载行业数据(中性化)...")
    sw1_dict = {}
    if ind_neu:
        
        if os.path.exists(os.path.join(input_dir, 'sw1.parquet')):
            sw1 = pd.read_parquet(os.path.join(input_dir, 'sw1.parquet'))
            sw1_dict = df2dict(sw1)
            del sw1

    abn_dates_dict = abn_dates
    if isinstance(abn_dates, str) and abn_dates not in ['rise', 'V']:
        if os.path.exists(abn_dates):
            intervals = pd.read_csv(abn_dates)
            abn_dates_dict = {abn_dates: intervals}

    print("  [3/4] 加载收益率数据...")
    ret_cols = ['TradingDay', 'SecuCode', 'ret_open5twap'] if ret_idx == 'Open5TWAP' else ['TradingDay', 'SecuCode', 'ret_c2c']
    ret_path = os.path.join(input_dir, 'ret_df.parquet')
    if not os.path.exists(ret_path):
         print(f"  [错误] 收益率文件缺失: {ret_path}")
         return
    
    # [性能关键修复] 提前对 ret_df 进行日期切片！
    ret_df = pd.read_parquet(ret_path, columns=ret_cols)
    ret_df.columns = ['TradingDay', 'SecuCode', 'next_return']
    ret_df = ret_df[(ret_df['TradingDay'] >= s_date_fmt) & (ret_df['TradingDay'] <= end_date)]

    print("  [4/4] 开始计算因子...")
    for name in name_list:
        icir_path = pm.get_icir_path(name, end_date, ret_idx, month_status, ind_neu, rankic)
        if os.path.exists(icir_path):
            print(f'  - 跳过已存在: {os.path.basename(icir_path)}')
            continue
        
        process_single_factor_group(name, input_dir, pm.start_date_str, end_date, ret_df, 
                                    abn_dates_dict, month_status, status_path, group_ret, 
                                    valid_pool_dict, sw1_dict, icir_path, rankic, output_ic, pm)


def auto_merge_results(pm: PathManager, end_date, ret_idx, month_status, ind_neu, rankic):
    print("正在合并计算结果...")
    base_pattern = pm.get_base_filename('*', end_date, ret_idx, month_status, ind_neu, rankic)
    search_path = os.path.join(pm.work_dir, base_pattern + '.csv')
    files = glob.glob(search_path)
    files = [f for f in files if 'Merged' not in os.path.basename(f) and 'Total' not in os.path.basename(f)]
    
    if not files: return

    dfs = []
    for f in files:
        try:
            temp = pd.read_csv(f, encoding='gbk', index_col=0)
            parts = os.path.basename(f).split('_')
            group_name = parts[2] if parts[1] == 'IndNeu' else parts[1]
            temp['FactorGroup'] = group_name
            dfs.append(temp)
        except: pass
            
    if dfs:
        merged = pd.concat(dfs, ignore_index=False)
        out_path = pm.get_merged_filename(end_date, ret_idx, month_status, ind_neu, rankic)
        merged.to_csv(out_path, encoding='gbk')
        print(f"合并完成: {os.path.basename(out_path)}")