# core/data_engine.py
import pandas as pd
import pymssql
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
from utils.helpers import is_valid_parquet_file

# ==============================================================================
#                                 SQL 交互部分
# ==============================================================================

def read_sql(query, database='Quant'):
    # 请根据实际情况填写数据库连接信息
    conn = pymssql.connect('', '', '', database, tds_version='7.0')
    data = pd.read_sql(query, conn)
    conn.close()
    return data

def get_factors_query(factors, name, start_date, end_date, isint=0, schema='Factor'):
    '''读取factor中的某个特定表的因子数据'''
    factor = ', '.join([f'[{item}]' for item in factors])
    name = 'LHRM3' if name == 'Risk' else name
    if isint == 1:
        start_int = int(start_date.replace('-', ''))
        end_int = int(end_date.replace('-', ''))
        query = f"""
                SELECT TradingDayInt, CodeInt, {factor}
                FROM [{schema}].[{name}]
                WHERE TradingDayInt BETWEEN {start_int} AND {end_int}
                ORDER BY TradingDayInt ASC
                """
    else:
        query = f"""
                SELECT TradingDay, SecuCode, {factor}
                FROM [{schema}].[{name}]
                WHERE TradingDay BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY TradingDay ASC
                """
    return query

def get_pool_ret_query(start_date, end_date, name):
    start_int = int(start_date.replace('-', ''))
    end_int = int(end_date.replace('-', ''))
    
    queries = {
        'stock_index_weight': f"SELECT TradingDay, SecuCode, IndexW300, IndexW500, IndexW1000, ID3000 FROM [Factor].[StockIndexWeightNew2] WHERE TradingDay BETWEEN '{start_date}' AND '{end_date}' ORDER BY TradingDay ASC",
        'market_info': f"SELECT TradingDay, SecuCode, ClosePrice, PrevClosePrice, SW1Code FROM [Factor].[MarketInfoNew] WHERE TradingDay BETWEEN '{start_date}' AND '{end_date}' ORDER BY TradingDay ASC",
        'beta_pool': f"SELECT TradingDayInt, CodeInt, HighBeta1800, LowBeta1800, HighBeta800, LowBeta800, HighBeta1000, LowBeta1000, HighBeta3000, LowBeta3000 FROM [Factor].[StockPoolHLBetaAdj] WHERE TradingDayInt BETWEEN {start_int} AND {end_int} ORDER BY TradingDayInt ASC",
        'bmsgv_pool': f"SELECT TradingDayInt, CodeInt, BigValue, MedValue, SmallValue, BigGrowth, MedGrowth, SmallGrowth FROM [Factor].[StockPoolValueGrowthAdj] WHERE TradingDayInt BETWEEN {start_int} AND {end_int} ORDER BY TradingDayInt ASC",
        'ind_pool': f"SELECT TradingDayInt, CodeInt, Industry FROM [Factor].[StockPoolInd] WHERE TradingDayInt BETWEEN {start_int} AND {end_int} ORDER BY TradingDayInt ASC",
        'daily_mean_price': f"SELECT TradingDayInt, CodeInt, Open5TWAP FROM [Market].[DailyMeanPrice1] WHERE TradingDayInt BETWEEN {start_int} AND {end_int} ORDER BY TradingDayInt ASC"
    }
    
    if name not in queries:
        raise ValueError(f'{name}取值错误,无法生成查询语句!')
    return queries[name]

def transform_from_dayint_codeint(df):
    if 'TradingDayInt' in df.columns:
        df['TradingDay'] = pd.to_datetime(df['TradingDayInt'].astype(str), format='%Y%m%d')
        df = df.drop(columns=['TradingDayInt'])
    else:
        df['TradingDay'] = pd.to_datetime(df['TradingDay'], format='%Y%m%d')

    if 'CodeInt' in df.columns:
        df = df.rename(columns={'CodeInt': 'SecuCode'})
        df['SecuCode'] = df['SecuCode'].astype(str).str.zfill(6)
    return df

# ==============================================================================
#                                 数据更新逻辑
# ==============================================================================

def update_factor_data(name, year, start_date, end_date, output_dir, factor_dict_path):
    factor_dict = pd.read_excel(factor_dict_path)
    part =  factor_dict.loc[factor_dict['name'] == name]
    factors = part['factors'].tolist()
    database = part['database'].iloc[0]
    schema = part['schema'].iloc[0]
    isint = part['int'].iloc[0]
    year_dir = os.path.join(output_dir, str(year))
    file_path = os.path.join(year_dir, f'Factors_{name}_all.parquet')
    os.makedirs(year_dir, exist_ok=True)
    query = get_factors_query(factors, name, start_date, end_date, isint, schema)
    new_data = read_sql(query, database)
    new_data[factors] = new_data[factors].astype(float)
    new_data = transform_from_dayint_codeint(new_data)
    if 'ClosePrice' in new_data.columns:
        new_data = new_data.rename(columns={'ClosePrice': 'ClosePriceRaw'})
    if os.path.exists(file_path) and is_valid_parquet_file(file_path):
        old_data = pd.read_parquet(file_path)
        updated_data = pd.concat([old_data, new_data], ignore_index=True)
        updated_data = updated_data.drop_duplicates(subset=['TradingDay', 'SecuCode'], keep='last').sort_values(['TradingDay', 'SecuCode'])
        updated_data.to_parquet(file_path, index=False)
    else:
        new_data = new_data.drop_duplicates(subset=['TradingDay', 'SecuCode'], keep='last').sort_values(['TradingDay', 'SecuCode'])
        new_data.to_parquet(file_path, index=False)        

def update_pool_ret_data(start_date, end_date, output_dir):
    names = ['daily_mean_price', 'market_info', 'stock_index_weight', 'beta_pool', 'bmsgv_pool', 'ind_pool']
    for name in names:
        query = get_pool_ret_query(start_date, end_date, name)
        new_data = read_sql(query)
        new_data = transform_from_dayint_codeint(new_data)
        
        if name in ['beta_pool', 'bmsgv_pool', 'ind_pool', 'stock_index_weight']:
            file_path = os.path.join(output_dir, f'{name}.parquet')
            if os.path.exists(file_path) and is_valid_parquet_file(file_path):
                old_data = pd.read_parquet(file_path)
                updated_data = pd.concat([old_data, new_data], ignore_index=True)
                updated_data = updated_data.drop_duplicates(subset=['TradingDay', 'SecuCode'], keep='last').sort_values(['SecuCode', 'TradingDay'])
                updated_data.to_parquet(file_path, index=False)
            else:
                new_data = new_data.drop_duplicates(subset=['TradingDay', 'SecuCode'], keep='last').sort_values(['SecuCode', 'TradingDay'])
                new_data.to_parquet(file_path, index=False)
                
        elif name == 'market_info':
            # 导出申万行业
            sw1_path = os.path.join(output_dir, 'sw1.parquet')
            market_info = new_data
            new_sw1 = market_info[['TradingDay', 'SecuCode', 'SW1Code']].copy()
            if os.path.exists(sw1_path):
                old_sw1 = pd.read_parquet(sw1_path)
                sw1 = pd.concat([old_sw1, new_sw1], ignore_index=True)
                sw1 = sw1.drop_duplicates(subset=['TradingDay', 'SecuCode'], keep='last').sort_values(['SecuCode', 'TradingDay'])
                sw1.to_parquet(sw1_path, index=False)
            else:
                sw1 = new_sw1.drop_duplicates(subset=['TradingDay', 'SecuCode'], keep='last').sort_values(['SecuCode', 'TradingDay'])
                sw1.to_parquet(sw1_path, index=False)

            # 导出收益率数据
            query_dm = get_pool_ret_query(start_date, end_date, 'daily_mean_price')
            daily_mean = transform_from_dayint_codeint(read_sql(query_dm)).sort_values(['SecuCode', 'TradingDay'])
            
            ret_path = os.path.join(output_dir, 'ret_df.parquet')
            new_ret = pd.merge(market_info.drop(columns=['SW1Code']), daily_mean, on=['TradingDay', 'SecuCode'], how='inner')
            
            if os.path.exists(ret_path):
                old_ret = pd.read_parquet(ret_path).drop(columns=['ret_open5twap', 'ret_c2c'], errors='ignore')
                new_ret_df = pd.concat([old_ret, new_ret], ignore_index=True)
            else:
                new_ret_df = new_ret

            new_ret_df = new_ret_df.drop_duplicates(subset=['TradingDay', 'SecuCode'], keep='last').sort_values(['SecuCode', 'TradingDay'])
            
            shifted_open5twap = new_ret_df.groupby('SecuCode')['Open5TWAP'].shift(1)
            shifted_close_price = new_ret_df.groupby('SecuCode')['ClosePrice'].shift(1)
            
            new_ret_df['ret_open5twap'] = ((new_ret_df['Open5TWAP'] / shifted_open5twap * shifted_close_price / new_ret_df['PrevClosePrice']) - 1).groupby(new_ret_df['SecuCode']).shift(-2)
            new_ret_df['ret_c2c'] = (new_ret_df['ClosePrice'] / new_ret_df['PrevClosePrice'] - 1).groupby(new_ret_df['SecuCode']).shift(-1)
            new_ret_df.to_parquet(ret_path, index=False)

def run_data_storage(factor_only, start_date, end_date, output_dir, factor_dict_path, name_list):
    """数据存储主函数"""
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    if factor_only in ['pools', 'all']:
        print("正在更新股票池和收益率数据...")
        update_pool_ret_data(start_date, end_date, output_dir)
    
    if factor_only in ['factors', 'all']:
        current_date_dt = start_date_dt
        while current_date_dt < end_date_dt:
            year = current_date_dt.year
            year_start = (max(current_date_dt, datetime(year, 1, 1))).strftime("%Y-%m-%d")
            year_end = (min(end_date_dt, datetime(year, 12, 31))).strftime("%Y-%m-%d")
            
            for name in tqdm(name_list, desc=f'更新{year}年因子数据', leave=True):
                update_factor_data(name, year, year_start, year_end, output_dir, factor_dict_path)
            
            current_date_dt = datetime(year + 1, 1, 1)

# ==============================================================================
#                                 自定义因子处理逻辑 (新)
# ==============================================================================

def process_custom_factor(file_path, output_dir, start_date, end_date, custom_name):
    """
    处理自定义因子文件：读取、清洗、按年切分并保存
    参数:
        file_path: 原始大文件路径
        output_dir: 结果存储根目录 (FACTOR_DIR)
        start_date: 开始处理日期
        end_date: 结束处理日期
        custom_name: 系统内部名称 (settings.CUSTOM_FACTOR_NAME)
    """
    print(f"正在处理自定义因子文件: {file_path}")
    
    # 1. 读取文件
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError("仅支持 .csv 或 .parquet 格式")

    # 2. 列名映射与清洗
    col_map = {
        'TradingDayInt': 'TradingDay',
        'CodeInt': 'SecuCode'
    }
    df = df.rename(columns=col_map)
    
    if 'TradingDay' not in df.columns or 'SecuCode' not in df.columns:
        raise ValueError("自定义文件必须包含 TradingDay 和 SecuCode 列")

    # 3. 日期格式转换
    # 处理 20210101 这种整数或字符串
    first_date = df['TradingDay'].iloc[0]
    if isinstance(first_date, (int, np.integer)) or (isinstance(first_date, str) and len(first_date) == 8 and first_date.isdigit()):
        print("检测到非标准日期格式，正在转换...")
        df['TradingDay'] = pd.to_datetime(df['TradingDay'].astype(str), format='%Y%m%d', errors='coerce')
    else:
        df['TradingDay'] = pd.to_datetime(df['TradingDay'], errors='coerce')
    
    df = df.dropna(subset=['TradingDay'])
    df['SecuCode'] = df['SecuCode'].astype(str).str.zfill(6)
    
    # 4. 按年切分保存
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    
    print(f"正在将数据切分至 {start_year}-{end_year} 年份文件夹...")
    
    for year in range(start_year, end_year + 1):
        year_dir = os.path.join(output_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)
        
        # 目标文件名: Factors_{CUSTOM_FACTOR_NAME}_all.parquet
        target_file = os.path.join(year_dir, f'Factors_{custom_name}_all.parquet')
        
        # 筛选当年数据
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year}-12-31")
        
        mask = (df['TradingDay'] >= year_start) & (df['TradingDay'] <= year_end)
        df_year = df[mask].copy()
        
        if not df_year.empty:
            df_year.sort_values(['TradingDay', 'SecuCode'], inplace=True)
            df_year.to_parquet(target_file, index=False)
            print(f"  - {year}年数据已保存: {target_file} ({len(df_year)}条)")
        else:
            print(f"  - 警告: {year}年无数据")

    print("自定义因子数据预处理完成。")