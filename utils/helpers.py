# utils/helpers.py
import os
import pandas as pd
import numpy as np
import re

def is_valid_parquet_file(filepath):
    """检查是否为有效的 parquet 文件"""
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return False
    try:
        # 尝试读取 schema 以验证文件完整性，比读取全量数据快
        pd.read_parquet(filepath, engine='pyarrow')
        return True
    except Exception:
        return False

def natural_sort_key(s):
    '''自然排序键值生成 (例如: factor_2 排在 factor_10 前面)'''
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]