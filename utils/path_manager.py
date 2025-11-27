# utils/path_manager.py
import os

class PathManager:
    def __init__(self, root_output_dir, start_date, stock_pool, abn_dates_marker=None):
        """
        路径管理器初始化
        
        参数:
            root_output_dir: 结果输出根目录 (来自 settings.OUTPUT_DIR，即 './results')
            start_date: 回测开始日期
            stock_pool: 股票池名称
            abn_dates_marker: 异常日期标记
        """
        self.root = root_output_dir
        self.start_date_str = start_date.replace('-', '')
        self.pool = stock_pool
        self.abn_marker = self._parse_abn_marker(abn_dates_marker)
        
        # [修改点] 构建工作目录结构
        # 结构: FactorTestProject/results/Start20210101_PoolAll/
        
        folder_name = f"Start{self.start_date_str}_Pool{self.pool}"
        if self.abn_marker:
            folder_name += f"_{self.abn_marker}"
            
        # 直接拼接 root 和 folder_name，不再额外添加 'results' 中间层
        self.work_dir = os.path.join(self.root, folder_name)
        
        # 图片单独存放
        self.fig_dir = os.path.join(self.work_dir, 'figures') 
        
    def _parse_abn_marker(self, marker):
        """解析异常日期参数用于文件夹命名"""
        if marker in [None, [], '']:
            return None
        if isinstance(marker, str):
            if marker in ['rise', 'V']:
                return marker
            return os.path.splitext(os.path.basename(marker))[0]
        return "CustomScenario"

    def create_dirs(self):
        """创建所需的文件夹"""
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)
        return self.work_dir

    def get_base_filename(self, name, end_date, ret_idx, month_status, ind_neu, rankic):
        """生成标准化的基础文件名 (不含后缀)"""
        end_date_str = end_date.replace('-', '')
        ret_name = 'o5twap' if ret_idx == 'Open5TWAP' else 'c2c'
        
        ic_type = 'RankICIR' if rankic else 'ICIR'
        if ind_neu:
            ic_type += '_IndNeu'
            
        fname = f'{ic_type}_{name}_{self.pool}_{ret_name}_{self.start_date_str}-{end_date_str}'
        
        if self.abn_marker:
            fname += f'_{self.abn_marker}'
        
        if month_status:
            status_str = ','.join([str(x) for x in month_status])
            fname += f'_{status_str}'
            
        return fname

    def get_icir_path(self, name, end_date, ret_idx, month_status, ind_neu, rankic):
        fname = self.get_base_filename(name, end_date, ret_idx, month_status, ind_neu, rankic)
        return os.path.join(self.work_dir, fname + '.csv')

    def get_groupret_path(self, icir_path):
        """根据ICIR路径推导分组收益路径"""
        base = os.path.basename(icir_path)
        if 'RankICIR' in base:
            new_name = base.replace('RankICIR', 'GroupRet')
        else:
            new_name = base.replace('ICIR', 'GroupRet')
        return os.path.join(self.work_dir, new_name)

    def get_groupret_ts_path(self, groupret_path):
        """根据分组收益路径推导时间序列路径"""
        return groupret_path.replace('GroupRet', 'GroupRet_ts')
    
    def get_icts_path(self, icir_path, rankic):
        """根据ICIR路径推导IC序列路径"""
        target = 'RankIC_ts' if rankic else 'IC_ts'
        source = 'RankICIR' if rankic else 'ICIR'
        base = os.path.basename(icir_path)
        new_name = base.replace(source, target)
        return os.path.join(self.work_dir, new_name)

    def get_figure_path(self, factor_name, plot_type):
        """
        获取图片路径
        plot_type: 'returns' (柱状图) 或 'cumret' (折线图)
        """
        return os.path.join(self.fig_dir, f'{factor_name}_{plot_type}.png')

    def get_pdf_path(self, icir_path, report_type):
        """
        获取PDF报告路径
        """
        return icir_path.replace('.csv', f'_{report_type}.pdf')

    def get_merged_filename(self, end_date, ret_idx, month_status, ind_neu, rankic):
        """生成合并文件的名称"""
        base = self.get_base_filename('Merged', end_date, ret_idx, month_status, ind_neu, rankic)
        return os.path.join(self.work_dir, base + '.csv')