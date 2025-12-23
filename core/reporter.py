
import os
import pandas as pd
import numpy as np
import matplotlib
# 设置后端为 Agg，防止在无显示器的服务器上报错
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm

# ReportLab 用于生成 PDF
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# PyPDF2 用于合并 PDF
from PyPDF2 import PdfMerger

# 项目内部工具
from utils.path_manager import PathManager
from utils.helpers import natural_sort_key

def plot_returns_bar(data, factor, pm: PathManager):
    """
    绘制分组收益柱状图
    """
    if data is None:
        return
    
    # 确保索引是整数（分组编号）
    try:
        data.index = data.index.astype(int)
    except:
        pass

    plt.figure(figsize=(6, 4))
    plt.bar(data.index, data.values)
    plt.title(f'{factor} Factor Returns (Mean)')
    plt.xlabel('Quantile')
    plt.ylabel('Return')
    plt.tight_layout()
    
    # 获取保存路径
    save_path = pm.get_figure_path(factor, 'returns')
    plt.savefig(save_path)
    plt.close()

def plot_cumulative_returns_line(data_ts, factor, pm: PathManager):
    """
    绘制10分组及多空累计收益折线图
    """
    if data_ts is None or data_ts.empty:
        return
    
    # 移除全空的列
    data_ts = data_ts.dropna(axis=1, how='all')
    if data_ts.empty:
        return

    data_ts.index = pd.to_datetime(data_ts.index)
    data_ts = data_ts.sort_index()
    # 计算累计收益
    cumulative_returns = (1 + data_ts.fillna(0)).cumprod()
    
    # 设置图表尺寸
    fig, ax = plt.subplots(figsize=(11, 7)) 
    
    # 绘制各分组曲线
    for col in cumulative_returns.columns:
        try:
            # 假设列名格式为 FactorName_G1, 提取 G1
            group_num_str = col.split('_G')[-1]
            ax.plot(cumulative_returns.index, cumulative_returns[col], label=f'G{group_num_str}')
        except Exception:
            pass
    
    # 绘制多空曲线 (G10 - G1)
    try:
        g1_col = [c for c in data_ts.columns if c.endswith('_G1')]
        g10_col = [c for c in data_ts.columns if c.endswith('_G10')]
        if g1_col and g10_col:
            g1_col = g1_col[0]
            g10_col = g10_col[0]
            long_short_ret = data_ts[g10_col] - data_ts[g1_col]
            long_short_cumret = (1 + long_short_ret.fillna(0)).cumprod()
            long_short_cumret.name = "G10-G1"
            ax.plot(long_short_cumret.index, long_short_cumret, label='G10-G1', color='red', linestyle='--', linewidth=2)
    except Exception as e:
        print(f"绘制多空曲线失败 {factor}: {e}")

    ax.set_title(f'{factor} Cumulative Returns by Quantile')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    
    # ---------------- [修改点 1] 图例移至左侧外部 ----------------
    # bbox_to_anchor=(-0.15, 1): 相对于坐标轴左上角，向左偏移 0.15
    # loc='upper right': 图例框的右上角对齐到 anchor 点
    ax.legend(bbox_to_anchor=(-0.1, 1), loc='upper right', borderaxespad=0., fontsize='small')
    
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    
    # 调整布局：增大 left 参数，为左侧图例留出空间
    plt.subplots_adjust(left=0.20, right=0.91) 
    fig.autofmt_xdate() 
    
    # 获取保存路径
    save_path = pm.get_figure_path(factor, 'cumret')
    plt.savefig(save_path)
    plt.close()

def _internal_create_pdf(pdf_path, factor_data, pm: PathManager, title_str, report_type):
    """
    内部实际执行 PDF 生成的函数
    """
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    styles['Normal'].fontSize = 8

    # 标题
    story.append(Paragraph(title_str, styles['Title']))
    story.append(Spacer(1, 12))

    # 因子排序
    factors = sorted(list(factor_data.keys()), key=natural_sort_key)
    
    # 分页处理：每页 4 个因子 (3行 x 2列)
    for i in range(0, len(factors), 4):
        table_data = []
        # 每页处理 4 个，步长为 2 (一行两个)
        for j in range(i, min(i+4, len(factors)), 2):
            row = []
            for k in range(2):
                if j+k < len(factors):
                    factor = factors[j+k]
                    groupret, icir_stats = factor_data[factor]

                    # 提取统计指标
                    icir = icir_stats.get('ICIR', np.nan)
                    turnover = icir_stats.get('TO', np.nan)
                    nw_t_stat = icir_stats.get('NW_T-stat', np.nan)
                    nw_p_value = icir_stats.get('NW_P-value', np.nan)

                    # 单元格标题文本
                    title_text = (
                        f"<b>{factor}</b><br/>"
                        f"ICIR: {icir:.3f} | TO: {turnover:.3f}<br/>"
                        f"NW T-stat: {nw_t_stat:.3f} (p={nw_p_value:.3f})"
                    )
                    cell_content = [Paragraph(title_text, styles['Normal'])]

                    # 获取图片路径
                    img_bar_path = pm.get_figure_path(factor, 'returns')
                    img_line_path = pm.get_figure_path(factor, 'cumret')

                    # 根据报告类型添加图片
                    if report_type in ['Bar', 'Combined']:
                        if os.path.exists(img_bar_path):
                            cell_content.append(Image(img_bar_path, width=3.3*inch, height=2.2*inch))
                        else:
                            cell_content.append(Paragraph("缺失柱状图", styles['Normal']))
                    
                    # ---------------- [修改点 2] 增加 Combined 模式下的垂直间距 ----------------
                    if report_type == 'Combined':
                        cell_content.append(Spacer(1, 0.3 * inch)) # 增加 0.3 英寸的间距

                    if report_type in ['Line', 'Combined']:
                        if os.path.exists(img_line_path):
                            cell_content.append(Image(img_line_path, width=3.8*inch, height=2.8*inch))
                        else:
                            cell_content.append(Paragraph("缺失折线图", styles['Normal']))
                    
                    row.append(cell_content)
                else:
                    row.append('') # 占位空单元格
            table_data.append(row)
        
        # 计算行高
        if report_type == 'Combined':
            row_height = 6.0 * inch 
        else:
            row_height = 3.8 * inch 
            
        # 计算当前页实际行数
        current_page_factors = min(i+4, len(factors)) - i
        num_rows = int(np.ceil(current_page_factors / 2))
        
        table = Table(table_data, colWidths=[3.5*inch]*2, rowHeights=[row_height]*num_rows)
        table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        story.append(table)
        story.append(PageBreak())

    if not story or len(story) <= 2:
         story.append(Paragraph("无因子数据，无法生成报告。", styles['Normal']))
    
    try:
        doc.build(story)
        return True
    except Exception as e:
        print(f"生成PDF失败 {pdf_path}: {e}")
        return False

def gen_plot_pdf(name, pm: PathManager, end_date, ret_idx, month_status, ind_neu, rankic):
    """
    生成单个因子集合的所有图表和PDF
    """
    # 1. 获取必要的输入文件路径
    icir_path = pm.get_icir_path(name, end_date, ret_idx, month_status, ind_neu, rankic)
    
    if not os.path.exists(icir_path):
        print(f"错误: 找不到ICIR结果文件，跳过绘图: {icir_path}")
        return [], [], []

    # 推导其他文件路径
    groupret_path = pm.get_groupret_path(icir_path)
    groupret_ts_path = pm.get_groupret_ts_path(groupret_path)
    
    # 2. 读取数据
    try:
        icir_df = pd.read_csv(icir_path, encoding='gbk', index_col=0)
        
        if os.path.exists(groupret_path):
            groupret_df = pd.read_csv(groupret_path, encoding='gbk')
        else:
            groupret_df = pd.DataFrame() 

        if os.path.exists(groupret_ts_path):
            groupret_ts_df = pd.read_csv(groupret_ts_path, encoding='gbk', index_col=0)
            groupret_ts_df.index = pd.to_datetime(groupret_ts_df.index)
        else:
            groupret_ts_df = pd.DataFrame()
            
    except Exception as e:
        print(f"读取数据文件失败 {name}: {e}")
        return [], [], []

    all_factors = icir_df.index.values
    icir_stats_dict = icir_df.to_dict('index')
    factor_data = {} 

    # 3. 循环生成图片
    for factor in tqdm(all_factors, desc=f"正在绘制图表: {name}", leave=False):
        # 处理柱状图数据
        groupret_series = None
        if factor in groupret_df.columns:
            groupret_series = groupret_df[factor]
            # 去均值用于绘图展示
            groupret_plot = groupret_series - groupret_series.mean()
            if groupret_series.notna().any():
                plot_returns_bar(groupret_plot, factor, pm)
        
        # 处理折线图数据
        if not groupret_ts_df.empty:
            factor_ts_cols = [col for col in groupret_ts_df.columns if col.startswith(f"{factor}_G")]
            if factor_ts_cols:
                factor_ts_data = groupret_ts_df[factor_ts_cols]
                plot_cumulative_returns_line(factor_ts_data, factor, pm)
        
        # 收集数据用于 PDF 生成
        if groupret_series is not None:
            factor_data[factor] = (groupret_series, icir_stats_dict.get(factor, {}))

    # 4. 生成三种类型的 PDF
    generated_pdfs = {'Bar': None, 'Line': None, 'Combined': None}
    
    for r_type in ['Bar', 'Line', 'Combined']:
        pdf_path = pm.get_pdf_path(icir_path, r_type)
        
        # 使用 pm.pool 替代 stk_range
        title = f"{name} Report ({pm.pool}) - {r_type} Charts"
        
        success = _internal_create_pdf(pdf_path, factor_data, pm, title, r_type)
        if success:
            generated_pdfs[r_type] = pdf_path

    return generated_pdfs['Bar'], generated_pdfs['Line'], generated_pdfs['Combined']

def make_pdf_task(pm: PathManager, name_list, end_date, ret_idx, month_status, ind_neu, rankic):
    """
    主任务：遍历所有因子组，生成图表和PDF，并合并成总报告
    """
    all_bar_pdfs = []
    all_line_pdfs = []
    all_combined_pdfs = []

    # 1. 遍历生成单个 PDF
    for name in name_list:
        icir_path = pm.get_icir_path(name, end_date, ret_idx, month_status, ind_neu, rankic)
        pdf_bar = pm.get_pdf_path(icir_path, 'Bar')
        pdf_line = pm.get_pdf_path(icir_path, 'Line')
        pdf_comb = pm.get_pdf_path(icir_path, 'Combined')
        
        if os.path.exists(pdf_bar) and os.path.exists(pdf_line) and os.path.exists(pdf_comb):
            print(f"检测到 {name} 的 PDF 报告已存在，跳过生成。")
            all_bar_pdfs.append(pdf_bar)
            all_line_pdfs.append(pdf_line)
            all_combined_pdfs.append(pdf_comb)
        else:
            p_bar, p_line, p_comb = gen_plot_pdf(name, pm, end_date, ret_idx, month_status, ind_neu, rankic)
            if p_bar: all_bar_pdfs.append(p_bar)
            if p_line: all_line_pdfs.append(p_line)
            if p_comb: all_combined_pdfs.append(p_comb)

    # 2. 定义合并函数
    def merge_pdfs(pdf_list, suffix_type):
        if not pdf_list:
            print(f"警告: 没有生成 {suffix_type} 类型的 PDF，无法合并。")
            return

        print(f"正在合并 {suffix_type} PDF 报告...")
        merger = PdfMerger()
        for pdf in pdf_list:
            if os.path.exists(pdf):
                try:
                    merger.append(pdf)
                except Exception as e:
                    print(f"合并文件失败 {pdf}: {e}")
        
        merged_csv_name = pm.get_merged_filename(end_date, ret_idx, month_status, ind_neu, rankic)
        merged_pdf_path = merged_csv_name.replace('.csv', f'_{suffix_type}.pdf')
        
        if 'Merged' in os.path.basename(merged_pdf_path):
             merged_pdf_path = merged_pdf_path.replace('Merged', 'Total')

        try:
            merger.write(merged_pdf_path)
            merger.close()
            print(f"合并完成: {os.path.basename(merged_pdf_path)}")
        except Exception as e:
            print(f"保存合并PDF失败: {e}")

    # 3. 执行合并
    merge_pdfs(all_bar_pdfs, 'Bar')
    merge_pdfs(all_line_pdfs, 'Line')
    merge_pdfs(all_combined_pdfs, 'Combined')