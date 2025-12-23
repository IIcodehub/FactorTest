# main.py
import time
import concurrent.futures
import config.settings as cfg
from utils.path_manager import PathManager
from core.data_engine import run_data_storage, process_custom_factor
from core.calculator import run_calculation_task, auto_merge_results
from core.reporter import make_pdf_task
import warnings
warnings.filterwarnings('ignore')

def run_test_pipeline(params):
    """
    单个测试任务的完整流水线
    params: (起始日期, 股票池, 异常情景)
    """
    start_date, pool, abn_condition = params
    
    # 1. 初始化路径管理器
    pm = PathManager(cfg.OUTPUT_DIR, start_date, pool, abn_condition)
    work_dir = pm.create_dirs()
    
    print(f"\n{'='*20} 开始新任务 {'='*20}")
    print(f"日期: {start_date} | 股票池: {pool} | 情景: {abn_condition}")
    print(f"输出目录: {work_dir}")
    
    t0 = time.time()

    # 2. 执行计算
    run_calculation_task(
        pm=pm,
        name_list=cfg.NAME_LIST,
        input_dir=cfg.FACTOR_DIR,
        end_date=cfg.END_DATE,
        ret_idx=cfg.RET_IDX,
        abn_dates=abn_condition,
        month_status=cfg.MONTH_STATUS,
        status_path=cfg.STATUS_DIR_PATH,
        group_ret=cfg.GROUP_RET,
        ind_neu=cfg.IND_NEU,
        rankic=cfg.RANKIC,
        output_ic=cfg.OUTPUT_IC
    )

    # 3. 结果合并
    auto_merge_results(pm, cfg.END_DATE, cfg.RET_IDX, cfg.MONTH_STATUS, cfg.IND_NEU, cfg.RANKIC)

    # 4. 生成报告
    if cfg.GROUP_RET:
        print("开始生成可视化报告...")
        make_pdf_task(pm, cfg.NAME_LIST, cfg.END_DATE, cfg.RET_IDX, cfg.MONTH_STATUS, cfg.IND_NEU, cfg.RANKIC)

    print(f"任务完成，耗时: {time.time() - t0:.2f}秒")
    return start_date, pool

def main():
    # ------------------ 模式一：数据提取 ------------------
    if cfg.MODE == 'save':
        print(">>> 进入数据提取模式 (SQL -> Local)")
        run_data_storage(cfg.FACTOR_ONLY, cfg.START_DATES[0], cfg.END_DATE, cfg.FACTOR_DIR, cfg.FACTOR_DICT_PATH, cfg.NAME_LIST)
        return

    # ------------------ 模式二：因子回测 ------------------
    if cfg.MODE == 'test':
        print(">>> 进入因子测试模式")
        
        # [关键修复] 处理自定义因子逻辑
        if cfg.FACTOR_ADD_PATH:
            print(f"检测到自定义因子文件配置: {cfg.FACTOR_ADD_PATH}")
            # 1. 调用数据引擎进行预处理 (切分文件)
            # 使用 settings 中配置的 CUSTOM_FACTOR_NAME 来命名文件
            process_custom_factor(
                file_path=cfg.FACTOR_ADD_PATH, 
                output_dir=cfg.FACTOR_DIR, 
                start_date=cfg.START_DATES[0], 
                end_date=cfg.END_DATE, 
                custom_name=cfg.CUSTOM_FACTOR_NAME
            )
            
            # 2. 强制覆盖待测试因子列表
            # 这样后续计算模块就会去寻找 Factors_{CUSTOM_FACTOR_NAME}_all.parquet
            print(f"已将测试目标锁定为自定义因子: {cfg.CUSTOM_FACTOR_NAME}")
            cfg.NAME_LIST = [cfg.CUSTOM_FACTOR_NAME]

        # 生成任务组合
        tasks = []
        for s_date in cfg.START_DATES:
            for pool in cfg.STOCK_POOLS:
                # 处理异常日期参数的多态性 (None, 字符串, 列表)
                conditions = []
                if isinstance(cfg.ABN_DATES_TEST, list):
                    conditions = cfg.ABN_DATES_TEST
                else:
                    conditions = [cfg.ABN_DATES_TEST]
                
                for cond in conditions:
                    tasks.append((s_date, pool, cond))

        print(f"共生成 {len(tasks)} 个测试任务，准备并行执行...")
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as executor:
            futures = [executor.submit(run_test_pipeline, t) for t in tasks]
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    print(f"任务反馈: {res} 已结束")
                except Exception as e:
                    print(f"任务执行异常: {e}")
                    import traceback
                    traceback.print_exc()

        print(f"\n全流程执行总耗时: {time.time() - start_time:.2f} 秒")

if __name__ == '__main__':
    main()