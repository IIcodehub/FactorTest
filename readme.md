
# 📈 FactorTestProject - 模块化量化因子回测框架

[](https://www.python.org/)
[](https://www.google.com/search?q=)
[](https://www.google.com/search?q=)

**FactorTestProject** 是一个基于 Python 开发的轻量级、高性能量化因子回测框架。它专为量化研究员和初学者设计，旨在提供从**数据清洗**、**因子计算**、**IC/ICIR 分析**到**可视化报告生成**的一站式解决方案。

本项目采用清晰的工程化模块设计，支持多线程并行计算，并针对大量数据的 Pandas 操作进行了内存与 I/O 优化。

-----

## ✨ 核心特性

  * **⚡ 高性能回测**：针对 `groupby` 和 `merge` 操作深度优化，支持大规模因子数据的高效计算。
  * **🧩 模块化架构**：配置、数据、计算、报表完全分离，代码逻辑清晰，易于扩展和维护。
  * **📊 自动化研报**：一键生成包含 IC 时序图、分组累计收益图、多空净值曲线的 PDF 研报。
  * **🛠 自定义因子**：支持读取外部 Parquet/CSV 格式的自定义因子文件，无需依赖特定数据库环境。
  * **🗂 智能路径管理**：内置 `PathManager`，根据回测参数自动规划输出目录，从此告别文件混乱。
  * **🚀 并行加速**：支持多线程并发测试不同的起始日期、股票池或异常情景。

-----

## 📂 项目目录结构

```text
FactorTestProject/
│
├── config/
│   ├── __init__.py
│   └── settings.py          # [控制台] 全局参数配置文件 (日期、股票池、因子路径等)
│
├── core/
│   ├── __init__.py
│   ├── data_engine.py       # [数据层] 负责数据库交互、数据清洗、Parquet切分
│   ├── calculator.py        # [计算层] 核心回测逻辑 (IC计算, Newey-West调整, 分组收益)
│   └── reporter.py          # [表现层] Matplotlib 绘图引擎与 ReportLab PDF 生成器
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py           # [工具箱] 通用辅助函数
│   └── path_manager.py      # [管家] 统一管理文件路径命名与目录创建
│
├── results/                 # [输出] 回测结果自动保存在此处 (按任务分类)
│
├── main.py                  # [入口] 程序主入口，负责任务调度
└── requirements.txt         # 项目依赖库列表
```

-----

## 🧮 核心算法与数学原理

本框架在 `core/calculator.py` 中实现了标准的单因子测试指标。以下是回测逻辑背后的数学细节：

### 1\. IC (Information Coefficient, 信息系数)

IC 用于衡量因子值与下期收益率的线性相关程度，反映因子的预测能力。

  * **Normal IC (Pearson)**:
    $$IC_t = \frac{\text{Cov}(F_t, R_{t+1})}{\sigma_{F_t} \sigma_{R_{t+1}}}$$
    其中 $F_t$ 为 $t$ 时刻的因子值，$R_{t+1}$ 为 $t+1$ 时刻的股票收益率。

  * **Rank IC (Spearman)**: 若配置 `RANKIC = True`，则先对因子值和收益率进行排序（转为秩），再计算相关系数。这能消除异常值影响并捕捉非线性关系。
    $$\text{RankIC}_t = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$
    *(其中 $d_i$ 为个股因子排名与收益排名的差值，$n$ 为截面股票数量)*

### 2\. ICIR (Information Ratio, 信息比率)

衡量 IC 的稳定性，即单位风险下的因子预测能力。

$$ICIR = \frac{\overline{IC}}{\sigma_{IC}} \times \sqrt{N}$$

  * $\overline{IC}$: IC 序列的时间序列均值
  * $\sigma_{IC}$: IC 序列的时间序列标准差
  * *(注：本框架输出的 ICIR 通常为年化前的原始比率，可视需要乘以 $\sqrt{252}$)*

### 3\. 换手率代理指标 (Turnover Proxy)

本项目采用**因子自相关性**来估算因子的换手率。因子越稳定（自相关性越高），持仓换手率越低。

$$\text{AutoCorr}_t = \text{RankCorr}(F_t, F_{t-1})$$
$$\text{Turnover} \approx \frac{1 - \overline{\text{AutoCorr}}}{2}$$

  * **含义**：若因子排名完全不变 ($\text{AutoCorr}=1$)，理论换手率为 0；若因子排名完全随机 ($\text{AutoCorr} \approx 0$)，理论换手率约为 50%。

### 4\. 显著性检验 (Newey-West T-statistic)

由于 IC 序列通常存在**自相关性 (Autocorrelation)** 和 **异方差性 (Heteroscedasticity)**，普通的 T 检验会高估显著性。本项目采用 **Newey-West HAC (Heteroscedasticity and Autocorrelation Consistent)** 调整来计算更稳健的 T 统计量。

$$t_{NW} = \frac{\overline{IC}}{\hat{\sigma}_{HAC}}$$

  * **滞后阶数 (Lags) 选择**：
    代码根据样本量 $T$ 自动计算最佳滞后阶数 $L$：
    $$L = \text{int}\left(4 \times \left(\frac{T}{100}\right)^{\frac{2}{9}}\right)$$

### 5\. 分组收益 (Group Return)

每日将股票池按因子值从大到小分为 10 组（G1 为因子值最小，G10 为因子值最大）。

$$R_{g,t} = \frac{1}{N_g} \sum_{i \in Group_g} R_{i, t+1}$$

  * **多空收益 (Long-Short Return)**: $R_{G10} - R_{G1}$
  * **超额收益 (Excess Return)**：若设置了特定市场情景（如 `ABN_DATES_TEST = 'rise'`），代码会自动减去当日全市场均值：
    $$R_{g,t}^{excess} = R_{g,t} - R_{market, t}$$

-----

## 🚀 快速开始

### 1\. 环境准备

确保安装 Python 3.8+，并在项目根目录下运行：

```bash
pip install -r requirements.txt
```

### 2\. 配置回测参数

打开 `config/settings.py`，根据需求修改参数。该文件包含详细注释，核心参数如下：

```python
# === 模式选择 ===
MODE = 'test'  # 'test': 跑回测; 'save': 提取数据

# === 自定义因子 ===
# 如果你有自己的因子文件（Parquet/CSV），填在这里
FACTOR_ADD_PATH = r"./MyFactorData.parquet"
CUSTOM_FACTOR_NAME = 'MyAlpha01'

# === 回测参数 ===
START_DATES = ['2021-01-01']  # 开始日期
END_DATE = '2025-06-30'       # 结束日期
STOCK_POOLS = ['all']         # 股票池: 'all', '300', '500', 'HighBeta1000'
RET_IDX = 'Open5TWAP'         # 收益计算: 'Open5TWAP'(开盘均价) 或 'ClosePrice'
GROUP_RET = True              # 是否计算分组收益
RANKIC = True                 # 是否使用 RankIC
```

### 3\. 运行回测

```bash
python main.py
```

程序运行流程：

1.  **数据预处理**：自动读取自定义因子文件，按年份切分并转换为 Parquet 格式存储在 `data/` 目录下。
2.  **并行计算**：根据配置的日期和股票池，并行计算 ICIR 和分组收益。
3.  **生成报告**：自动绘制图表并生成 PDF。

-----

## 📊 输出结果说明

回测完成后，结果将保存在 `results/` 目录下，文件夹命名格式为：
`Start{开始日期}_Pool{股票池}_{场景}`

**文件夹内包含：**

| 文件名/文件夹 | 说明 |
| :--- | :--- |
| `figures/` | 存放所有生成的 PNG 图片（分组收益柱状图、累计净值曲线） |
| `ICIR_... .csv` | 因子的整体统计指标（IC均值, ICIR, 换手率, t-stat 等） |
| `GroupRet_... .csv` | 10个分组的平均收益率统计 |
| `GroupRet_ts_... .csv` | 分组收益率的每日时间序列数据（用于画图） |
| `IC_ts_... .csv` | 每日 IC 值的时间序列数据 |
| **`Merged_..._Combined.pdf`** | **最终汇总报告**，包含所有统计数据和图表 |

-----

## ⚙️ 进阶功能

### 1\. 行业中性化

在 `settings.py` 中设置 `IND_NEU = True`。

  * **逻辑**：在计算 IC 和分组前，对因子值进行申万一级行业 (SW1) 内的 Z-Score 标准化，并剔除 3 倍标准差之外的极值。
  * **公式**：$F_{neutral} = \frac{F_{raw} - \mu_{ind}}{\sigma_{ind}}$

### 2\. 异常情景测试

在 `settings.py` 中设置 `ABN_DATES_TEST`。

  * `'rise'`: 仅测试历史上市场暴涨的区间。
  * `'V'`: 仅测试市场深 V 反转的区间。
  * `['path/to/dates.csv']`: 传入自定义的日期列表文件，只在特定日期进行测试。

### 3\. 自定义股票池

除内置的 `300/500/800` 外，你可以提供一个包含 `TradingDay`, `SecuCode`, `Weight` 的 CSV 文件路径给 `STOCK_POOLS` 参数，框架会自动加载该文件作为股票池。

-----

## 🤝 贡献

欢迎提交 Issue 或 Pull Request！
如果你觉得这个项目对你有帮助，请给它一个 ⭐️ Star！

-----

