# 时间序列特征提取说明文档

本文档详细说明了 `enhanced_metric_analyzer.py` 中提取的各类时间序列特征及其含义。特征提取主要包含三个部分：自定义曲线特征、自定义统计特征以及基于 `tsfresh` 的特征。

## 1. 自定义曲线特征 (Curve Features)

这部分特征主要用于检测时间序列中的形态变化和异常模式。

| 特征类型 | 具体模式 | 说明 |
| :--- | :--- | :--- |
| **突变 (Sudden Changes)** | `sudden_change` | 基于统计学断点检测 (Ruptures) 发现的均值突变点，包含突变幅度 (`change_magnitude`) 和方向 (`increase`/`decrease`)。 |
| | `derivative_spike` | 基于一阶导数发现的瞬时剧烈波动，通常对应数据中的跳变。 |
| **趋势变化 (Trend Changes)** | `trend_reversal` | 趋势反转，即斜率正负号发生改变（如由涨转跌）。 |
| | `acceleration` | 趋势加速，当前斜率绝对值显著大于前一阶段斜率（增加2倍以上）。 |
| | `deceleration` | 趋势减速，当前斜率绝对值显著小于前一阶段斜率（减少一半以上）。 |
| | `slope_change` | 显著的斜率变化，但未达到上述特定分类标准。 |
| **阈值违规 (Threshold Violations)** | `threshold_violation` | 数据超出预设或动态计算的警告/临界阈值（Upper/Lower Warning/Critical）。记录违规持续时间、最大违规值等。 |
| **尖峰与低谷 (Spikes & Dips)** | `spike` | 局部显著峰值，其高度显著超过周围数据（基于均值和标准差判定）。 |
| | `dip` | 局部显著低谷，其深度显著低于周围数据。 |

## 2. 自定义统计特征 (Statistical Features)

这部分特征用于描述数据的整体分布属性和波动情况。

| 特征键名 | 中文含义 | 说明 |
| :--- | :--- | :--- |
| `mean` | 均值 | 数据的平均水平。 |
| `std` | 标准差 | 数据的离散程度。 |
| `variance` | 方差 | 标准差的平方。 |
| `min` | 最小值 | 序列中的最小值。 |
| `max` | 最大值 | 序列中的最大值。 |
| `range` | 极差 | 最大值与最小值的差。 |
| `cv` | 变异系数 | 标准差除以均值，用于衡量相对波动性（归一化的波动率）。 |
| `skewness` | 偏度 | 衡量数据分布的不对称性。正值表示右偏（长尾在右），负值表示左偏。 |
| `kurtosis` | 峰度 | 衡量数据分布的陡峭程度或尾部厚度。高峰度表示有更多极端值。 |
| `mean_rate_of_change` | 平均变化率 | 一阶差分的均值，表示总体趋势方向。 |
| `std_rate_of_change` | 变化率标准差 | 一阶差分的标准差，表示变化的剧烈程度。 |
| `max_increase` | 最大增幅 | 单步最大的正向变化量。 |
| `max_decrease` | 最大降幅 | 单步最大的负向变化量。 |
| `smoothness` | 平滑度 | 基于二阶差分计算。值越接近1表示越平滑，越接近0表示越粗糙。 |
| `volatility_clustering` | 波动聚集度 | 滚动标准差的标准差，用于检测波动是否随时间成簇出现。 |

## 3. Tsfresh 特征 (Tsfresh Features)

使用 `tsfresh` 库的 `MinimalFCParameters` 配置提取的基础特征

| 特征键名 | 中文含义 | 说明 |
| :--- | :--- | :--- |
| `sum_values` | 总和 | 序列所有值的总和。 |
| `median` | 中位数 | 序列的中值，比均值更抗干扰。 |
| `mean` | 均值 | 同自定义统计特征中的均值。 |
| `length` | 长度 | 时间序列的数据点数量。 |
| `standard_deviation` | 标准差 | 同自定义统计特征中的标准差。 |
| `variance` | 方差 | 同自定义统计特征中的方差。 |
| `root_mean_square` | 均方根 (RMS) | 值的平方和除以数量后的平方根，衡量能量或幅度。 |
| `maximum` | 最大值 | 同自定义统计特征中的最大值。 |
| `absolute_maximum` | 绝对最大值 | 序列中绝对值的最大值。 |
| `minimum` | 最小值 | 同自定义统计特征中的最小值。 |

## 输出格式说明

目前代码分析结果的输出格式如下：

```text
[指标粒度_指标名][实例类型 实例名称][[特征:值]...]
```

例如：
```text
[apm_response_time][service payment-service][sum_values:15.22 median:0.20 mean:0.15 length:100.00 standard_deviation:0.67 variance:0.45 root_mean_square:0.69 maximum:1.21 absolute_maximum:1.21 minimum:-1.12]
```
