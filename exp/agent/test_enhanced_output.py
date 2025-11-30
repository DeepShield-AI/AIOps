import numpy as np
import pandas as pd
import sys
sys.path.append('/home/ubuntu/smore/aiops2025')

from exp.agent.enhanced_metric_analyzer import CurveFeatureExtractor, MetricAnalyzer

def generate_test_series():
    """生成测试时间序列"""
    np.random.seed(42)
    
    # 1. 尖峰异常
    spike_series = np.random.normal(100, 5, 200)
    spike_series[100:105] += 50  # 添加尖峰
    
    # 2. 水平位移异常
    shift_series = np.random.normal(100, 5, 200)
    shift_series[100:] += 40  # 水平位移
    
    # 3. 高波动异常
    volatile_series = np.random.normal(100, 5, 200)
    volatile_series[100:] += 25 * np.sin(np.linspace(0, 10*np.pi, 100))
    
    # 4. 趋势反转异常
    trend_series = np.random.normal(100, 5, 200)
    trend_series[:100] += np.linspace(0, 30, 100)
    trend_series[100:] += 30 - np.linspace(0, 40, 100)
    
    # 5. 正常序列
    normal_series = np.random.normal(100, 5, 200)
    normal_series += np.linspace(0, 2, 200)  # 轻微趋势
    
    return {
        'spike': spike_series,
        'shift': shift_series,
        'volatile': volatile_series,
        'trend': trend_series,
        'normal': normal_series
    }

def test_feature_extraction():
    """测试特征提取和格式化"""
    print("="*100)
    print("🔬 增强特征输出测试")
    print("="*100)
    
    # 创建分析器（模拟环境）
    analyzer = MetricAnalyzer.__new__(MetricAnalyzer)
    analyzer.feature_extractor = CurveFeatureExtractor(min_change_threshold=0.05)
    analyzer.min_change_threshold = 0.05
    
    # 生成测试数据
    test_data = generate_test_series()
    
    for anomaly_type, series_data in test_data.items():
        print(f"\n{'='*100}")
        print(f"📊 测试场景: {anomaly_type.upper()}")
        print(f"{'='*100}\n")
        
        # 转换为Series
        series = pd.Series(series_data)
        timestamps = pd.Series(range(len(series_data)))
        
        # 分析时间序列
        analysis = analyzer.analyze_time_series_features(
            series, 
            timestamps, 
            f'test_metric_{anomaly_type}', 
            f'test-service-{anomaly_type}', 
            'apm'
        )
        
        if analysis:
            # 格式化输出
            formatted = analyzer._format_result(analysis)
            print("📝 格式化输出:")
            print("-" * 100)
            print(formatted)
            print("-" * 100)
            
            # 详细特征展示
            print("\n📋 详细特征分解:")
            print(f"  • 指标粒度: {analysis['metric_granularity']}")
            print(f"  • 实例类型: {analysis['instance_type']}")
            print(f"  • 实例名称: {analysis['instance_name']}")
            print(f"  • 变化率: {analysis['change_rate']:.2f}%")
            print(f"  • 正常均值: {analysis['normal_mean']:.2f}")
            print(f"  • 异常均值: {analysis['anomalous_mean']:.2f}")
            
            print(f"\n  🎯 曲线模式特征:")
            print(f"    - 突变次数: {len(analysis['sudden_changes'])}")
            print(f"    - 趋势变化: {len(analysis['trend_changes'])}")
            print(f"    - 尖峰数量: {len([x for x in analysis['spikes_dips'] if x['type']=='spike'])}")
            print(f"    - 低谷数量: {len([x for x in analysis['spikes_dips'] if x['type']=='dip'])}")
            print(f"    - 阈值违规: {len(analysis['threshold_violations'])}")
            print(f"    - 描述: {analysis['curve_features']}")
            
            stat_feat = analysis.get('statistical_features', {})
            if stat_feat:
                print(f"\n  📊 统计特征:")
                print(f"    - 变异系数(CV): {stat_feat.get('cv', 0):.4f}")
                print(f"    - 偏度(Skewness): {stat_feat.get('skewness', 0):.4f}")
                print(f"    - 峰度(Kurtosis): {stat_feat.get('kurtosis', 0):.4f}")
                print(f"    - 平滑度(Smoothness): {stat_feat.get('smoothness', 0):.4f}")
                print(f"    - 波动聚集: {stat_feat.get('volatility_clustering', 0):.4f}")
            
            # 异常严重程度评估
            severity = assess_severity(analysis)
            print(f"\n  🚨 严重程度评估: {severity['level']} - {severity['description']}")
            print(f"    建议操作: {severity['action']}")
        else:
            print("✅ 未检测到显著异常（在正常范围内）")
    
    print("\n" + "="*100)
    print("✅ 测试完成")
    print("="*100)

def assess_severity(analysis):
    """评估异常严重程度"""
    stat_feat = analysis.get('statistical_features', {})
    cv = stat_feat.get('cv', 0)
    kurtosis = stat_feat.get('kurtosis', 0)
    smoothness = stat_feat.get('smoothness', 1.0)
    pattern_count = (len(analysis['sudden_changes']) + 
                     len(analysis['trend_changes']) + 
                     len(analysis['spikes_dips']))
    change_rate = abs(analysis['change_rate'])
    
    # 计算严重程度得分
    score = 0
    
    # 变化率权重
    if change_rate > 100:
        score += 3
    elif change_rate > 50:
        score += 2
    elif change_rate > 20:
        score += 1
    
    # 波动性权重
    if cv > 0.3:
        score += 3
    elif cv > 0.2:
        score += 2
    elif cv > 0.1:
        score += 1
    
    # 峰度权重
    if kurtosis > 5:
        score += 2
    elif kurtosis > 3:
        score += 1
    
    # 平滑度权重
    if smoothness < 0.5:
        score += 2
    elif smoothness < 0.7:
        score += 1
    
    # 模式数量权重
    if pattern_count > 10:
        score += 3
    elif pattern_count > 5:
        score += 2
    elif pattern_count > 2:
        score += 1
    
    # 根据得分判断严重程度
    if score >= 10:
        return {
            'level': '🔴 CRITICAL',
            'description': '严重异常，系统可能不可用',
            'action': '立即处理，触发P0告警，检查日志和监控，考虑回滚'
        }
    elif score >= 7:
        return {
            'level': '🟠 HIGH',
            'description': '高优先级异常，需要尽快处理',
            'action': '30分钟内响应，分析根因，准备应急方案'
        }
    elif score >= 4:
        return {
            'level': '🟡 MEDIUM',
            'description': '中等异常，需要关注',
            'action': '2小时内响应，监控趋势，计划优化'
        }
    elif score >= 2:
        return {
            'level': '🟢 LOW',
            'description': '轻微异常，持续观察',
            'action': '记录并观察，无需立即处理'
        }
    else:
        return {
            'level': '✅ NORMAL',
            'description': '正常范围内的波动',
            'action': '无需处理'
        }

if __name__ == '__main__':
    test_feature_extraction()
