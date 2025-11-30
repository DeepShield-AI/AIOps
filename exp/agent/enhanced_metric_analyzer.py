from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime
import pyarrow.dataset as ds
import numpy as np
from pandas import Series
import ruptures as rpt
from exp.utils.input import load_parquet
from exp.utils.time import daterange
from sklearn.decomposition import PCA
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
import logging
from scipy import stats
from scipy.signal import find_peaks
import warnings
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CurveFeatureExtractor:
    """Enhanced curve feature extraction for anomalous time series patterns"""
    
    def __init__(self, min_change_threshold=0.5):
        self.min_change_threshold = min_change_threshold  # 50% minimum change to consider significant

    def extract_tsfresh_features(self, series: Series) -> Dict:
        """Extract features using tsfresh"""
        try:
            df = pd.DataFrame({
                "id": [1] * len(series),
                "time": range(len(series)),
                "value": series.values
            })
            
            # Use MinimalFCParameters for basic features as a start
            settings = MinimalFCParameters()
            
            extracted_features = extract_features(
                df, 
                column_id="id", 
                column_sort="time", 
                default_fc_parameters=settings,
                disable_progressbar=True,
                n_jobs=0 
            )
            
            if extracted_features.empty:
                return {}

            # Convert to dict and simplify keys
            features = extracted_features.iloc[0].to_dict()
            # clean keys: value__sum_values -> sum_values
            clean_features = {k.replace('value__', ''): v for k, v in features.items()}
            return clean_features
            
        except Exception as e:
            logger.error(f"tsfresh extraction failed: {e}")
            return {}
            
    def detect_sudden_changes(self, series: Series, sensitivity=2.0) -> List[Dict]:
        """检测突变点 - Detect sudden change points in time series"""
        if len(series) < 5:
            return []
            
        # Use multiple methods for robust change point detection
        changes = []
        
        # Method 1: Statistical change point detection using ruptures
        try:
            algo = rpt.Pelt(model="rbf").fit(series.values)
            change_points = algo.predict(pen=10)
            
            for cp in change_points[:-1]:  # Exclude the last point (end of series)
                if cp > 0 and cp < len(series):
                    before_mean = series.iloc[:cp].mean()
                    after_mean = series.iloc[cp:].mean()
                    change_magnitude = abs(after_mean - before_mean) / before_mean if before_mean != 0 else 0
                    
                    if change_magnitude > self.min_change_threshold:
                        changes.append({
                            'type': 'sudden_change',
                            'position': cp,
                            'timestamp': series.index[cp] if hasattr(series.index, '__getitem__') else cp,
                            'before_mean': before_mean,
                            'after_mean': after_mean,
                            'change_magnitude': change_magnitude,
                            'direction': 'increase' if after_mean > before_mean else 'decrease'
                        })
        except Exception as e:
            logger.debug(f"Change point detection failed: {e}")
            
        # Method 2: Derivative-based sudden change detection
        try:
            diff = series.diff().fillna(0)
            threshold = diff.std() * sensitivity
            sudden_changes = np.where(np.abs(diff) > threshold)[0]
            
            for idx in sudden_changes:
                if idx > 0 and idx < len(series) - 1:
                    change_val = diff.iloc[idx]
                    prev_val = series.iloc[idx-1]
                    curr_val = series.iloc[idx]
                    change_pct = abs(change_val) / prev_val if prev_val != 0 else 0
                    
                    if change_pct > self.min_change_threshold:
                        changes.append({
                            'type': 'derivative_spike',
                            'position': idx,
                            'timestamp': series.index[idx] if hasattr(series.index, '__getitem__') else idx,
                            'change_value': change_val,
                            'change_percentage': change_pct,
                            'direction': 'spike_up' if change_val > 0 else 'spike_down'
                        })
        except Exception as e:
            logger.debug(f"Derivative change detection failed: {e}")
            
        return changes
    
    def detect_trend_changes(self, series: Series, window_size=10) -> List[Dict]:
        """检测趋势变化 - Detect trend changes using sliding window regression"""
        if len(series) < window_size * 2:
            return []
            
        trends = []
        slopes = []
        
        # Calculate slopes for sliding windows
        for i in range(len(series) - window_size + 1):
            window = series.iloc[i:i+window_size]
            x = np.arange(len(window))
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, window.values)
                slopes.append({
                    'position': i + window_size // 2,
                    'slope': slope,
                    'r_squared': r_value**2,
                    'p_value': p_value
                })
            except Exception:
                slopes.append({
                    'position': i + window_size // 2,
                    'slope': 0,
                    'r_squared': 0,
                    'p_value': 1.0
                })
        
        # Detect significant trend changes
        for i in range(1, len(slopes)):
            prev_slope = slopes[i-1]['slope']
            curr_slope = slopes[i]['slope']
            
            # Check for trend reversal or significant slope change
            if (prev_slope * curr_slope < 0 or  # Sign change (trend reversal)
                abs(curr_slope - prev_slope) > np.std([s['slope'] for s in slopes]) * 2):
                
                pos = slopes[i]['position']
                if pos < len(series):
                    trends.append({
                        'type': 'trend_change',
                        'position': pos,
                        'timestamp': series.index[pos] if hasattr(series.index, '__getitem__') else pos,
                        'previous_slope': prev_slope,
                        'current_slope': curr_slope,
                        'trend_type': self._classify_trend_change(prev_slope, curr_slope),
                        'r_squared': slopes[i]['r_squared']
                    })
        
        return trends
    
    def detect_threshold_violations(self, series: Series, thresholds: Dict) -> List[Dict]:
        """检测阈值违规 - Detect threshold violations"""
        violations = []
        
        # Calculate dynamic thresholds if not provided
        if not thresholds:
            mean_val = series.mean()
            std_val = series.std()
            thresholds = {
                'upper_warning': mean_val + 2 * std_val,
                'upper_critical': mean_val + 3 * std_val,
                'lower_warning': mean_val - 2 * std_val,
                'lower_critical': mean_val - 3 * std_val
            }
        
        # Detect violations
        for threshold_name, threshold_value in thresholds.items():
            if 'upper' in threshold_name:
                violating_indices = series[series > threshold_value].index
                direction = 'above'
            else:
                violating_indices = series[series < threshold_value].index
                direction = 'below'
            
            if len(violating_indices) > 0:
                # Group consecutive violations
                violation_groups = self._group_consecutive_violations(violating_indices, series.index)
                
                for group in violation_groups:
                    max_violation_idx = group['indices'][np.argmax([abs(series.loc[idx] - threshold_value) for idx in group['indices']])]
                    max_violation_value = series.loc[max_violation_idx]
                    
                    violations.append({
                        'type': 'threshold_violation',
                        'threshold_name': threshold_name,
                        'threshold_value': threshold_value,
                        'direction': direction,
                        'start_time': group['start'],
                        'end_time': group['end'],
                        'duration': len(group['indices']),
                        'max_violation_value': max_violation_value,
                        'max_violation_position': max_violation_idx,
                        'severity': abs(max_violation_value - threshold_value) / threshold_value if threshold_value != 0 else 0
                    })
        
        return violations
    
    def detect_spikes_and_dips(self, series: Series, prominence_factor=2.0) -> List[Dict]:
        """检测尖峰和低谷 - Detect spikes and dips in the time series"""
        spikes_dips = []
        
        try:
            # Detect peaks (spikes)
            mean_val = series.mean()
            std_val = series.std()
            min_height = mean_val + prominence_factor * std_val
            
            peaks, properties = find_peaks(series.values, height=min_height, prominence=std_val)
            
            for peak_idx in peaks:
                peak_value = series.iloc[peak_idx]
                spikes_dips.append({
                    'type': 'spike',
                    'position': peak_idx,
                    'timestamp': series.index[peak_idx] if hasattr(series.index, '__getitem__') else peak_idx,
                    'value': peak_value,
                    'magnitude': (peak_value - mean_val) / mean_val if mean_val != 0 else 0,
                    'prominence': properties['prominences'][list(peaks).index(peak_idx)] if 'prominences' in properties else 0
                })
            
            # Detect dips (inverted peaks)
            max_height = mean_val - prominence_factor * std_val
            inverted_series = -series.values
            dips, dip_properties = find_peaks(inverted_series, height=-max_height, prominence=std_val)
            
            for dip_idx in dips:
                dip_value = series.iloc[dip_idx]
                spikes_dips.append({
                    'type': 'dip',
                    'position': dip_idx,
                    'timestamp': series.index[dip_idx] if hasattr(series.index, '__getitem__') else dip_idx,
                    'value': dip_value,
                    'magnitude': (mean_val - dip_value) / mean_val if mean_val != 0 else 0,
                    'prominence': dip_properties['prominences'][list(dips).index(dip_idx)] if 'prominences' in dip_properties else 0
                })
                
        except Exception as e:
            logger.debug(f"Spike/dip detection failed: {e}")
            
        return spikes_dips
    
    def _classify_trend_change(self, prev_slope: float, curr_slope: float) -> str:
        """Classify the type of trend change"""
        if prev_slope > 0 and curr_slope < 0:
            return "upward_to_downward"
        elif prev_slope < 0 and curr_slope > 0:
            return "downward_to_upward"
        elif abs(curr_slope) > abs(prev_slope) * 2:
            return "acceleration"
        elif abs(curr_slope) < abs(prev_slope) * 0.5:
            return "deceleration"
        else:
            return "slope_change"
    
    def extract_statistical_features(self, series: Series) -> Dict:
        """提取统计特征 - Extract statistical features for anomaly characterization"""
        features = {}
        
        try:
            # Basic statistics
            features['mean'] = series.mean()
            features['std'] = series.std()
            features['variance'] = series.var()
            features['min'] = series.min()
            features['max'] = series.max()
            features['range'] = features['max'] - features['min']
            
            # Coefficient of Variation (CV) - normalized volatility
            features['cv'] = features['std'] / features['mean'] if features['mean'] != 0 else 0
            
            # Skewness and Kurtosis - distribution shape
            features['skewness'] = series.skew()
            features['kurtosis'] = series.kurtosis()
            
            # Rate of change statistics
            diff = series.diff().dropna()
            if len(diff) > 0:
                features['mean_rate_of_change'] = diff.mean()
                features['std_rate_of_change'] = diff.std()
                features['max_increase'] = diff.max()
                features['max_decrease'] = diff.min()
            
            # Smoothness (inverse of second derivative)
            if len(series) > 2:
                second_diff = series.diff().diff().dropna()
                features['smoothness'] = 1.0 / (1.0 + second_diff.abs().mean())
            else:
                features['smoothness'] = 1.0
                
            # Volatility clustering (std of rolling std)
            if len(series) > 10:
                rolling_std = series.rolling(window=min(10, len(series)//2)).std().dropna()
                features['volatility_clustering'] = rolling_std.std() if len(rolling_std) > 0 else 0
            else:
                features['volatility_clustering'] = 0
                
        except Exception as e:
            logger.debug(f"Statistical feature extraction failed: {e}")
            
        return features
    
    def _group_consecutive_violations(self, indices, series_index) -> List[Dict]:
        """Group consecutive threshold violations"""
        if len(indices) == 0:
            return []
        
        groups = []
        current_group = [indices[0]]
        
        for i in range(1, len(indices)):
            # Check if indices are consecutive (considering datetime index)
            if hasattr(series_index, 'to_pydatetime'):
                # For datetime index
                prev_time = indices[i-1]
                curr_time = indices[i]
                time_diff = abs((curr_time - prev_time).total_seconds()) if hasattr(curr_time - prev_time, 'total_seconds') else 1
                if time_diff <= 300:  # 5 minutes threshold for consecutive
                    current_group.append(indices[i])
                else:
                    groups.append({
                        'indices': current_group,
                        'start': current_group[0],
                        'end': current_group[-1]
                    })
                    current_group = [indices[i]]
            else:
                # For numeric index
                if indices[i] - indices[i-1] <= 2:  # Allow small gaps
                    current_group.append(indices[i])
                else:
                    groups.append({
                        'indices': current_group,
                        'start': current_group[0],
                        'end': current_group[-1]
                    })
                    current_group = [indices[i]]
        
        # Add the last group
        groups.append({
            'indices': current_group,
            'start': current_group[0],
            'end': current_group[-1]
        })
        
        return groups


class MetricAnalyzer:
    """Enhanced metric analyzer with comprehensive curve feature extraction"""
    
    def __init__(self, root_path: str, min_change_threshold=0.05):
        self.root_path = Path(root_path)
        self.feature_extractor = CurveFeatureExtractor(min_change_threshold)
        self.min_change_threshold = min_change_threshold
        
        # Field definitions
        self.apm_fields = [
            "time", "request", "response", "rrt", "rrt_max", "error",
            "client_error", "server_error", "timeout",
            "error_ratio", "client_error_ratio", "server_error_ratio", "object_id", "object_type"
        ]
        self.infra_fields = [
            "time", "cf", "device", "instance", "kpi_key", "kpi_name", "kubernetes_node",
            "mountpoint", "namespace", "object_type", "pod", "value", "sql_type", "type"
        ]
        self.infra_schema_fields = [
            "time", "cf", "device", "instance", "kpi_key", "kpi_name", "kubernetes_node",
            "mountpoint", "namespace", "object_type", "pod", "sql_type", "type"
        ]
        
    def load_apm_data(self, start: datetime, end: datetime, max_workers=4) -> pd.DataFrame:
        """Load APM data for the specified time range"""
        files = []
        for day in daterange(start, end):
            files.extend(glob.glob(f"{self.root_path}/{day}/metric-parquet/apm/service/*.parquet"))

        results = []
        filter = (ds.field("time") >= start) & (ds.field("time") <= end)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(load_parquet, Path(f), self.apm_fields, filter_=filter): f for f in files}
            for future in as_completed(futures):
                df = future.result()
                if not df.empty:
                    results.append(df[self.apm_fields])

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    def load_infra_or_other_data(self, file_pattern: str, start: datetime, end: datetime, max_workers=4) -> pd.DataFrame:
        """Load infrastructure or other metric data"""
        files = []
        for day in daterange(start, end):
            files.extend(glob.glob(f"{self.root_path}/{day}/metric-parquet/{file_pattern}"))

        results = []
        filter = (ds.field("time") >= start) & (ds.field("time") <= end)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(load_parquet, Path(f), filter_=filter): f for f in files}
            for future in as_completed(futures):
                df = future.result()
                if not df.empty:
                    metric_candidates = list(set(df.columns) - set(self.infra_schema_fields))
                    if len(metric_candidates) == 1:
                        df["value"] = df[metric_candidates[0]]
                        df["pod"] = df["pod"].astype(str).str.replace(r"-\d+$", "", regex=True)
                        results.append(df[self.infra_fields])

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    def analyze_time_series_features(self, series: Series, timestamps: Series, 
                                   metric_name: str, instance_name: str, 
                                   metric_type: str) -> Optional[Dict]:
        """Analyze time series and extract curve features"""
        if len(series) < 10:
            logger.debug(f"Skipping {metric_name} for {instance_name}: insufficient data points ({len(series)})")
            return None
            
        try:
            # Calculate baseline statistics
            normal_mean = series.mean()
            normal_std = series.std()
            
            logger.debug(f"Analyzing {metric_name} for {instance_name}: {len(series)} points, mean={normal_mean:.2f}")
            
            # Extract all curve features
            sudden_changes = self.feature_extractor.detect_sudden_changes(series)
            trend_changes = self.feature_extractor.detect_trend_changes(series)
            threshold_violations = self.feature_extractor.detect_threshold_violations(series, {})
            spikes_dips = self.feature_extractor.detect_spikes_and_dips(series)
            
            # Extract statistical features
            statistical_features = self.feature_extractor.extract_statistical_features(series)
            
            # Extract tsfresh features
            tsfresh_features = self.feature_extractor.extract_tsfresh_features(series)
            
            logger.debug(f"Feature extraction for {metric_name}: sudden_changes={len(sudden_changes)}, "
                        f"trend_changes={len(trend_changes)}, violations={len(threshold_violations)}, "
                        f"spikes_dips={len(spikes_dips)}")
        except Exception as e:
            logger.error(f"Error in feature extraction for {metric_name} on {instance_name}: {e}")
            return None
        
        # Determine if this is anomalous
        is_anomalous = (len(sudden_changes) > 0 or len(trend_changes) > 0 or 
                       len(threshold_violations) > 0 or len(spikes_dips) > 0)
        
        if not is_anomalous:
            return None
            
        # Calculate anomalous period statistics
        anomaly_indices = set()
        for change in sudden_changes + trend_changes + spikes_dips:
            if 'position' in change:
                pos = change['position']
                # Convert numpy types to int and handle Timestamp objects
                if hasattr(pos, 'item'):  # numpy scalar
                    pos = pos.item()
                elif hasattr(pos, 'timestamp'):  # Timestamp object
                    # Find the index in series that matches this timestamp
                    try:
                        pos = series.index.get_loc(pos)
                    except KeyError:
                        continue
                if isinstance(pos, int):
                    anomaly_indices.add(pos)
        for violation in threshold_violations:
            if 'max_violation_position' in violation:
                pos = violation['max_violation_position']
                # Convert numpy types to int and handle Timestamp objects
                if hasattr(pos, 'item'):  # numpy scalar
                    pos = pos.item()
                elif hasattr(pos, 'timestamp'):  # Timestamp object
                    try:
                        pos = series.index.get_loc(pos)
                    except KeyError:
                        continue
                if isinstance(pos, int):
                    anomaly_indices.add(pos)
        
        if anomaly_indices:
            # Filter out invalid indices
            valid_indices = [idx for idx in anomaly_indices if 0 <= idx < len(series)]
            if valid_indices:
                anomaly_values = series.iloc[valid_indices]
                anomalous_mean = anomaly_values.mean()
            else:
                anomalous_mean = normal_mean
        else:
            anomalous_mean = normal_mean
            
        # Calculate change rate
        change_rate = ((anomalous_mean - normal_mean) / normal_mean * 100) if normal_mean != 0 else 0
        
        # Filter out minor changes
        if abs(change_rate) < self.min_change_threshold * 100:
            return None
            
        # Generate curve feature description
        curve_features = self._generate_curve_feature_description(
            sudden_changes, trend_changes, threshold_violations, spikes_dips, series
        )
        
        # Determine instance type based on metric type and name
        instance_type = self._determine_instance_type(metric_type, metric_name, instance_name)
        
        return {
            'metric_granularity': f"{metric_type}_{metric_name}",
            'instance_type': instance_type,
            'instance_name': instance_name,
            'change_rate': change_rate,
            'normal_mean': normal_mean,
            'anomalous_mean': anomalous_mean,
            'curve_features': curve_features,
            'sudden_changes': sudden_changes,
            'trend_changes': trend_changes,
            'threshold_violations': threshold_violations,
            'spikes_dips': spikes_dips,
            'statistical_features': statistical_features,
            'tsfresh_features': tsfresh_features
        }
    
    def _determine_instance_type(self, metric_type: str, metric_name: str, instance_name: str) -> str:
        """Determine the instance type based on metric information"""
        if metric_type == "apm":
            return "service"
        elif metric_type == "infra":
            if "node" in metric_name.lower() or "kubernetes_node" in instance_name:
                return "node"
            elif "pod" in metric_name.lower():
                return "pod"
            else:
                return "infra"
        elif metric_type == "other":
            if instance_name.startswith("pd"):
                return "pd"
            elif instance_name.startswith("tikv"):
                return "tikv"
            else:
                return "other"
        return "unknown"
    
    def _generate_curve_feature_description(self, sudden_changes, trend_changes, 
                                          threshold_violations, spikes_dips, series) -> str:
        """Generate a human-readable description of curve features"""
        features = []
        
        # Describe spikes and dips
        max_spike = None
        max_dip = None
        for item in spikes_dips:
            if item['type'] == 'spike':
                if max_spike is None or item['value'] > max_spike['value']:
                    max_spike = item
            elif item['type'] == 'dip':
                if max_dip is None or item['value'] < max_dip['value']:
                    max_dip = item
        
        if max_spike:
            features.append(f"spike to {max_spike['value']:.1f}")
        if max_dip:
            features.append(f"dip to {max_dip['value']:.1f}")
            
        # Describe sudden changes
        for change in sudden_changes:
            if change['type'] == 'sudden_change' and change.get('change_magnitude', 0) > 0.2:  # 20% change
                direction = "jump" if change['direction'] == 'increase' else "drop"
                features.append(f"{direction} {change['change_magnitude']*100:.1f}%")
            elif change['type'] == 'derivative_spike' and change.get('change_percentage', 0) > 0.2:
                direction = "spike up" if change['direction'] == 'spike_up' else "spike down"
                features.append(f"{direction} {change['change_percentage']*100:.1f}%")
        
        # Describe trend changes
        for trend in trend_changes:
            if trend['trend_type'] == 'upward_to_downward':
                features.append("trend reversal up→down")
            elif trend['trend_type'] == 'downward_to_upward':
                features.append("trend reversal down→up")
            elif trend['trend_type'] == 'acceleration':
                features.append("accelerating trend")
        
        # Describe threshold violations
        critical_violations = [v for v in threshold_violations if 'critical' in v['threshold_name']]
        if critical_violations:
            features.append(f"critical threshold breach")
        
        return ", ".join(features) if features else "anomalous pattern"
    
    def analyze_metrics(self, start_time: datetime, end_time: datetime) -> List[str]:
        """Main analysis function that returns formatted results"""
        results = []
        
        # Load different types of data
        apm_data = self.load_apm_data(start_time, end_time)
        infra_data = self.load_infra_or_other_data('infra/infra_pod/*.parquet', start_time, end_time)
        other_data = self.load_infra_or_other_data('other/*.parquet', start_time, end_time)
        
        # Analyze APM data
        if not apm_data.empty:
            apm_results = self._analyze_apm_metrics(apm_data)
            results.extend(apm_results)
        
        # Analyze Infrastructure data
        if not infra_data.empty:
            infra_results = self._analyze_infra_metrics(infra_data)
            results.extend(infra_results)
        
        # Analyze Other data
        if not other_data.empty:
            other_results = self._analyze_other_metrics(other_data)
            results.extend(other_results)
        
        # Format results according to specification
        formatted_results = []
        for result in results:
            if result:
                formatted_line = self._format_result(result)
                formatted_results.append(formatted_line)
        
        return formatted_results
    
    def _analyze_apm_metrics(self, apm_data: pd.DataFrame) -> List[Dict]:
        """Analyze APM metrics"""
        results = []
        
        # Group by service and analyze key metrics
        key_metrics = ['error_ratio', 'client_error_ratio', 'server_error_ratio', 'timeout', 'rrt', 'rrt_max']
        
        for service, service_group in apm_data.groupby('object_id'):
            for metric in key_metrics:
                if metric in service_group.columns:
                    series = service_group[metric].dropna()
                    timestamps = service_group['time']
                    
                    if len(series) >= 10:
                        analysis = self.analyze_time_series_features(
                            series, timestamps, metric, service, 'apm'
                        )
                        if analysis:
                            results.append(analysis)
        
        return results
    
    def _analyze_infra_metrics(self, infra_data: pd.DataFrame) -> List[Dict]:
        """Analyze infrastructure metrics"""
        results = []
        
        for pod, pod_group in infra_data.groupby('pod'):
            for kpi, kpi_group in pod_group.groupby('kpi_key'):
                series = kpi_group['value'].dropna()
                timestamps = kpi_group['time']
                
                if len(series) >= 10:
                    analysis = self.analyze_time_series_features(
                        series, timestamps, kpi, pod, 'infra'
                    )
                    if analysis:
                        results.append(analysis)
        
        return results
    
    def _analyze_other_metrics(self, other_data: pd.DataFrame) -> List[Dict]:
        """Analyze other metrics (PD, TiKV, etc.)"""
        results = []
        
        for pod, pod_group in other_data.groupby('pod'):
            for kpi, kpi_group in pod_group.groupby('kpi_key'):
                series = kpi_group['value'].dropna()
                timestamps = kpi_group['time']
                
                if len(series) >= 10:
                    analysis = self.analyze_time_series_features(
                        series, timestamps, kpi, pod, 'other'
                    )
                    if analysis:
                        results.append(analysis)
        
        return results
    
    def _format_result(self, analysis: Dict) -> str:
        """Format analysis result according to specification"""
        # Enhanced Format: [指标粒度_指标名][实例类型 实例名称][[特征:值]...]
        
        metric_granularity = analysis['metric_granularity']
        instance_type = analysis['instance_type']
        instance_name = analysis['instance_name']
        tsfresh_features = analysis.get('tsfresh_features', {})
        
        # Format features
        feature_parts = []
        for k, v in tsfresh_features.items():
            # Format value to 2 decimal places if it's a float
            if isinstance(v, float):
                feature_parts.append(f"{k}:{v:.2f}")
            else:
                feature_parts.append(f"{k}:{v}")
        
        features_str = "[" + " ".join(feature_parts) + "]"
        
        # Format the result string
        result = (f"[{metric_granularity}][{instance_type} {instance_name}]{features_str}")
        
        return result


def analyze_metrics_with_enhanced_features(root_path: str, start_time: datetime, end_time: datetime) -> List[str]:
    """
    Main function to analyze metrics with enhanced curve feature extraction
    
    Args:
        root_path: Path to the metric data
        start_time: Start time for analysis
        end_time: End time for analysis
        
    Returns:
        List of formatted analysis results
    """
    analyzer = MetricAnalyzer(root_path)
    return analyzer.analyze_metrics(start_time, end_time)
