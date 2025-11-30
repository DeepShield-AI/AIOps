import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add current directory to path so we can import the module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_metric_analyzer import MetricAnalyzer, CurveFeatureExtractor

def verify_tsfresh_features():
    print("=== Verifying tsfresh feature extraction ===")
    
    # Create a dummy time series
    # Sine wave with some noise
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)
    series = pd.Series(y)
    
    extractor = CurveFeatureExtractor()
    
    print("Extracting features using tsfresh...")
    features = extractor.extract_tsfresh_features(series)
    
    print(f"Extracted {len(features)} features.")
    for k, v in list(features.items())[:5]:  # Show first 5
        print(f"  {k}: {v}")
        
    print("\n=== Verifying MetricAnalyzer integration and formatting ===")
    
    analyzer = MetricAnalyzer(".")
    
    # Simulate an analysis result dictionary
    # We need to mock what analyze_time_series_features returns
    # But let's try to actually run it if possible, or just use _format_result
    
    analysis_result = {
        'metric_granularity': 'apm_response_time',
        'instance_type': 'service',
        'instance_name': 'payment-service',
        'change_rate': 50.0,
        'normal_mean': 100.0,
        'anomalous_mean': 150.0,
        'curve_features': 'spike to 200',
        'statistical_features': {'mean': 100, 'std': 10},
        'tsfresh_features': features
    }
    
    formatted_result = analyzer._format_result(analysis_result)
    print("Formatted Result:")
    print(formatted_result)
    
    # Verify format
    if "[apm_response_time][service payment-service][" in formatted_result:
        print("\nSUCCESS: Format matches expectations.")
    else:
        print("\nFAILURE: Format does not match expectations.")

if __name__ == "__main__":
    verify_tsfresh_features()
