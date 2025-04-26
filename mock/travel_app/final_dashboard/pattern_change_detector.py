import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any

class PatternChangeDetector:
    """
    Advanced pattern change detection system for API performance metrics
    
    Key Detection Strategies:
    1. Statistical trend analysis
    2. Change point detection
    3. Long-term pattern recognition
    """
    
    def __init__(self, window_size=10, significance_level=0.05):
        """
        Initialize pattern change detector
        
        Args:
            window_size (int): Number of intervals to analyze for pattern changes
            significance_level (float): Statistical significance threshold
        """
        self.window_size = window_size
        self.significance_level = significance_level
    
    def detect_trend_changes(self, time_series: List[float]) -> Dict[str, Any]:
        """
        Detect trend changes using multiple statistical methods
        
        Args:
            time_series (List[float]): Time series of metric values
        
        Returns:
            Dict with trend change analysis results
        """
        if len(time_series) < self.window_size:
            return {
                "trend_detected": False,
                "trend_type": "insufficient_data",
                "confidence": 0.0
            }
        
        # Convert to numpy array for easier manipulation
        data = np.array(time_series)
        
        # Linear regression for trend detection
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        
        # Mann-Kendall trend test for monotonic trends
        tau, mk_p_value = stats.kendalltau(x, data)
        
        # Detect potential change points using different techniques
        change_points = self._detect_change_points(data)
        
        # Interpret trend characteristics
        if p_value <= self.significance_level:
            if slope > 0:
                trend_type = "increasing"
            elif slope < 0:
                trend_type = "decreasing"
            else:
                trend_type = "stable"
        else:
            trend_type = "no_significant_trend"
        
        # Calculate trend confidence
        trend_confidence = min(1.0, abs(r_value) * abs(slope) * 10)
        
        return {
            "trend_detected": p_value <= self.significance_level,
            "trend_type": trend_type,
            "slope": slope,
            "r_squared": r_value**2,
            "p_value": p_value,
            "trend_confidence": trend_confidence,
            "change_points": change_points,
            "mann_kendall": {
                "tau": tau,
                "p_value": mk_p_value
            }
        }
    
    def _detect_change_points(self, data: np.ndarray, max_change_points=3) -> List[Dict[str, Any]]:
        """
        Advanced change point detection using multiple techniques
        
        Args:
            data (np.ndarray): Time series data
            max_change_points (int): Maximum number of change points to detect
        
        Returns:
            List of detected change points with details
        """
        change_points = []
        
        # Compute differences between consecutive points
        differences = np.diff(data)
        
        # Detect abrupt changes using standard deviation multiplier
        std_dev = np.std(differences)
        mean_diff = np.mean(differences)
        
        for i in range(1, len(data)):
            # Look for significant jumps relative to overall variability
            jump_magnitude = abs(data[i] - data[i-1])
            
            if jump_magnitude > (mean_diff + 2 * std_dev):
                change_point = {
                    "index": i,
                    "value_before": data[i-1],
                    "value_after": data[i],
                    "jump_magnitude": jump_magnitude,
                    "relative_change": (data[i] - data[i-1]) / data[i-1] * 100
                }
                change_points.append(change_point)
                
                # Limit to max change points
                if len(change_points) >= max_change_points:
                    break
        
        return change_points
    
    def analyze_api_performance_patterns(self, api_time_series: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Comprehensive pattern analysis for multiple API metrics
        
        Args:
            api_time_series (Dict): Time series data for different APIs
        
        Returns:
            Detailed pattern analysis report
        """
        pattern_report = {}
        
        for api_name, time_series in api_time_series.items():
            trend_analysis = self.detect_trend_changes(time_series)
            
            pattern_report[api_name] = {
                "trend_analysis": trend_analysis,
                "raw_data": {
                    "mean": np.mean(time_series),
                    "median": np.median(time_series),
                    "std_dev": np.std(time_series)
                }
            }
        
        return pattern_report
    
    def generate_pattern_insights(self, performance_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate actionable insights from pattern analysis
        
        Args:
            performance_report (Dict): Comprehensive performance pattern report
        
        Returns:
            List of insights with recommendations
        """
        insights = []
        
        for api_name, analysis in performance_report.items():
            trend_data = analysis['trend_analysis']
            
            # High-risk pattern changes
            if trend_data['trend_type'] in ['increasing', 'decreasing'] and trend_data['trend_confidence'] > 0.7:
                insight = {
                    "api": api_name,
                    "severity": "high" if trend_data['trend_type'] == 'increasing' else "medium",
                    "type": "performance_trend",
                    "description": f"Detected {'increasing' if trend_data['trend_type'] == 'increasing' else 'decreasing'} performance trend",
                    "details": {
                        "slope": trend_data['slope'],
                        "confidence": trend_data['trend_confidence'],
                        "p_value": trend_data['p_value']
                    },
                    "recommendation": (
                        "Increasing trend suggests potential performance degradation. "
                        "Consider optimizing API implementation, reviewing recent changes, "
                        "and conducting detailed performance profiling."
                    ) if trend_data['trend_type'] == 'increasing' else (
                        "Decreasing trend might indicate performance improvements. "
                        "Validate the positive changes and document successful optimization strategies."
                    )
                }
                insights.append(insight)
            
            # Detect significant change points
            if trend_data.get('change_points'):
                for change_point in trend_data['change_points']:
                    change_insight = {
                        "api": api_name,
                        "severity": "medium",
                        "type": "sudden_change",
                        "description": f"Detected significant performance change at interval point",
                        "details": {
                            "jump_magnitude": change_point['jump_magnitude'],
                            "relative_change": change_point['relative_change']
                        },
                        "recommendation": (
                            "Investigate the specific time interval for potential "
                            "architectural, code, or infrastructure changes that might "
                            "have caused the performance variation."
                        )
                    }
                    insights.append(change_insight)
        
        return insights

# Example usage in the context of the existing forecasting system
def enhance_forecasting_system(forecasting_system):
    """
    Enhance existing forecasting system with pattern change detection
    
    Args:
        forecasting_system (APIForecastingSystem): Existing forecasting system
    
    Returns:
        Updated forecasting system with pattern change detection
    """
    pattern_detector = PatternChangeDetector()
    
    # Extract time series data from the forecasting system
    api_time_series = {
        api: forecasting_system.time_series[api]['error_rates']
        for api in forecasting_system.api_names + ['system']
    }
    
    # Perform pattern analysis
    performance_report = pattern_detector.analyze_api_performance_patterns(api_time_series)
    
    # Generate insights
    pattern_insights = pattern_detector.generate_pattern_insights(performance_report)
    
    # Attach insights to forecasting system
    forecasting_system.pattern_insights = pattern_insights
    
    return forecasting_system