import os
import datetime
import json
from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from enhanced_spike_detector import EnhancedSpikeDetector
from error_rate_monitor import ErrorRateMonitor

class UnifiedAlertingSystem:
    """
    Unified system that combines response time spike detection and error rate monitoring
    with percentile-based thresholds and environment-aware sensitivity
    """
    
    def __init__(self, 
                 response_time_percentile=99, 
                 error_rate_percentile=99,
                 min_response_time=1000,
                 min_error_rate=5.0):
        """
        Initialize the unified alerting system
        
        Args:
            response_time_percentile: Percentile threshold for response times
            error_rate_percentile: Percentile threshold for error rates
            min_response_time: Minimum response time to trigger alerts (ms)
            min_error_rate: Minimum error rate to trigger alerts (%)
        """
        self.spike_detector = EnhancedSpikeDetector(
            percentile_threshold=response_time_percentile,
            min_response_time=min_response_time
        )
        
        self.error_monitor = ErrorRateMonitor(
            percentile_threshold=error_rate_percentile,
            min_error_rate=min_error_rate
        )
        
        self.response_time_percentile = response_time_percentile
        self.error_rate_percentile = error_rate_percentile
    
    def analyze_all_apis(self, travel_app_dir, time_window_minutes=60):
        """
        Run a comprehensive analysis of all APIs
        
        Args:
            travel_app_dir: Path to the travel_app directory
            time_window_minutes: Time window for error rate analysis
            
        Returns:
            Dictionary with unified analysis results
        """
        # Detect response time spikes
        spike_data = self.spike_detector.detect_all_api_spikes(travel_app_dir)
        
        # Monitor error rates
        error_data = self.error_monitor.monitor_all_apis(
            travel_app_dir, 
            time_window_minutes=time_window_minutes
        )
        
        # Generate alerts from both systems
        response_time_alerts = self.spike_detector.generate_alerts(spike_data)
        error_rate_alerts = error_data["alerts"]  # Already formatted
        
        # Combine alerts
        all_alerts = response_time_alerts + error_rate_alerts
        
        # Sort by severity
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
        all_alerts.sort(key=lambda x: (severity_order.get(x["severity"], 3), 
                                       x.get("timestamp", "")), reverse=True)
        
        # Extract high-error APIs
        high_error_apis = error_data.get("high_error_apis", [])
        
        # Count alerts by severity
        critical_count = sum(1 for alert in all_alerts if alert["severity"] == "CRITICAL")
        high_count = sum(1 for alert in all_alerts if alert["severity"] == "HIGH")
        medium_count = sum(1 for alert in all_alerts if alert["severity"] == "MEDIUM")
        
        # Combine API statistics
        api_stats = {}
        
        # Add response time stats
        for api_name, stats in spike_data.get("api_stats", {}).items():
            if api_name not in api_stats:
                api_stats[api_name] = {}
            
            api_stats[api_name].update({
                "normal_avg": stats.get("normal_avg", 0),
                "spike_count": stats.get("spike_count", 0),
                "total_logs": stats.get("total_logs", 0),
                "percentile_value": stats.get("percentile_value", 0)
            })
        
        # Add error rate stats
        for api_name, stats in error_data.get("error_rates", {}).items():
            if api_name not in api_stats:
                api_stats[api_name] = {}
            
            api_stats[api_name].update({
                "error_rate": stats.get("error_rate", 0),
                "error_count": stats.get("error_count", 0),
                "total_requests": stats.get("total_requests", 0),
                "primary_environment": stats.get("primary_environment")
            })
        
        # Prepare unified result
        result = {
            "alerts": all_alerts,
            "critical_count": critical_count,
            "high_count": high_count,
            "medium_count": medium_count,
            "api_stats": api_stats,
            "has_alerts": len(all_alerts) > 0,
            "high_error_apis": high_error_apis,
            "response_time_percentile": self.response_time_percentile,
            "error_rate_percentile": self.error_rate_percentile,
            "spikes": spike_data.get("all_spikes", []),
            "api_spikes": spike_data.get("api_spikes", {}),
            "timestamp": datetime.datetime.now().isoformat(),
            "time_window_minutes": time_window_minutes
        }
        
        return result

def create_alerting_routes(app):
    """
    Add routes for the unified alerting system to a Flask app
    
    Args:
        app: Flask application
    """
    @app.route('/alerts')
    def alerts():
        """
        Display detailed alerts page with filtering and management options
        """
        # Create unified alerting system with default settings
        alerting_system = UnifiedAlertingSystem()
        
        # Get current directory and travel_app directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        travel_app_dir = os.path.dirname(current_dir)  # Go up one level to reach travel_app
        
        # Run comprehensive analysis
        analysis_result = alerting_system.analyze_all_apis(travel_app_dir)
        
        # Pass all required variables to the template
        return render_template('enhanced_alerts.html', 
                            alerts=analysis_result["alerts"],
                            critical_count=analysis_result["critical_count"],
                            high_count=analysis_result["high_count"],
                            medium_count=analysis_result["medium_count"],
                            api_stats=analysis_result["api_stats"],
                            high_error_apis=analysis_result["high_error_apis"],
                            percentile_threshold=analysis_result["response_time_percentile"])
    
    @app.route('/api/alerts')
    def api_alerts():
        """API endpoint to retrieve all alerts"""
        try:
            # Create unified alerting system
            alerting_system = UnifiedAlertingSystem()
            
            # Get current directory and travel_app directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            travel_app_dir = os.path.dirname(current_dir)
            
            # Run comprehensive analysis
            analysis_result = alerting_system.analyze_all_apis(travel_app_dir)
            
            return jsonify(analysis_result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/alerts/acknowledge', methods=['POST'])
    def acknowledge_alert():
        """API endpoint to acknowledge an alert"""
        try:
            data = request.json
            alert_id = data.get('alert_id')
            
            if not alert_id:
                return jsonify({"error": "Missing alert_id parameter"}), 400
            
            # Here you would implement the logic to acknowledge the alert
            # This could involve storing the acknowledged state in a database
            
            return jsonify({"success": True, "message": f"Alert {alert_id} acknowledged"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/alerts/settings', methods=['GET', 'POST'])
    def alert_settings():
        """Route for managing alert settings"""
        if request.method == 'POST':
            # Update alert settings
            try:
                # Response time settings
                response_time_percentile = float(request.form.get('response_time_percentile', 99))
                min_response_time = int(request.form.get('min_response_time', 1000))
                
                # Error rate settings
                error_rate_percentile = float(request.form.get('error_rate_percentile', 99))
                min_error_rate = float(request.form.get('min_error_rate', 5.0))
                
                # Environment factors for response time
                rt_env_factors = {}
                for env in ['dev', 'test', 'staging', 'production', 'AWS', 'Azure', 'GCP', 'on-premises']:
                    factor_key = f'rt_env_factor_{env}'
                    if factor_key in request.form:
                        rt_env_factors[env] = float(request.form.get(factor_key))
                
                # Environment factors for error rate
                er_env_factors = {}
                for env in ['dev', 'test', 'staging', 'production', 'AWS', 'Azure', 'GCP', 'on-premises']:
                    factor_key = f'er_env_factor_{env}'
                    if factor_key in request.form:
                        er_env_factors[env] = float(request.form.get(factor_key))
                
                # Region factors
                region_factors = {}
                for region in ['us-east', 'us-west', 'eu-central', 'ap-south', 'global']:
                    factor_key = f'region_factor_{region}'
                    if factor_key in request.form:
                        region_factors[region] = float(request.form.get(factor_key))
                
                # Save settings to session
                session['alert_settings'] = {
                    'response_time_percentile': response_time_percentile,
                    'min_response_time': min_response_time,
                    'error_rate_percentile': error_rate_percentile,
                    'min_error_rate': min_error_rate,
                    'rt_env_factors': rt_env_factors,
                    'er_env_factors': er_env_factors,
                    'region_factors': region_factors
                }
                
                return redirect(url_for('alert_settings'))
            except ValueError as e:
                error = f"Invalid input: {str(e)}"
                # Pass the error to the template
        
        # Get current settings from session or use defaults
        settings = session.get('alert_settings', {
            'response_time_percentile': 99,
            'min_response_time': 1000,
            'error_rate_percentile': 99,
            'min_error_rate': 5.0,
            'rt_env_factors': {
                'dev': 1.2,
                'test': 1.2,
                'staging': 1.1,
                'production': 1.0,
                'AWS': 1.0,
                'Azure': 1.05,
                'GCP': 1.03,
                'on-premises': 1.1
            },
            'er_env_factors': {
                'dev': 1.5,
                'test': 1.5,
                'staging': 1.2,
                'production': 1.0,
                'AWS': 1.0,
                'Azure': 1.1,
                'GCP': 1.1,
                'on-premises': 1.3
            },
            'region_factors': {
                'us-east': 1.0,
                'us-west': 1.0,
                'eu-central': 1.02,
                'ap-south': 1.05,
                'global': 1.1
            }
        })
        
        return render_template('alert_settings.html', settings=settings)
    
    @app.route('/dashboard')
    def dashboard():
        """Enhanced dashboard with unified alerts"""
        # Create unified alerting system
        alerting_system = UnifiedAlertingSystem()
        
        # Get current directory and travel_app directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        travel_app_dir = os.path.dirname(current_dir)
        
        # Run comprehensive analysis
        analysis_result = alerting_system.analyze_all_apis(travel_app_dir)
        
        # Pass data to template
        return render_template('dashboard.html', 
                            stats=analysis_result,
                            api_stats=analysis_result["api_stats"],
                            original_anomaly_count=0,  # Not used in new system
                            response_anomaly_count=len(analysis_result["spikes"]),
                            spikes=analysis_result["spikes"],
                            api_spikes=analysis_result["api_spikes"],
                            alerts=analysis_result["alerts"],
                            has_alerts=analysis_result["has_alerts"],
                            high_error_apis=analysis_result["high_error_apis"])
    
    @app.context_processor
    def inject_alerts_count():
        """
        Inject the alerts count into all templates
        """
        try:
            # Create unified alerting system
            alerting_system = UnifiedAlertingSystem()
            
            # Get current directory and travel_app directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            travel_app_dir = os.path.dirname(current_dir)
            
            # Run comprehensive analysis
            analysis_result = alerting_system.analyze_all_apis(travel_app_dir)
            
            return {
                'alerts_count': len(analysis_result["alerts"]),
                'critical_alerts_count': analysis_result["critical_count"]
            }
        except Exception as e:
            print(f"Error getting alerts count for context processor: {str(e)}")
            return {'alerts_count': 0, 'critical_alerts_count': 0}


# Example integration with Flask app
def integrate_with_app(app):
    """
    Integrate the unified alerting system with a Flask app
    
    Args:
        app: Flask application
    """
    # Create all required routes
    create_alerting_routes(app)
    
    # Initialize the unified alerting system with default settings
    @app.before_first_request
    def initialize_alerting():
        """Initialize alerting system before first request"""
        # Load or initialize settings
        settings = app.config.get('ALERT_SETTINGS', {
            'response_time_percentile': 99,
            'error_rate_percentile': 99
        })
        
        # Store settings in app config
        app.config['ALERT_SETTINGS'] = settings
    
    return app

# When used standalone
if __name__ == "__main__":
    # Create a Flask app
    app = Flask(__name__)
    
    # Integrate alerting system
    app = integrate_with_app(app)
    
    # Run the app
    app.run(debug=True, port=5050)