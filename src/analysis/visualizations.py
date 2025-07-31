"""Comprehensive visualization module for Chicago 311 data analysis."""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
import json
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class Chicago311Visualizer:
    """Comprehensive visualization toolkit for Chicago 311 data."""
    
    def __init__(self, output_dir: str = "data/exports"):
        """Initialize visualizer with output directory."""
        self.output_dir = output_dir
        self.ensure_output_dir()
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Chicago coordinates for maps
        self.chicago_coords = [41.8781, -87.6298]
        
    def ensure_output_dir(self):
        """Ensure output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)
        os.makedirs(f"{self.output_dir}/maps", exist_ok=True)
    
    def create_status_distribution_chart(self, status_data: List[Dict]) -> str:
        """Create status distribution visualization."""
        logger.info("=Ê Creating status distribution chart...")
        
        try:
            if not status_data:
                logger.warning("No status data provided")
                return ""
            
            # Convert to DataFrame
            df = pd.DataFrame(status_data)
            df.columns = ['Status', 'Count']
            
            # Create figure with subplots
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "xy"}, {"type": "domain"}]],
                subplot_titles=("Service Request Status Distribution", "Status Breakdown")
            )
            
            # Bar chart
            fig.add_trace(
                go.Bar(
                    x=df['Status'],
                    y=df['Count'],
                    name="Status Count",
                    marker_color='viridis'
                ),
                row=1, col=1
            )
            
            # Pie chart
            fig.add_trace(
                go.Pie(
                    labels=df['Status'],
                    values=df['Count'],
                    name="Status Distribution"
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title="Chicago 311 Service Request Status Analysis",
                height=500,
                showlegend=False
            )
            
            # Save plot
            filename = f"{self.output_dir}/plots/status_distribution.html"
            fig.write_html(filename)
            logger.info(f" Status distribution chart saved: {filename}")
            
            return filename
            
        except Exception as e:
            logger.error(f"L Error creating status chart: {e}")
            return ""
    
    def create_temporal_trends_chart(self, temporal_data: List[Dict]) -> str:
        """Create temporal trends visualization."""
        logger.info("=Å Creating temporal trends chart...")
        
        try:
            if not temporal_data:
                logger.warning("No temporal data provided")
                return ""
            
            # Convert to DataFrame
            df = pd.DataFrame(temporal_data)
            
            # Create date column
            if 'year' in df.columns:
                df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
            else:
                logger.warning("Temporal data format not recognized")
                return ""
            
            # Sort by date
            df = df.sort_values('date')
            
            # Create line chart
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['count'],
                    mode='lines+markers',
                    name='Service Requests',
                    line=dict(width=3),
                    marker=dict(size=8)
                )
            )
            
            fig.update_layout(
                title="Chicago 311 Service Requests - Monthly Trends",
                xaxis_title="Date",
                yaxis_title="Number of Requests",
                height=500,
                template="plotly_white"
            )
            
            # Save plot
            filename = f"{self.output_dir}/plots/temporal_trends.html"
            fig.write_html(filename)
            logger.info(f" Temporal trends chart saved: {filename}")
            
            return filename
            
        except Exception as e:
            logger.error(f"L Error creating temporal chart: {e}")
            return ""
    
    def create_service_type_analysis(self, service_data: List[Dict]) -> str:
        """Create service type analysis visualization."""
        logger.info("=' Creating service type analysis...")
        
        try:
            if not service_data:
                logger.warning("No service type data provided")
                return ""
            
            # Convert to DataFrame
            df = pd.DataFrame(service_data)
            df.columns = ['Service_Type', 'Count']
            
            # Get top 15 for better visualization
            df_top = df.head(15)
            
            # Create horizontal bar chart
            fig = go.Figure()
            
            fig.add_trace(
                go.Bar(
                    x=df_top['Count'],
                    y=df_top['Service_Type'],
                    orientation='h',
                    marker_color='lightblue',
                    text=df_top['Count'],
                    textposition='outside'
                )
            )
            
            fig.update_layout(
                title="Top 15 Service Request Types",
                xaxis_title="Number of Requests",
                yaxis_title="Service Type",
                height=600,
                template="plotly_white",
                yaxis={'categoryorder': 'total ascending'}
            )
            
            # Save plot
            filename = f"{self.output_dir}/plots/service_types.html"
            fig.write_html(filename)
            logger.info(f" Service type analysis saved: {filename}")
            
            return filename
            
        except Exception as e:
            logger.error(f"L Error creating service type chart: {e}")
            return ""
    
    def create_performance_dashboard(self, performance_data: Dict[str, Any]) -> str:
        """Create comprehensive performance dashboard."""
        logger.info("¡ Creating performance dashboard...")
        
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Response Times by Service Type",
                    "Completion Rates by Service Type", 
                    "Department Performance",
                    "Response Time Distribution"
                ),
                specs=[
                    [{"type": "xy"}, {"type": "xy"}],
                    [{"type": "xy"}, {"type": "xy"}]
                ]
            )
            
            # Response times chart
            if 'response_times_by_type' in performance_data:
                response_data = performance_data['response_times_by_type'][:10]  # Top 10
                if response_data:
                    df_response = pd.DataFrame(response_data)
                    fig.add_trace(
                        go.Bar(
                            x=[item['_id'] for item in response_data],
                            y=[item['avg_response_time_hours'] for item in response_data],
                            name="Avg Response Time (Hours)",
                            marker_color='orange'
                        ),
                        row=1, col=1
                    )
            
            # Completion rates chart
            if 'completion_rates_by_type' in performance_data:
                completion_data = performance_data['completion_rates_by_type'][:10]  # Top 10
                if completion_data:
                    fig.add_trace(
                        go.Bar(
                            x=[item['sr_type'] for item in completion_data],
                            y=[item['completion_rate'] for item in completion_data],
                            name="Completion Rate (%)",
                            marker_color='green'
                        ),
                        row=1, col=2
                    )
            
            # Department performance
            if 'department_performance' in performance_data:
                dept_data = performance_data['department_performance'][:8]  # Top 8
                if dept_data:
                    fig.add_trace(
                        go.Bar(
                            x=[item['department'] for item in dept_data],
                            y=[item['completion_rate'] for item in dept_data],
                            name="Dept Completion Rate (%)",
                            marker_color='purple'
                        ),
                        row=2, col=1
                    )
            
            # Response time distribution (if available)
            if 'response_times_by_type' in performance_data:
                response_times = [item['avg_response_time_hours'] for item in performance_data['response_times_by_type']]
                if response_times:
                    fig.add_trace(
                        go.Histogram(
                            x=response_times,
                            name="Response Time Distribution",
                            marker_color='red',
                            opacity=0.7
                        ),
                        row=2, col=2
                    )
            
            fig.update_layout(
                title="Chicago 311 Service Performance Dashboard",
                height=800,
                showlegend=False,
                template="plotly_white"
            )
            
            # Save plot
            filename = f"{self.output_dir}/plots/performance_dashboard.html"
            fig.write_html(filename)
            logger.info(f" Performance dashboard saved: {filename}")
            
            return filename
            
        except Exception as e:
            logger.error(f"L Error creating performance dashboard: {e}")
            return ""
    
    def create_geographic_heatmap(self, mongo_handler) -> str:
        """Create geographic heatmap of service requests."""
        logger.info("=ú Creating geographic heatmap...")
        
        try:
            # Get geographic data with coordinates
            pipeline = [
                {
                    "$match": {
                        "location": {"$exists": True},
                        "sr_type": {"$ne": None}
                    }
                },
                {
                    "$group": {
                        "_id": {
                            "lat": {"$arrayElemAt": ["$location.coordinates", 1]},
                            "lon": {"$arrayElemAt": ["$location.coordinates", 0]},
                            "ward": "$ward"
                        },
                        "count": {"$sum": 1}
                    }
                },
                {"$match": {"count": {"$gte": 5}}},  # At least 5 requests
                {"$limit": 1000}  # Limit for performance
            ]
            
            geo_data = list(mongo_handler.collection.aggregate(pipeline))
            
            if not geo_data:
                logger.warning("No geographic data available")
                return ""
            
            # Create folium map
            m = folium.Map(
                location=self.chicago_coords,
                zoom_start=11,
                tiles='OpenStreetMap'
            )
            
            # Prepare data for heatmap
            heat_data = []
            for item in geo_data:
                if item['_id']['lat'] and item['_id']['lon']:
                    try:
                        lat = float(item['_id']['lat'])
                        lon = float(item['_id']['lon'])
                        if 41.6 <= lat <= 42.1 and -87.9 <= lon <= -87.5:  # Chicago bounds
                            heat_data.append([lat, lon, item['count']])
                    except (ValueError, TypeError):
                        continue
            
            if heat_data:
                # Add heatmap
                heatmap = plugins.HeatMap(heat_data, radius=15, blur=10)
                heatmap.add_to(m)
                
                # Add ward boundaries if available
                folium.Marker(
                    self.chicago_coords,
                    popup="Chicago Center",
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
            
            # Save map
            filename = f"{self.output_dir}/maps/service_requests_heatmap.html"
            m.save(filename)
            logger.info(f" Geographic heatmap saved: {filename}")
            
            return filename
            
        except Exception as e:
            logger.error(f"L Error creating geographic heatmap: {e}")
            return ""
    
    def create_ward_analysis_map(self, ward_data: List[Dict]) -> str:
        """Create ward-based analysis map."""
        logger.info("<Ø Creating ward analysis map...")
        
        try:
            if not ward_data:
                logger.warning("No ward data provided")
                return ""
            
            # Create map
            m = folium.Map(
                location=self.chicago_coords,
                zoom_start=10,
                tiles='OpenStreetMap'
            )
            
            # Add ward data as markers
            for ward_info in ward_data[:20]:  # Top 20 wards
                ward_num = ward_info['_id']
                count = ward_info['count']
                
                # Approximate ward center (this would ideally use real ward boundaries)
                # For now, we'll create a simple distribution around Chicago
                lat_offset = (ward_num % 10 - 5) * 0.02
                lon_offset = (ward_num % 7 - 3) * 0.03
                
                lat = self.chicago_coords[0] + lat_offset
                lon = self.chicago_coords[1] + lon_offset
                
                # Create circle marker with size based on request count
                radius = min(max(count / 100, 5), 50)  # Scale between 5-50
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=radius,
                    popup=f"Ward {ward_num}: {count:,} requests",
                    tooltip=f"Ward {ward_num}",
                    color='blue',
                    fillColor='lightblue',
                    fillOpacity=0.7
                ).add_to(m)
            
            # Save map
            filename = f"{self.output_dir}/maps/ward_analysis.html"
            m.save(filename)
            logger.info(f" Ward analysis map saved: {filename}")
            
            return filename
            
        except Exception as e:
            logger.error(f"L Error creating ward analysis map: {e}")
            return ""
    
    def create_temporal_patterns_dashboard(self, temporal_data: Dict[str, Any]) -> str:
        """Create comprehensive temporal patterns dashboard."""
        logger.info("ð Creating temporal patterns dashboard...")
        
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Day of Week Pattern",
                    "Hour of Day Pattern",
                    "Seasonal Pattern (Month)",
                    "Monthly Trends"
                )
            )
            
            # Day of week pattern
            if 'day_of_week_pattern' in temporal_data:
                dow_data = temporal_data['day_of_week_pattern']
                if dow_data:
                    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                    day_counts = [0] * 7
                    
                    for item in dow_data:
                        day_idx = item['_id'] - 1  # MongoDB day of week is 1-7
                        if 0 <= day_idx < 7:
                            day_counts[day_idx] = item['count']
                    
                    fig.add_trace(
                        go.Bar(
                            x=days,
                            y=day_counts,
                            name="Day of Week",
                            marker_color='lightcoral'
                        ),
                        row=1, col=1
                    )
            
            # Hour of day pattern
            if 'hour_of_day_pattern' in temporal_data:
                hour_data = temporal_data['hour_of_day_pattern']
                if hour_data:
                    hours = list(range(24))
                    hour_counts = [0] * 24
                    
                    for item in hour_data:
                        hour = item['_id']
                        if 0 <= hour < 24:
                            hour_counts[hour] = item['count']
                    
                    fig.add_trace(
                        go.Scatter(
                            x=hours,
                            y=hour_counts,
                            mode='lines+markers',
                            name="Hour of Day",
                            line=dict(color='green', width=3)
                        ),
                        row=1, col=2
                    )
            
            # Seasonal pattern
            if 'seasonal_pattern' in temporal_data:
                seasonal_data = temporal_data['seasonal_pattern']
                if seasonal_data:
                    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    month_counts = [0] * 12
                    
                    for item in seasonal_data:
                        month_idx = item['_id'] - 1  # MongoDB month is 1-12
                        if 0 <= month_idx < 12:
                            month_counts[month_idx] = item['count']
                    
                    fig.add_trace(
                        go.Bar(
                            x=months,
                            y=month_counts,
                            name="Seasonal",
                            marker_color='gold'
                        ),
                        row=2, col=1
                    )
            
            # Monthly trends
            if 'monthly_trends' in temporal_data:
                monthly_data = temporal_data['monthly_trends']
                if monthly_data:
                    dates = []
                    counts = []
                    
                    for item in monthly_data:
                        if '_id' in item and 'year' in item['_id'] and 'month' in item['_id']:
                            date = pd.to_datetime(f"{item['_id']['year']}-{item['_id']['month']}-01")
                            dates.append(date)
                            counts.append(item['count'])
                    
                    if dates and counts:
                        fig.add_trace(
                            go.Scatter(
                                x=dates,
                                y=counts,
                                mode='lines+markers',
                                name="Monthly Trends",
                                line=dict(color='purple', width=3)
                            ),
                            row=2, col=2
                        )
            
            fig.update_layout(
                title="Chicago 311 Temporal Patterns Analysis",
                height=800,
                showlegend=False,
                template="plotly_white"
            )
            
            # Save plot
            filename = f"{self.output_dir}/plots/temporal_patterns_dashboard.html"
            fig.write_html(filename)
            logger.info(f" Temporal patterns dashboard saved: {filename}")
            
            return filename
            
        except Exception as e:
            logger.error(f"L Error creating temporal patterns dashboard: {e}")
            return ""
    
    def generate_comprehensive_report(self, analysis_results: Dict[str, Any], 
                                    mongo_handler=None) -> Dict[str, str]:
        """Generate comprehensive visualization report."""
        logger.info("=È Generating comprehensive visualization report...")
        
        generated_files = {}
        
        try:
            # Basic statistics visualizations
            if 'basic_statistics' in analysis_results:
                basic = analysis_results['basic_statistics']
                
                # Status distribution
                if 'status_distribution' in basic:
                    file_path = self.create_status_distribution_chart(basic['status_distribution'])
                    if file_path:
                        generated_files['status_distribution'] = file_path
                
                # Service types
                if 'top_service_types' in basic:
                    file_path = self.create_service_type_analysis(basic['top_service_types'])
                    if file_path:
                        generated_files['service_types'] = file_path
                
                # Temporal trends
                if 'temporal_overview' in basic:
                    file_path = self.create_temporal_trends_chart(basic['temporal_overview'])
                    if file_path:
                        generated_files['temporal_trends'] = file_path
            
            # Performance visualizations
            if 'service_performance' in analysis_results:
                file_path = self.create_performance_dashboard(analysis_results['service_performance'])
                if file_path:
                    generated_files['performance_dashboard'] = file_path
            
            # Temporal patterns
            if 'temporal_patterns' in analysis_results:
                file_path = self.create_temporal_patterns_dashboard(analysis_results['temporal_patterns'])
                if file_path:
                    generated_files['temporal_patterns_dashboard'] = file_path
            
            # Geographic visualizations
            if mongo_handler:
                # Heatmap
                file_path = self.create_geographic_heatmap(mongo_handler)
                if file_path:
                    generated_files['geographic_heatmap'] = file_path
                
                # Ward analysis
                if 'geospatial_patterns' in analysis_results:
                    geo = analysis_results['geospatial_patterns']
                    if 'ward_distribution' in geo:
                        file_path = self.create_ward_analysis_map(geo['ward_distribution'])
                        if file_path:
                            generated_files['ward_analysis_map'] = file_path
            
            logger.info(f" Generated {len(generated_files)} visualization files")
            
            return generated_files
            
        except Exception as e:
            logger.error(f"L Error generating comprehensive report: {e}")
            return generated_files

def run_sample_visualization():
    """Run sample visualization for testing."""
    from src.databases.mongodb_handler import MongoDBHandler
    from src.analysis.eda import Chicago311EDA
    
    try:
        # Initialize handlers
        mongo_handler = MongoDBHandler()
        eda = Chicago311EDA(mongo_handler=mongo_handler)
        visualizer = Chicago311Visualizer()
        
        # Run analysis
        logger.info("Running EDA analysis...")
        results = eda.run_comprehensive_analysis()
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        files = visualizer.generate_comprehensive_report(results, mongo_handler)
        
        print("="*60)
        print("=Ê VISUALIZATION REPORT GENERATED")
        print("="*60)
        for name, filepath in files.items():
            print(f"=È {name}: {filepath}")
        print("="*60)
        
        mongo_handler.close()
        return files
        
    except Exception as e:
        logger.error(f"L Sample visualization failed: {e}")
        return None

if __name__ == "__main__":
    run_sample_visualization()