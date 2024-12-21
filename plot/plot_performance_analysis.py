import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import re

def extract_config(folder_name):
    """Extract configuration from folder name (r_X_tp_Y)"""
    match = re.match(r'r_(\d+)_tp_(\d+)', folder_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

def read_metrics(folder_path):
    """Read all metrics from a folder"""
    metrics = {}
    
    # Read CSV files
    csv_files = {
        'e2e_latency': 'request_e2e_time.csv',
        'scheduling_delay': 'request_scheduling_delay.csv',
        'execution_time': 'request_execution_time.csv',
        'preemption_time': 'request_preemption_time.csv',
        'batch_size': 'batch_size.csv',
        'prefill_e2e': 'prefill_e2e_time.csv',
    }
    
    for metric, filename in csv_files.items():
        file_path = folder_path / filename
        if file_path.exists():
            df = pd.read_csv(file_path)
            if not df.empty:
                metrics[metric] = df
    
    # Read GPU utilization from JSON files
    mfu_values = []
    for i in range(1, 9):  # Assuming maximum 8 replicas
        mfu_file = folder_path / f'replica_{i}_stage_1_mfu.json'
        if mfu_file.exists():
            with open(mfu_file) as f:
                mfu_data = json.load(f)
                mfu_values.append(float(mfu_data['value']))
    
    if mfu_values:
        metrics['gpu_utilization'] = sum(mfu_values) / len(mfu_values)
    
    return metrics

def collect_all_metrics():
    """Collect metrics from all experiment folders"""
    base_path = Path('simulator_output')
    all_data = []
    
    for folder in os.listdir(base_path):
        if folder in ['comparison_plots', 'old']:
            continue
            
        config = extract_config(folder)
        if not config:
            continue
            
        requests, thread_pool = config
        folder_path = base_path / folder / 'plots'
        
        if not folder_path.exists():
            continue
            
        metrics = read_metrics(folder_path)
        if not metrics:
            continue
            
        # Calculate summary statistics
        summary = {
            'requests': requests,
            'thread_pool': thread_pool,
            'folder': folder
        }
        
        # Process each metric
        for metric, data in metrics.items():
            if isinstance(data, pd.DataFrame):
                if 'value' in data.columns:
                    values = data['value']
                else:
                    values = data.iloc[:, -1]  # Take the last column
                    
                summary.update({
                    f'{metric}_p50': values.quantile(0.50),
                    f'{metric}_p95': values.quantile(0.95),
                    f'{metric}_p99': values.quantile(0.99),
                    f'{metric}_avg': values.mean()
                })
            else:
                summary[metric] = data
        
        # Calculate QPS
        if 'e2e_latency' in metrics:
            total_time = metrics['e2e_latency']['value'].max()
            summary['qps'] = len(metrics['e2e_latency']) / total_time
        
        all_data.append(summary)
    
    return pd.DataFrame(all_data)

def plot_performance_metrics():
    """Create comprehensive performance analysis plots"""
    df = collect_all_metrics()
    if df.empty:
        print("No data found to plot")
        return
    
    # Create multiple plots
    metrics_groups = [
        # Latency metrics
        {
            'title': 'End-to-End Latency Analysis',
            'metrics': [
                ('e2e_latency_p50', 'P50 E2E Latency (s)'),
                ('e2e_latency_p95', 'P95 E2E Latency (s)'),
                ('e2e_latency_p99', 'P99 E2E Latency (s)'),
                ('e2e_latency_avg', 'Average E2E Latency (s)')
            ]
        },
        # Scheduling and execution metrics
        {
            'title': 'Processing Time Breakdown',
            'metrics': [
                ('scheduling_delay_avg', 'Avg Scheduling Delay (s)'),
                ('execution_time_avg', 'Avg Execution Time (s)'),
                ('preemption_time_avg', 'Avg Preemption Time (s)'),
                ('prefill_e2e_avg', 'Avg Prefill Time (s)')
            ]
        }
    ]
    
    for group in metrics_groups:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(group['title'], fontsize=16)
        
        for (metric, title), ax in zip(group['metrics'], axes.flat):
            if metric in df.columns:
                for tp in df['thread_pool'].unique():
                    tp_data = df[df['thread_pool'] == tp]
                    ax.plot(tp_data['qps'], tp_data[metric], 
                           marker='o', label=f'TP={tp}', linewidth=2)
                
                ax.set_xlabel('QPS')
                ax.set_ylabel(title)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend(title='Thread Pool Size')
                ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path('simulator_output/comparison_plots')
        output_dir.mkdir(exist_ok=True)
        safe_title = group['title'].lower().replace(' ', '_')
        plt.savefig(output_dir / f'{safe_title}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create batch size and GPU utilization plot
    if 'batch_size_avg' in df.columns or 'gpu_utilization' in df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('System Efficiency Metrics', fontsize=16)
        
        if 'batch_size_avg' in df.columns:
            for tp in df['thread_pool'].unique():
                tp_data = df[df['thread_pool'] == tp]
                ax1.plot(tp_data['qps'], tp_data['batch_size_avg'],
                        marker='o', label=f'TP={tp}', linewidth=2)
            ax1.set_xlabel('QPS')
            ax1.set_ylabel('Average Batch Size')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend(title='Thread Pool Size')
            ax1.set_ylim(bottom=0)
        
        if 'gpu_utilization' in df.columns:
            for tp in df['thread_pool'].unique():
                tp_data = df[df['thread_pool'] == tp]
                ax2.plot(tp_data['qps'], tp_data['gpu_utilization'] * 100,
                        marker='o', label=f'TP={tp}', linewidth=2)
            ax2.set_xlabel('QPS')
            ax2.set_ylabel('GPU Utilization (%)')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend(title='Thread Pool Size')
            ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'system_efficiency_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    plot_performance_metrics() 