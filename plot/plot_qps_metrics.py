import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

def extract_config(folder_name):
    """从文件夹名称中提取配置信息 (r_X_tp_Y)"""
    match = re.match(r'r_(\d+)_tp_(\d+)', folder_name)
    if match:
        return int(match.group(1)), int(match.group(2))  # requests, thread_pool
    return None

def collect_metrics():
    """收集所有文件夹中的指标数据"""
    base_path = Path('simulator_output')
    data = []
    
    for folder in os.listdir(base_path):
        if folder == 'comparison_plots' or folder == 'old':
            continue
            
        config = extract_config(folder)
        if not config:
            continue
            
        requests, thread_pool = config
        folder_path = base_path / folder / 'plots'
        
        # 读取执行时间数据
        time_file = folder_path / 'request_model_execution_time.csv'
        if time_file.exists():
            df = pd.read_csv(time_file)
            if not df.empty:
                # 计算各种延迟指标
                latencies = df['request_model_execution_time'].sort_values()
                p50 = latencies.quantile(0.50)
                avg = latencies.mean()
                
                # 计算QPS (假设总时间是最后一个请求的时间)
                total_time = df['request_model_execution_time'].max()
                qps = len(df) / total_time
                
                data.append({
                    'requests': requests,
                    'thread_pool': thread_pool,
                    'qps': qps,
                    'p50_latency': p50,
                    'avg_latency': avg
                })
    
    return pd.DataFrame(data)

def set_style():
    """设置全局绘图样式"""
    # 只使用 seaborn 的设置
    sns.set_theme(style="whitegrid", font_scale=1.2)
    sns.set_palette("husl")
    
    plt.rcParams.update({
        'figure.figsize': (16, 6),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.linewidth': 1.5
    })

def plot_latency_qps():
    """绘制QPS vs 延迟的关系图"""
    set_style()
    df = collect_metrics()
    if df.empty:
        print("No data found to plot")
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Latency vs QPS Analysis (Different Thread Pool Sizes)', 
                 fontsize=20, y=1.05, fontweight='bold')
    
    metrics = [
        ('p50_latency', 'P50 Latency (s)'),
        ('avg_latency', 'Average Latency (s)')
    ]
    
    colors = sns.color_palette("husl", n_colors=len(df['thread_pool'].unique()))
    
    for (metric, title), ax in zip(metrics, axes.flat):
        for i, tp in enumerate(sorted(df['thread_pool'].unique())):
            tp_data = df[df['thread_pool'] == tp]
            ax.plot(tp_data['qps'], tp_data[metric], 
                   marker='o', label=f'Thread Pool Size={tp}', 
                   linewidth=3, markersize=8,
                   color=colors[i])
        
        ax.set_xlabel('Queries Per Second (QPS)', fontweight='bold')
        ax.set_ylabel(title, fontweight='bold')
        
        # 优化网格线
        ax.grid(True, linestyle='--', alpha=0.3, which='both')
        ax.set_axisbelow(True)
        
        # 美化边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        # 优化图例
        ax.legend(title='Thread Pool Config', 
                 title_fontsize=12,
                 bbox_to_anchor=(1.02, 1),
                 loc='upper left',
                 borderaxespad=0)
        
        ax.set_ylim(bottom=0)
        ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    output_dir = Path('simulator_output/comparison_plots')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'qps_latency_analysis.png', 
                dpi=300, bbox_inches='tight',
                facecolor='white')
    print(f"Plot saved to: {output_dir / 'qps_latency_analysis.png'}")
    plt.close()

def plot_thread_pool_comparison():
    """为每个QPS级别绘制线程池大小与延迟的关系"""
    set_style()
    df = collect_metrics()
    if df.empty:
        print("No data found to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Thread Pool Size Impact on Latency at Different QPS', 
                 fontsize=20, y=1.05, fontweight='bold')
    
    metrics = [
        ('p50_latency', 'P50 Latency (s)'),
        ('avg_latency', 'Average Latency (s)')
    ]
    
    colors = sns.color_palette("husl", n_colors=len(df['requests'].unique()))
    
    for (metric, title), ax in zip(metrics, axes.flat):
        for i, req in enumerate(sorted(df['requests'].unique())):
            req_data = df[df['requests'] == req]
            ax.plot(req_data['thread_pool'], req_data[metric], 
                   marker='o', label=f'Requests={req}', 
                   linewidth=3, markersize=8,
                   color=colors[i])
        
        ax.set_xlabel('Thread Pool Size', fontweight='bold')
        ax.set_ylabel(title, fontweight='bold')
        
        # 优化网格线
        ax.grid(True, linestyle='--', alpha=0.3, which='both')
        ax.set_axisbelow(True)
        
        # 美化边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        # 优化图例
        ax.legend(title='Request Config', 
                 title_fontsize=12,
                 bbox_to_anchor=(1.02, 1),
                 loc='upper left',
                 borderaxespad=0)
        
        ax.set_ylim(bottom=0)
        ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    output_dir = Path('simulator_output/comparison_plots')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'thread_pool_impact_analysis.png', 
                dpi=300, bbox_inches='tight',
                facecolor='white')
    print(f"Plot saved to: {output_dir / 'thread_pool_impact_analysis.png'}")
    plt.close()

def plot_qps_avg_time():
    """绘制QPS vs Average Time的关系图"""
    set_style()
    
    # 收集QPS数据
    df_qps = collect_metrics()
    if df_qps.empty:
        print("No QPS data found to plot")
        return
        
    # 收集平均执行时间数据
    base_path = Path('simulator_output')
    avg_times = []
    
    for folder in os.listdir(base_path):
        if folder == 'comparison_plots' or folder == 'old':
            continue
            
        config = extract_config(folder)
        if not config:
            continue
            
        csv_path = base_path / folder / 'plots' / 'request_model_execution_time.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if not df.empty:
                avg_time = df['request_model_execution_time'].mean()
                requests, thread_pool = config
                avg_times.append({
                    'requests': requests,
                    'thread_pool': thread_pool,
                    'avg_time': avg_time
                })
    
    df_time = pd.DataFrame(avg_times)
    
    # 合并数据
    df = pd.merge(df_qps, df_time, on=['requests', 'thread_pool'])
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 为每个线程池大小绘制一条线
    colors = sns.color_palette("husl", n_colors=len(df['thread_pool'].unique()))
    
    for i, tp in enumerate(sorted(df['thread_pool'].unique())):
        tp_data = df[df['thread_pool'] == tp]
        ax.plot(tp_data['qps'], tp_data['avg_time'], 
               marker='o', label=f'Thread Pool Size={tp}', 
               linewidth=2, markersize=6,
               color=colors[i])
    
    # 设置图表属性
    ax.set_title('Average Execution Time vs QPS', 
                fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel('Queries Per Second (QPS)', fontweight='bold')
    ax.set_ylabel('Average Execution Time (s)', fontweight='bold')
    
    # 优化网格线
    ax.grid(True, linestyle='--', alpha=0.3, which='both')
    ax.set_axisbelow(True)
    
    # 美化边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    # 优化图例
    ax.legend(title='Thread Pool Config', 
             title_fontsize=12,
             bbox_to_anchor=(1.02, 1),
             loc='upper left',
             borderaxespad=0)
    
    ax.set_ylim(bottom=0)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path('simulator_output/comparison_plots')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'qps_avg_time_analysis.png', 
                dpi=300, bbox_inches='tight',
                facecolor='white')
    print(f"Plot saved to: {output_dir / 'qps_avg_time_analysis.png'}")
    plt.close()

if __name__ == "__main__":
    plot_latency_qps()
    plot_thread_pool_comparison()
    plot_qps_avg_time()  # 添加新的图表 