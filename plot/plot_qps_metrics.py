import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

def extract_config(folder_name):
    """从文件夹名称中提取配置信息 (r_X_tp_Y_qps_Z)"""
    match = re.match(r'r_(\d+)_tp_(\d+)_qps_(\d+)', folder_name)
    if match:
        return int(match.group(1)), int(match.group(2)), float(match.group(3))  # requests, tensor_parallel_size, qps
    return None

def collect_metrics():
    """收集所有文件夹中的指标数据"""
    base_path = Path('simulator_output/Meta-Llama-3-8B')
    data = []
    
    for folder in os.listdir(base_path):
        if folder == 'comparison_plots' or folder == 'old':
            continue
            
        config = extract_config(folder)
        if not config:
            continue
            
        requests, tensor_parallel_size, target_qps = config  # 改名
        folder_path = base_path / folder / 'plots'
        
        # 读取执行时间数据
        time_file = folder_path / 'request_model_execution_time.csv'
        if time_file.exists():
            df = pd.read_csv(time_file)
            if not df.empty:
                latencies = df['request_model_execution_time'].sort_values()
                p50 = latencies.quantile(0.50)
                avg = latencies.mean()
                
                data.append({
                    'requests': requests,
                    'tensor_parallel_size': tensor_parallel_size,  # 改名
                    'target_qps': target_qps,
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
    fig.suptitle('Latency vs QPS Analysis (Different Tensor Parallel Sizes)', 
                 fontsize=20, y=1.05, fontweight='bold')
    
    metrics = [
        ('p50_latency', 'P50 Latency (s)'),
        ('avg_latency', 'Average Latency (s)')
    ]
    
    colors = sns.color_palette("husl", n_colors=len(df['tensor_parallel_size'].unique()))
    
    for (metric, title), ax in zip(metrics, axes.flat):
        for i, tp in enumerate(sorted(df['tensor_parallel_size'].unique())):
            tp_data = df[df['tensor_parallel_size'] == tp]
            ax.plot(tp_data['target_qps'], tp_data[metric],
                   marker='o', label=f'Tensor Parallel Size={tp}', 
                   linewidth=3, markersize=8,
                   color=colors[i])
        
        ax.set_xlabel('Queries Per Second (QPS)', fontweight='bold')
        ax.set_ylabel(title, fontweight='bold')
        
        ax.grid(True, linestyle='--', alpha=0.3, which='both')
        ax.set_axisbelow(True)
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        ax.legend(title='Tensor Parallel Config', 
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

def plot_tensor_parallel_comparison():
    """为每个QPS级别绘制张量并行大小与延迟的关系"""
    set_style()
    df = collect_metrics()
    if df.empty:
        print("No data found to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Tensor Parallel Size Impact on Latency at Different QPS', 
                 fontsize=20, y=1.05, fontweight='bold')
    
    metrics = [
        ('p50_latency', 'P50 Latency (s)'),
        ('avg_latency', 'Average Latency (s)')
    ]
    
    colors = sns.color_palette("husl", n_colors=len(df['requests'].unique()))
    
    for (metric, title), ax in zip(metrics, axes.flat):
        for i, req in enumerate(sorted(df['requests'].unique())):
            req_data = df[df['requests'] == req]
            ax.plot(req_data['tensor_parallel_size'], req_data[metric], 
                   marker='o', label=f'Requests={req}', 
                   linewidth=3, markersize=8,
                   color=colors[i])
        
        ax.set_xlabel('Tensor Parallel Size', fontweight='bold')
        ax.set_ylabel(title, fontweight='bold')
        
        ax.grid(True, linestyle='--', alpha=0.3, which='both')
        ax.set_axisbelow(True)
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
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
    plt.savefig(output_dir / 'tensor_parallel_impact_analysis.png', 
                dpi=300, bbox_inches='tight',
                facecolor='white')
    print(f"Plot saved to: {output_dir / 'tensor_parallel_impact_analysis.png'}")
    plt.close()

def plot_qps_avg_time():
    """绘制不同配置下的QPS vs Average Time关系图"""
    set_style()
    
    df_qps = collect_metrics()
    if df_qps.empty:
        print("No QPS data found to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 获取所有唯一的配置组合
    configs = df_qps[['requests', 'tensor_parallel_size']].drop_duplicates()
    colors = sns.color_palette("husl", n_colors=len(configs))
    
    # 为每个配置绘制一条曲线
    for idx, (_, config) in enumerate(configs.iterrows()):
        r = config['requests']
        tp = config['tensor_parallel_size']
        
        # 获取该配置下的所有数据点
        mask = (df_qps['requests'] == r) & (df_qps['tensor_parallel_size'] == tp)
        data = df_qps[mask].sort_values('target_qps')
        
        # 绘制曲线
        ax.plot(data['target_qps'], data['avg_latency'],
               marker='o',
               label=f'R={r}, TP={tp}',
               linewidth=2,
               markersize=6,
               color=colors[idx])
    
    # 设置图表属性
    ax.set_title('Average Execution Time vs QPS for Different Configurations', 
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
    ax.legend(title='Configuration (Replica, Tensor Parallel)', 
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
    plot_tensor_parallel_comparison()
    plot_qps_avg_time()