import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib as mpl
import subprocess
import sys

def set_font():
    """Set plot fonts"""
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10

def plot_execution_times():
    set_font()

    # Get all folders under simulator_output directory
    base_path = Path('simulator_output')
    if not base_path.exists():
        print(f"Error: Directory {base_path} not found")
        return
        
    folders = [f for f in os.listdir(base_path) if os.path.isdir(base_path / f)]
    if not folders:
        print(f"Error: No folders found in {base_path}")
        return

    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Read and plot data for each folder
    data_found = False
    for folder in folders:
        if folder == 'comparison_plots':  # Skip the output folder
            continue
            
        print(f"Processing folder: {folder}")
        csv_path = base_path / folder / 'plots' / 'request_model_execution_time.csv'
        if csv_path.exists():
            try:
                print(f"Reading CSV file: {csv_path}")
                # Read CSV file
                df = pd.read_csv(csv_path)
                if not df.empty:
                    # Sort by execution time for proper CDF plot
                    df = df.sort_values('request_model_execution_time')
                    # Plot CDF
                    plt.plot(df['request_model_execution_time'], df['cdf'], 
                            label=folder, marker='', linewidth=2)
                    data_found = True
                    print(f"Successfully plotted data from {folder}")
                else:
                    print(f"Warning: File {csv_path} is empty")
            except Exception as e:
                print(f"Warning: Error reading file {csv_path}: {str(e)}")
        else:
            print(f"File not found: {csv_path}")
    
    if not data_found:
        print("Error: No data found to plot")
        return

    # Set plot properties
    plt.title('CDF of Request Model Execution Time')
    plt.xlabel('Execution Time (s)')
    plt.ylabel('CDF')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust legend position and layout
    if plt.gca().get_legend_handles_labels()[0]:
        plt.legend(title='Test Scenarios', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.subplots_adjust(right=0.85)
    
    # Save plot
    output_dir = base_path / 'comparison_plots'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'cdf_execution_time_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()

if __name__ == "__main__":
    plot_execution_times()
