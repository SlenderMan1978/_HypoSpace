"""
3D Benchmark Result Comparison Visualization Tool

Reads original and optimized result JSON files, generates comparison charts
for Validity, Uniqueness, and Recovery metrics.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import warnings
import matplotlib.font_manager as fm

# 抑制matplotlib字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Configure Chinese font (for compatibility, though not needed for English-only charts)
def setup_chinese_font():
    """Configure matplotlib font for better compatibility"""
    import sys
    import os

    # Windows font directory
    font_dirs = [
        r'C:\Windows\Fonts',
        os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
    ]

    # Try font files (in priority order)
    font_files = [
        'msyh.ttc',      # Microsoft YaHei
        'msyhbd.ttc',    # Microsoft YaHei Bold
        'simhei.ttf',    # SimHei
        'simsun.ttc',    # SimSun
        'simkai.ttf',    # KaiTi
    ]

    font_path = None
    font_name = None

    # Find available font files
    for font_dir in font_dirs:
        if not os.path.exists(font_dir):
            continue
        for font_file in font_files:
            full_path = os.path.join(font_dir, font_file)
            if os.path.exists(full_path):
                font_path = full_path
                font_name = font_file.split('.')[0]
                break
        if font_path:
            break

    if font_path:
        # Add font to matplotlib
        try:
            from matplotlib.font_manager import fontManager
            fontManager.addfont(font_path)
            # Get font family name
            font_prop = fm.FontProperties(fname=font_path)
            font_family = font_prop.get_name()

            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [font_family, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

            print(f"✓ Successfully loaded font: {font_family} ({font_name})")
            return True
        except Exception as e:
            print(f"Font loading failed: {e}")

    # If no font file found, try system registered fonts
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
    available_fonts = set(f.name for f in fm.fontManager.ttflist)

    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✓ Using system font: {font}")
            return True

    # Final fallback
    print("⚠ Warning: No suitable font found, using default")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return False

# 设置字体
setup_chinese_font()
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_results(file_path):
    """Load result JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_metrics_original(data):
    """Extract metrics from original version results"""
    stats = data.get('statistics', {})

    metrics = {
        'valid_rate': {
            'mean': stats.get('valid_rate', {}).get('mean', 0),
            'std': stats.get('valid_rate', {}).get('std', 0),
            'values': []
        },
        'novelty_rate': {
            'mean': stats.get('novelty_rate', {}).get('mean', 0),
            'std': stats.get('novelty_rate', {}).get('std', 0),
            'values': []
        },
        'recovery_rate': {
            'mean': stats.get('recovery_rate', {}).get('mean', 0),
            'std': stats.get('recovery_rate', {}).get('std', 0),
            'values': []
        }
    }

    # Extract values from each sample
    if 'sample_results' in data:
        for sample in data['sample_results']:
            metrics['valid_rate']['values'].append(sample.get('valid_rate', 0))
            metrics['novelty_rate']['values'].append(sample.get('novelty_rate', 0))
            metrics['recovery_rate']['values'].append(sample.get('recovery_rate', 0))

    return metrics


def extract_metrics_optimized(data):
    """Extract metrics from optimized version results"""
    # Get from aggregate_metrics
    agg = data.get('aggregate_metrics', {})

    metrics = {
        'valid_rate': {
            'mean': agg.get('mean_valid_rate', 0),
            'std': agg.get('std_valid_rate', 0),
            'values': []
        },
        'novelty_rate': {
            'mean': agg.get('mean_novelty_rate', 0),
            'std': agg.get('std_novelty_rate', 0),
            'values': []
        },
        'recovery_rate': {
            'mean': agg.get('mean_recovery_rate', 0),
            'std': agg.get('std_recovery_rate', 0),
            'values': []
        }
    }

    # Extract values from each sample
    if 'results' in data:
        for sample in data['results']:
            metrics['valid_rate']['values'].append(sample.get('valid_rate', 0))
            metrics['novelty_rate']['values'].append(sample.get('novelty_rate', 0))
            metrics['recovery_rate']['values'].append(sample.get('recovery_rate', 0))

    return metrics


def create_comparison_plot(original_metrics, optimized_metrics, output_path='comparison_plot.png'):
    """Create comparison plot"""

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('3D Benchmark: Original vs Optimized Performance Comparison', fontsize=20, fontweight='bold', y=0.995)

    metrics_names = ['valid_rate', 'novelty_rate', 'recovery_rate']
    metrics_labels = ['Validity', 'Uniqueness', 'Recovery']
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    # 1. Bar chart comparison (top left)
    ax1 = axes[0, 0]
    x = np.arange(len(metrics_names))
    width = 0.35

    original_means = [original_metrics[m]['mean'] for m in metrics_names]
    optimized_means = [optimized_metrics[m]['mean'] for m in metrics_names]
    original_stds = [original_metrics[m]['std'] for m in metrics_names]
    optimized_stds = [optimized_metrics[m]['std'] for m in metrics_names]

    bars1 = ax1.bar(x - width/2, original_means, width, label='Original',
                    yerr=original_stds, capsize=5, alpha=0.8, color='#95a5a6')
    bars2 = ax1.bar(x + width/2, optimized_means, width, label='Optimized',
                    yerr=optimized_stds, capsize=5, alpha=0.8, color='#27ae60')

    ax1.set_ylabel('Rate', fontsize=12)
    ax1.set_title('Average Performance Comparison (with Std Dev)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_labels)
    ax1.legend(fontsize=11)
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom', fontsize=9)

    # 2. Improvement percentage chart (top right)
    ax2 = axes[0, 1]
    improvements = [(optimized_means[i] - original_means[i]) / original_means[i] * 100
                   if original_means[i] > 0 else 0
                   for i in range(len(metrics_names))]

    bars = ax2.barh(metrics_labels, improvements, color=colors, alpha=0.8)
    ax2.set_xlabel('Improvement (%)', fontsize=12)
    ax2.set_title('Relative Improvement', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
        symbol = '↑' if val > 0 else '↓' if val < 0 else '→'
        ax2.text(val, bar.get_y() + bar.get_height()/2,
                f' {symbol} {val:+.1f}%',
                va='center', fontweight='bold', color=color, fontsize=11)

    # 3. Box plot comparison (bottom left)
    ax3 = axes[1, 0]

    box_data = []
    labels = []
    positions = []
    pos = 1

    for i, (metric, label) in enumerate(zip(metrics_names, metrics_labels)):
        if original_metrics[metric]['values']:
            box_data.append(original_metrics[metric]['values'])
            labels.append(f'{label}\nOriginal')
            positions.append(pos)
            pos += 1

        if optimized_metrics[metric]['values']:
            box_data.append(optimized_metrics[metric]['values'])
            labels.append(f'{label}\nOptimized')
            positions.append(pos)
            pos += 1

        pos += 0.5  # Add spacing

    bp = ax3.boxplot(box_data, positions=positions, tick_labels=labels,
                     patch_artist=True, widths=0.6)

    # Set box plot colors
    for i, patch in enumerate(bp['boxes']):
        if 'Original' in labels[i]:
            patch.set_facecolor('#95a5a6')
            patch.set_alpha(0.6)
        else:
            patch.set_facecolor('#27ae60')
            patch.set_alpha(0.8)

    ax3.set_ylabel('Rate', fontsize=12)
    ax3.set_title('Data Distribution Comparison (Box Plot)', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 1.0])
    ax3.grid(axis='y', alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0, fontsize=9)

    # 4. Detailed statistics table (bottom right)
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')

    # Prepare table data
    table_data = [
        ['Metric', 'Original\nMean±Std', 'Optimized\nMean±Std', 'Absolute\nImprovement', 'Relative\nImprovement']
    ]

    for metric, label in zip(metrics_names, metrics_labels):
        orig_mean = original_metrics[metric]['mean']
        orig_std = original_metrics[metric]['std']
        opt_mean = optimized_metrics[metric]['mean']
        opt_std = optimized_metrics[metric]['std']

        abs_improve = opt_mean - orig_mean
        rel_improve = (abs_improve / orig_mean * 100) if orig_mean > 0 else 0

        table_data.append([
            label,
            f'{orig_mean:.2%}\n±{orig_std:.2%}',
            f'{opt_mean:.2%}\n±{opt_std:.2%}',
            f'{abs_improve:+.2%}',
            f'{rel_improve:+.1f}%'
        ])

    # Create table
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.22, 0.22, 0.15, 0.16])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Set header style
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')

    # Set data row colors
    for i in range(1, 4):
        for j in range(5):
            cell = table[(i, j)]
            if j == 0:
                cell.set_facecolor('#ecf0f1')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('white')

            # Add color for improvement columns
            if j >= 3:
                text = table_data[i][j]
                if '+' in text:
                    cell.set_facecolor('#d5f4e6')
                elif '-' in text:
                    cell.set_facecolor('#fadbd8')

    ax4.set_title('Detailed Statistics', fontsize=14, fontweight='bold', pad=20)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Chart saved to: {output_path}")

    return fig


def print_summary(original_metrics, optimized_metrics):
    """Print comparison summary"""
    print("\n" + "="*70)
    print("3D Benchmark Performance Comparison Summary".center(70))
    print("="*70)

    metrics_names = ['valid_rate', 'novelty_rate', 'recovery_rate']
    metrics_labels = ['Validity', 'Uniqueness', 'Recovery']

    for metric, label in zip(metrics_names, metrics_labels):
        orig = original_metrics[metric]['mean']
        opt = optimized_metrics[metric]['mean']
        diff = opt - orig
        rel_diff = (diff / orig * 100) if orig > 0 else 0

        print(f"\n{label}:")
        print(f"  Original:  {orig:.2%} ± {original_metrics[metric]['std']:.2%}")
        print(f"  Optimized: {opt:.2%} ± {optimized_metrics[metric]['std']:.2%}")

        symbol = "↑" if diff > 0 else "↓" if diff < 0 else "→"
        color_code = "\033[92m" if diff > 0 else "\033[91m" if diff < 0 else "\033[93m"
        reset_code = "\033[0m"

        print(f"  {color_code}Improvement: {diff:+.2%} ({symbol} {rel_diff:+.1f}%){reset_code}")

    print("\n" + "="*70)

    # Calculate overall improvement
    total_orig = sum(original_metrics[m]['mean'] for m in metrics_names) / len(metrics_names)
    total_opt = sum(optimized_metrics[m]['mean'] for m in metrics_names) / len(metrics_names)
    total_diff = total_opt - total_orig
    total_rel = (total_diff / total_orig * 100) if total_orig > 0 else 0

    print(f"\nAverage Performance:")
    print(f"  Original:  {total_orig:.2%}")
    print(f"  Optimized: {total_opt:.2%}")
    print(f"  Improvement: {total_diff:+.2%} ({total_rel:+.1f}%)")
    print("="*70 + "\n")


def main():
    """Main function"""
    # Define file paths
    results_dir = Path(__file__).parent / 'results'

    # Auto-find result files
    original_file = None
    optimized_file = None

    for file in results_dir.glob('*.json'):
        if 'optimized' in file.name:
            optimized_file = file
        else:
            original_file = file

    if not original_file:
        print("❌ Error: Original result file not found")
        return

    if not optimized_file:
        print("❌ Error: Optimized result file not found")
        return

    print(f"\nReading result files...")
    print(f"  Original:  {original_file.name}")
    print(f"  Optimized: {optimized_file.name}")

    # Load data
    original_data = load_results(original_file)
    optimized_data = load_results(optimized_file)

    # Extract metrics
    print("\nExtracting performance metrics...")
    original_metrics = extract_metrics_original(original_data)
    optimized_metrics = extract_metrics_optimized(optimized_data)

    # Print summary
    print_summary(original_metrics, optimized_metrics)

    # Create visualization
    print("Generating comparison chart...")
    output_path = results_dir / 'performance_comparison.png'
    create_comparison_plot(original_metrics, optimized_metrics, output_path)

    # Display chart
    plt.show()

    print("\n✓ Done!")


if __name__ == "__main__":
    main()

