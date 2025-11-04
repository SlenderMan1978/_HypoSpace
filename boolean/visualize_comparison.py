#!/usr/bin/env python3
"""
Simple bar chart comparison for Boolean benchmark results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_latest_results(results_dir="results"):
    """Load the latest result files for each version"""
    results_data = {}
    for version in ['v1', 'v2', 'v3']:
        version_files = list(Path(results_dir).glob(f"boolean_benchmark_{version}_*.json"))
        if version_files:
            latest_file = max(version_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, 'r') as f:
                results_data[version] = json.load(f)
            print(f"Loaded {version}: {latest_file.name}")
        else:
            print(f"No results found for {version}")
    return results_data


def load_specific_files(v1_file=None, v2_file=None, v3_file=None):
    """Load specific result files"""
    results_data = {}
    if v1_file and Path(v1_file).exists():
        with open(v1_file, 'r') as f:
            results_data['v1'] = json.load(f)
    if v2_file and Path(v2_file).exists():
        with open(v2_file, 'r') as f:
            results_data['v2'] = json.load(f)
    if v3_file and Path(v3_file).exists():
        with open(v3_file, 'r') as f:
            results_data['v3'] = json.load(f)
    return results_data


def create_simple_bar_chart(results_data, save_path=None):
    """Create a simple bar chart comparing the three versions"""
    if not results_data:
        print("No results data to visualize!")
        return

    # Prepare data for plotting
    versions = []
    valid_rates = []
    novelty_rates = []
    recovery_rates = []

    for version in ['v1', 'v2', 'v3']:
        if version in results_data:
            data = results_data[version]
            versions.append(version.upper())
            valid_rates.append(data['statistics']['valid_rate']['mean'])
            novelty_rates.append(data['statistics']['novelty_rate']['mean'])
            recovery_rates.append(data['statistics']['recovery_rate']['mean'])

    if not versions:
        print("No valid versions found!")
        return

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Set bar positions and width
    x = np.arange(len(versions))
    width = 0.25

    # Create bars
    bars1 = plt.bar(x - width, valid_rates, width, label='Valid Rate',
                    color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = plt.bar(x, novelty_rates, width, label='Novelty Rate',
                    color='#A23B72', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars3 = plt.bar(x + width, recovery_rates, width, label='Recovery Rate',
                    color='#F18F01', alpha=0.8, edgecolor='black', linewidth=0.5)

    # Customize the chart
    plt.xlabel('Benchmark Version', fontsize=12, fontweight='bold')
    plt.ylabel('Performance Rate', fontsize=12, fontweight='bold')
    plt.title('Boolean Benchmark: Performance Comparison',
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x, versions)
    plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1)

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    # Adjust layout
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")

    plt.show()

    # Print simple comparison
    print("\n" + "=" * 50)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 50)
    for i, version in enumerate(versions):
        print(f"{version}: Valid={valid_rates[i]:.3f}, "
              f"Novelty={novelty_rates[i]:.3f}, "
              f"Recovery={recovery_rates[i]:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Simple bar chart comparison for Boolean benchmarks")
    parser.add_argument("--v1", help="Path to V1 result file")
    parser.add_argument("--v2", help="Path to V2 result file")
    parser.add_argument("--v3", help="Path to V3 result file")
    parser.add_argument("--results-dir", default="results", help="Results directory (default: results)")
    parser.add_argument("--save", help="Save chart to specified path")

    args = parser.parse_args()

    # Load results
    if args.v1 or args.v2 or args.v3:
        results_data = load_specific_files(args.v1, args.v2, args.v3)
    else:
        results_data = load_latest_results(args.results_dir)

    # Create chart
    create_simple_bar_chart(results_data, save_path=args.save)


if __name__ == "__main__":
    main()