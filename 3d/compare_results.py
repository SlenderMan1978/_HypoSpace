"""
Compare three experimental results: Original, Prompt Optimized, and Optimizer Enhanced
Metrics: Validity, Uniqueness (Novelty), Recovery
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_original_results(file_path):
    """Load original results (first JSON file)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stats = data['statistics']
    return {
        'validity': stats['valid_rate']['mean'],
        'uniqueness': stats['novelty_rate']['mean'],
        'recovery': stats['recovery_rate']['mean'],
        'validity_std': stats['valid_rate']['std'],
        'uniqueness_std': stats['novelty_rate']['std'],
        'recovery_std': stats['recovery_rate']['std']
    }

def load_prompt_optimized_results(file_path):
    """Load prompt-optimized results (second JSON file)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data['results']
    validity_rates = [r['valid_rate'] for r in results]
    uniqueness_rates = [r['novelty_rate'] for r in results]
    recovery_rates = [r['recovery_rate'] for r in results]

    return {
        'validity': np.mean(validity_rates),
        'uniqueness': np.mean(uniqueness_rates),
        'recovery': np.mean(recovery_rates),
        'validity_std': np.std(validity_rates),
        'uniqueness_std': np.std(uniqueness_rates),
        'recovery_std': np.std(recovery_rates)
    }

def load_optimizer_results(file_path):
    """Load optimizer-enhanced results (third JSON - includes ConstraintFixer, LocalSearchOptimizer, DiversityEnhancer)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Use aggregate_stats for averages
    aggregate = data['aggregate_stats']

    # Calculate standard deviation from per_observation_results
    per_obs = data['per_observation_results']
    validity_rates = [r['valid_rate'] for r in per_obs]
    uniqueness_rates = [r['novelty_rate'] for r in per_obs]
    recovery_rates = [r['recovery_rate'] for r in per_obs]

    return {
        'validity': aggregate['avg_valid_rate'],
        'uniqueness': aggregate['avg_novelty_rate'],
        'recovery': aggregate['avg_recovery_rate'],
        'validity_std': np.std(validity_rates),
        'uniqueness_std': np.std(uniqueness_rates),
        'recovery_std': np.std(recovery_rates)
    }

def create_comparison_chart(original, prompt_opt, optimizer):
    """Create comparison chart"""

    # Metric names
    metrics = ['Validity', 'Uniqueness', 'Recovery']

    # Data for three methods
    original_values = [original['validity'], original['uniqueness'], original['recovery']]
    prompt_opt_values = [prompt_opt['validity'], prompt_opt['uniqueness'], prompt_opt['recovery']]
    optimizer_values = [optimizer['validity'], optimizer['uniqueness'], optimizer['recovery']]

    # Standard deviations
    original_stds = [original['validity_std'], original['uniqueness_std'], original['recovery_std']]
    prompt_opt_stds = [prompt_opt['validity_std'], prompt_opt['uniqueness_std'], prompt_opt['recovery_std']]
    optimizer_stds = [optimizer['validity_std'], optimizer['uniqueness_std'], optimizer['recovery_std']]

    # Set up chart
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))

    # Create bar chart
    bars1 = ax.bar(x - width, original_values, width, label='Original',
                   yerr=original_stds, capsize=5, alpha=0.8, color='#3498db')
    bars2 = ax.bar(x, prompt_opt_values, width, label='Prompt Optimized',
                   yerr=prompt_opt_stds, capsize=5, alpha=0.8, color='#2ecc71')
    bars3 = ax.bar(x + width, optimizer_values, width,
                   label='Optimizer Enhanced\n(ConstraintFixer + LocalSearch + Diversity)',
                   yerr=optimizer_stds, capsize=5, alpha=0.8, color='#e74c3c')

    # Add value labels
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontsize=9)

    add_value_labels(bars1, original_values)
    add_value_labels(bars2, prompt_opt_values)
    add_value_labels(bars3, optimizer_values)

    # Set labels and title
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('3D Block World Hypothesis Generation Performance Comparison\n(DeepSeek-Chat Model)',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    # Save chart
    output_path = Path(__file__).parent / 'results' / 'performance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {output_path}")

    plt.show()

    return fig

def create_detailed_comparison_table(original, prompt_opt, optimizer):
    """Create detailed numerical comparison table"""
    print("\n" + "="*80)
    print("Detailed Performance Comparison")
    print("="*80)
    print(f"{'Metric':<20} {'Original':<25} {'Prompt Optimized':<25} {'Optimizer Enhanced':<25}")
    print("-"*80)

    metrics = [
        ('Validity', 'validity'),
        ('Uniqueness', 'uniqueness'),
        ('Recovery', 'recovery')
    ]

    for metric_name, metric_key in metrics:
        orig_val = original[metric_key]
        orig_std = original[f'{metric_key}_std']
        prompt_val = prompt_opt[metric_key]
        prompt_std = prompt_opt[f'{metric_key}_std']
        opt_val = optimizer[metric_key]
        opt_std = optimizer[f'{metric_key}_std']

        print(f"{metric_name:<20} {orig_val:.4f} ± {orig_std:.4f}    {prompt_val:.4f} ± {prompt_std:.4f}    {opt_val:.4f} ± {opt_std:.4f}")

    print("-"*80)
    print("\nImprovement:")
    print("-"*80)

    for metric_name, metric_key in metrics:
        orig_val = original[metric_key]
        prompt_val = prompt_opt[metric_key]
        opt_val = optimizer[metric_key]

        prompt_improvement = ((prompt_val - orig_val) / orig_val * 100) if orig_val > 0 else 0
        opt_improvement = ((opt_val - orig_val) / orig_val * 100) if orig_val > 0 else 0

        print(f"{metric_name:<20} Prompt Opt: {prompt_improvement:+.2f}%    Optimizer Enhanced: {opt_improvement:+.2f}%")

    print("="*80 + "\n")

def main():
    """Main function"""
    # File paths
    results_dir = Path(__file__).parent / 'results'

    original_file = results_dir / '3d_complete_deepseek-chat_20251030_130728.json'
    prompt_opt_file = results_dir / '3d_complete_deepseek-chat_optimized_20251030_132636.json'
    optimizer_file = results_dir / '3d_complete_deepseek-chat_with_optimizer_20251030_160648.json'

    # Check if files exist
    for file_path in [original_file, prompt_opt_file, optimizer_file]:
        if not file_path.exists():
            print(f"Error: File not found - {file_path}")
            return

    print("Loading result files...")

    # Load data
    original = load_original_results(original_file)
    prompt_opt = load_prompt_optimized_results(prompt_opt_file)
    optimizer = load_optimizer_results(optimizer_file)

    print("Data loaded successfully!\n")

    # Print detailed comparison table
    create_detailed_comparison_table(original, prompt_opt, optimizer)

    # Create comparison chart
    print("Generating comparison chart...")
    create_comparison_chart(original, prompt_opt, optimizer)

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()

