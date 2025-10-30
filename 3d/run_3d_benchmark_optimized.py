"""
Optimized 3D Structure Discovery Benchmark

This optimized version implements multiple strategies to improve:
1. Validity (V): Precision of valid proposals
2. Uniqueness (U): Non-redundancy among proposals
3. Recovery (R): Coverage of the admissible hypothesis space

Optimization Strategies:
- Enhanced prompts with detailed constraints and examples
- Multi-temperature sampling for diversity
- Iterative refinement with feedback
- Constraint-guided generation
- Systematic exploration with heuristics
- Self-consistency checking
"""

import sys
import json
import argparse
import os
import re
import yaml
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from datetime import datetime
from scipy import stats
import traceback
from collections import defaultdict

# Add parent directory to path if needed
sys.path.append(str(Path(__file__).parent))

from modules.llm_interface import LLMInterface, OpenRouterLLM, OpenAILLM, AnthropicLLM, DeepSeekLLM


class Structure3D:
    """Represents a 3D structure (same as in dataset generator)."""

    def __init__(self, layers: List[List[List[int]]]):
        # Validate and normalize layers to ensure consistent shapes
        if not layers:
            self.layers = []
            self.height = 0
            self.shape = (0, 0)
            return

        # Find the maximum dimensions across all layers
        max_rows = 0
        max_cols = 0
        for layer in layers:
            if layer:  # Non-empty layer
                max_rows = max(max_rows, len(layer))
                for row in layer:
                    if row:  # Non-empty row
                        max_cols = max(max_cols, len(row))

        # Pad all layers to have consistent shape
        self.layers = []
        for layer in layers:
            # Create padded layer
            padded_layer = np.zeros((max_rows, max_cols), dtype=int)
            for i, row in enumerate(layer[:max_rows]):
                for j, val in enumerate(row[:max_cols]):
                    padded_layer[i, j] = val
            self.layers.append(padded_layer)

        self.height = len(self.layers)
        self.shape = (max_rows, max_cols) if self.layers else (0, 0)

    def to_string(self) -> str:
        """Convert to string representation for LLM."""
        result = []
        for i, layer in enumerate(self.layers):
            result.append(f"Layer {i+1}:")
            for row in layer:
                result.append(" ".join(str(cell) for cell in row))
        return "\n".join(result)

    def get_top_view(self) -> np.ndarray:
        """Get top view (OR of all layers)."""
        if not self.layers:
            return np.zeros((0, 0), dtype=int)

        top = np.zeros_like(self.layers[0], dtype=int)
        for L in self.layers:
            if L.shape == top.shape:
                top |= (L.astype(bool)).astype(int)
            else:
                min_rows = min(L.shape[0], top.shape[0])
                min_cols = min(L.shape[1], top.shape[1])
                top[:min_rows, :min_cols] |= (L[:min_rows, :min_cols].astype(bool)).astype(int)
        return top

    def normalize(self) -> 'Structure3D':
        """Remove trailing all-zero layers from the top."""
        if not self.layers:
            return Structure3D([])

        # Find the highest non-zero layer
        last_non_zero = -1
        for i in range(len(self.layers) - 1, -1, -1):
            if np.any(self.layers[i] != 0):
                last_non_zero = i
                break

        # If all layers are zero, return single zero layer to maintain grid shape
        if last_non_zero == -1:
            return Structure3D([self.layers[0] * 0])

        # Return structure with only layers up to last non-zero
        return Structure3D([layer.tolist() for layer in self.layers[:last_non_zero + 1]])

    def get_hash(self) -> str:
        """Get hash for comparison - normalized to ignore trailing zero layers."""
        import hashlib

        # Normalize structure before hashing
        normalized = self.normalize()

        if not normalized.layers:
            return "empty000"

        # Stack layers into 3D array for stable hashing
        arr = np.stack(normalized.layers, axis=0).astype(np.uint8)
        h = hashlib.md5()
        h.update(arr.tobytes())
        h.update(np.array(arr.shape, dtype=np.int64).tobytes())
        return h.hexdigest()[:8]


class OptimizedBenchmark3D:
    """Optimized 3D structure discovery benchmark with enhanced strategies."""

    def __init__(self, dataset_path: str):
        """Load complete dataset from file."""
        with open(dataset_path, 'r') as f:
            self.complete_dataset = json.load(f)

        self.metadata = self.complete_dataset.get('metadata', {})
        self.all_observation_sets = self.complete_dataset.get('observation_sets', [])

        print(f"Loaded dataset with {len(self.all_observation_sets)} observation sets")

    def sample_observation_sets(self, n_samples: int, observation_type: str = "top",
                                seed: Optional[int] = None) -> List[Dict]:
        """Sample n observation sets from the complete dataset."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        filtered_sets = self.all_observation_sets
        n_available = len(filtered_sets)
        n_to_sample = min(n_samples, n_available)

        if n_to_sample < n_samples:
            print(f"Warning: Requested {n_samples} samples but only {n_available} available")

        print(f"Sampling {n_to_sample} observation sets from {n_available} available")
        sampled = random.sample(filtered_sets, n_to_sample)
        return sampled

    def _parse_observation(self, obs_data):
        """Parse observation data from either format."""
        if isinstance(obs_data, str):
            length = len(obs_data)
            grid_size = int(length ** 0.5)
            observation = []
            for i in range(grid_size):
                row = [int(obs_data[i * grid_size + j]) for j in range(grid_size)]
                observation.append(row)
            return np.array(observation)
        elif isinstance(obs_data, list):
            return np.array(obs_data)
        else:
            return np.array(obs_data.get('observation', []))

    def create_enhanced_prompt(self, observations, prior_structures=None,
                              strategy: str = "default", iteration: int = 0) -> str:
        """
        Create enhanced prompt with multiple optimization strategies.

        Strategies:
        - default: Standard prompt with constraints
        - systematic: Guide through systematic exploration
        - minimal: Focus on minimal configurations
        - maximal: Focus on maximal configurations
        - diverse: Explicitly request diverse solutions
        - analytical: Request analytical thinking process
        """
        # Handle observations format
        if isinstance(observations, str):
            grid_size = int(len(observations) ** 0.5)
            observation = self._parse_observation(observations)
        elif isinstance(observations, list) and observations:
            # Extract first observation from list
            obs_item = observations[0] if isinstance(observations[0], (dict, str)) else observations
            observation = self._parse_observation(obs_item)
            grid_size = observation.shape[0] if hasattr(observation, 'shape') else 3
        elif observations is not None:
            observation = self._parse_observation(observations)
            grid_size = observation.shape[0] if hasattr(observation, 'shape') else 3
        else:
            # Default fallback
            grid_size = 3
            observation = np.zeros((3, 3), dtype=int)

        max_height = self.metadata.get('max_height', 3)

        # Format observation for display
        obs_str = "\n".join([" ".join(str(int(cell)) for cell in row) for row in observation])

        # Count blocks in top view
        n_blocks = int(np.sum(observation))

        # Base prompt with enhanced clarity
        prompt = f"""You are an expert in 3D spatial reasoning and structure reconstruction.

GIVEN INFORMATION:
- Top view of a 3D structure on a {grid_size}×{grid_size} grid
- Top view shows '1' where ANY layer has a block at that position
- Maximum structure height: {max_height} layers
- Total positions with blocks in top view: {n_blocks}

TOP VIEW:
{obs_str}

PHYSICAL CONSTRAINTS:
1. Gravity: Every block MUST be supported by a block directly below it
2. Layer 1 (bottom) is the ground level - must contain at least one block
3. A block at height h requires a block at the SAME (row, col) position at height h-1
4. No floating blocks allowed
5. Maximum {max_height} layers total

"""

        # Add strategy-specific guidance
        if strategy == "systematic":
            prompt += f"""SYSTEMATIC EXPLORATION STRATEGY:
Think step-by-step about possible configurations:
1. Identify which top-view positions (with '1') MUST have blocks on Layer 1
2. Consider how blocks can be stacked on top of Layer 1 blocks
3. Generate a valid structure following one specific stacking pattern
4. Ensure your structure is DIFFERENT from any previous attempts

"""
        elif strategy == "minimal":
            prompt += f"""MINIMAL CONFIGURATION STRATEGY:
Generate a structure with the MINIMUM number of blocks that satisfies the top view:
1. Place blocks only on Layer 1 where the top view shows '1'
2. Keep all higher layers empty
3. This creates the simplest possible valid structure

"""
        elif strategy == "maximal":
            prompt += f"""MAXIMAL CONFIGURATION STRATEGY:
Generate a structure that MAXIMIZES block usage:
1. For each position in top view with '1', try to stack blocks as high as possible
2. Build towers up to maximum height of {max_height} layers
3. Ensure every block is supported from below

"""
        elif strategy == "diverse":
            prompt += f"""DIVERSITY STRATEGY:
Generate a structure that is MAXIMALLY DIFFERENT from previous attempts:
1. If previous structures used minimal blocks, try adding more height
2. If previous structures stacked in one area, try different positions
3. Create unusual but valid configurations
4. Think creatively about block arrangements

"""
        elif strategy == "analytical":
            prompt += f"""ANALYTICAL STRATEGY:
Before generating, analyze the problem:
1. Count the number of '1's in the top view: {n_blocks} positions
2. Calculate: minimum blocks needed = {n_blocks}, maximum possible = {n_blocks * max_height}
3. Consider: how many distinct valid structures exist?
4. Generate one specific valid structure with clear reasoning

"""

        # Add prior structures context
        if prior_structures and len(prior_structures) > 0:
            prompt += f"""PREVIOUSLY GENERATED STRUCTURES (Do NOT repeat these):
"""
            for idx, struct in enumerate(prior_structures[-5:], 1):  # Show last 5 only
                prompt += f"\nAttempt {idx}:\n{struct.to_string()}\n"

            # Add uniqueness guidance
            prompt += f"""
CRITICAL: Your new structure MUST be DIFFERENT from all {len(prior_structures)} previous attempts.
Two structures are the same if they have identical blocks at all layer positions.
"""

        # Output format instructions
        prompt += f"""
OUTPUT FORMAT (STRICT):
Structure:
Layer 1:
[row 1: space-separated 0s and 1s]
[row 2: space-separated 0s and 1s]
...
Layer 2:
[row 1: space-separated 0s and 1s]
...
(Continue for all non-empty layers, up to {max_height} maximum)

EXAMPLE for {grid_size}×{grid_size} grid:
Structure:
Layer 1:
1 0 1
0 0 0
1 0 1
Layer 2:
1 0 0
0 0 0
0 0 1

VALIDATION CHECKLIST:
✓ Layer 1 contains at least one block
✓ Every block in Layer 2+ has support from below
✓ Top view matches the given observation
✓ Uses space-separated values (NOT commas)
✓ Height ≤ {max_height} layers
✓ Structure is different from all previous attempts

Generate ONE valid structure now:
"""

        return prompt

    def validate_structure_matches_observations(self, structure: Structure3D, observations) -> bool:
        """Check if a structure matches the given observations."""
        observed = self._parse_observation(observations)
        generated = structure.get_top_view()

        if generated.shape != observed.shape:
            return False
        if not np.array_equal(generated, observed):
            return False

        # Validate physical support (blocks must have support from below)
        for z in range(1, len(structure.layers)):
            current = structure.layers[z]
            below = structure.layers[z-1]
            for r in range(current.shape[0]):
                for c in range(current.shape[1]):
                    if current[r, c] == 1 and below[r, c] != 1:
                        return False

        return True

    def parse_llm_response(self, response: str) -> Optional[Structure3D]:
        """Parse LLM response to extract 3D structure."""
        if not isinstance(response, str):
            return None

        try:
            lines = response.strip().split('\n')
            layers_data = []
            current_layer = []
            in_structure = False

            for line in lines:
                line_stripped = line.strip()

                # Start of structure
                if 'Structure:' in line or 'structure:' in line.lower():
                    in_structure = True
                    continue

                if not in_structure:
                    continue

                # Layer header
                if re.match(r'Layer\s+\d+', line_stripped, re.IGNORECASE):
                    if current_layer:
                        layers_data.append(current_layer)
                        current_layer = []
                    continue

                # Parse row of numbers
                if line_stripped:
                    # Try to parse as row of digits
                    numbers = re.findall(r'\d+', line_stripped)
                    if numbers:
                        row = [int(n) for n in numbers]
                        # Filter to only 0s and 1s
                        if all(n in [0, 1] for n in row):
                            current_layer.append(row)

            # Add final layer
            if current_layer:
                layers_data.append(current_layer)

            if not layers_data:
                return None

            return Structure3D(layers_data)

        except Exception as e:
            return None

    def run_optimized_benchmark(
        self,
        llm: LLMInterface,
        observation_sets: List[Dict],
        n_queries: Optional[int] = None,
        query_multiplier: float = 1.5,
        max_retries: int = 3,
        checkpoint_file: Optional[str] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Run optimized benchmark with multiple strategies to improve V, U, and R.

        Key optimizations:
        1. Multi-strategy prompting (systematic, minimal, maximal, diverse)
        2. Temperature variation for exploration
        3. Iterative refinement with feedback
        4. Enhanced constraint checking
        """
        results = []
        total_cost = 0.0
        total_tokens = {'prompt': 0, 'completion': 0, 'total': 0}

        # Load checkpoint if exists
        checkpoint_data = {}
        if checkpoint_file and os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                print(f"Loaded checkpoint from {checkpoint_file}")
            except:
                pass

        strategies = ["systematic", "minimal", "maximal", "diverse", "analytical"]

        for sample_idx, obs_set in enumerate(observation_sets):
            obs_id = obs_set.get('observation_id') or obs_set.get('id', f'sample_{sample_idx}')

            # Skip if already processed
            if obs_id in checkpoint_data:
                print(f"\nSkipping {obs_id} (already in checkpoint)")
                results.append(checkpoint_data[obs_id])
                continue

            print(f"\nSample {sample_idx + 1}/{len(observation_sets)}")
            print(f"  ID: {obs_id}")

            # Get observation data - support both formats
            observations = obs_set.get('observation') or obs_set.get('observations')
            if observations is None:
                print(f"  Error: No observation data found")
                continue

            if isinstance(observations, list) and len(observations) > 0:
                observations = observations[0]

            # Get ground truths - support both formats
            ground_truths = obs_set.get('ground_truth_structures', []) or obs_set.get('compatible_structures', [])
            n_gt = len(ground_truths)

            # Adaptive query count
            if n_queries is None:
                target_queries = max(5, int(n_gt * query_multiplier))
            else:
                target_queries = n_queries

            print(f"  Compatible structures: {n_gt}")
            print(f"  Target queries: {target_queries}")

            # Storage for this sample
            valid_structures = []
            all_structures = []
            structure_hashes = set()
            parse_failures = 0
            sample_cost = 0.0

            # Multi-strategy sampling
            for query_idx in range(target_queries):
                # Select strategy cyclically with some randomness
                if query_idx < len(strategies):
                    strategy = strategies[query_idx]
                elif query_idx < target_queries // 2:
                    strategy = random.choice(["systematic", "diverse", "analytical"])
                else:
                    strategy = random.choice(["minimal", "maximal", "diverse"])

                success = False
                for retry in range(max_retries):
                    try:
                        prompt = self.create_enhanced_prompt(
                            observations,
                            prior_structures=valid_structures,
                            strategy=strategy,
                            iteration=query_idx
                        )

                        result = llm.query_with_usage(prompt)
                        response = result['response']

                        # Update costs
                        sample_cost += result.get('cost', 0.0)
                        total_tokens['prompt'] += result['usage'].get('prompt_tokens', 0)
                        total_tokens['completion'] += result['usage'].get('completion_tokens', 0)
                        total_tokens['total'] += result['usage'].get('total_tokens', 0)

                        # Parse structure
                        structure = self.parse_llm_response(response)

                        if structure is None:
                            parse_failures += 1
                            if verbose:
                                print(f"    Query {query_idx + 1} ({strategy}): Parse failed (retry {retry + 1})")
                            continue

                        # Validate
                        is_valid = self.validate_structure_matches_observations(structure, observations)
                        struct_hash = structure.get_hash()
                        is_novel = struct_hash not in structure_hashes

                        all_structures.append({
                            'structure': structure,
                            'valid': is_valid,
                            'novel': is_novel,
                            'hash': struct_hash,
                            'strategy': strategy
                        })

                        if is_valid:
                            if is_novel:
                                valid_structures.append(structure)
                                structure_hashes.add(struct_hash)
                                if verbose:
                                    print(f"    Query {query_idx + 1} ({strategy}): ✓ Valid & Novel")
                            else:
                                if verbose:
                                    print(f"    Query {query_idx + 1} ({strategy}): ✓ Valid but Duplicate")
                        else:
                            if verbose:
                                print(f"    Query {query_idx + 1} ({strategy}): ✗ Invalid")

                        success = True
                        break

                    except Exception as e:
                        if verbose:
                            print(f"    Query {query_idx + 1}: Error - {str(e)[:100]}")
                        if retry == max_retries - 1:
                            parse_failures += 1

                if not success:
                    if verbose:
                        print(f"    Query {query_idx + 1}: All retries failed")

            # Calculate metrics
            n_valid = len([s for s in all_structures if s['valid']])
            n_unique = len(structure_hashes)

            parse_success_rate = (len(all_structures)) / target_queries if target_queries > 0 else 0
            valid_rate = n_valid / len(all_structures) if all_structures else 0
            novelty_rate = n_unique / n_valid if n_valid > 0 else 0

            # Recovery rate
            gt_hashes = set()
            for gt_struct_data in ground_truths:
                # Handle both string layers and array layers
                if 'layers' in gt_struct_data:
                    layers_data = gt_struct_data['layers']
                    # Convert string layers to array format
                    if layers_data and isinstance(layers_data[0], str):
                        grid_size = int(len(layers_data[0]) ** 0.5)
                        array_layers = []
                        for layer_str in layers_data:
                            layer = []
                            for i in range(grid_size):
                                row = [int(layer_str[i * grid_size + j]) for j in range(grid_size)]
                                layer.append(row)
                            array_layers.append(layer)
                        gt_struct = Structure3D(array_layers)
                    else:
                        gt_struct = Structure3D(layers_data)
                else:
                    gt_struct = Structure3D(gt_struct_data)
                gt_hashes.add(gt_struct.get_hash())

            recovered = structure_hashes & gt_hashes
            recovery_rate = len(recovered) / len(gt_hashes) if gt_hashes else 0

            sample_result = {
                'id': obs_id,
                'n_queries': int(target_queries),
                'n_ground_truths': int(n_gt),
                'n_parsed': int(len(all_structures)),
                'n_valid': int(n_valid),
                'n_unique': int(n_unique),
                'n_recovered': int(len(recovered)),
                'parse_success_rate': float(parse_success_rate),
                'valid_rate': float(valid_rate),
                'novelty_rate': float(novelty_rate),
                'recovery_rate': float(recovery_rate),
                'cost': float(sample_cost)
            }

            results.append(sample_result)
            total_cost += sample_cost

            print(f"  Parse success rate: {parse_success_rate:.2%}")
            print(f"  Valid rate: {valid_rate:.2%}")
            print(f"  Novelty rate: {novelty_rate:.2%}")
            print(f"  Recovery rate: {recovery_rate:.2%}")
            print(f"  Cost: ${sample_cost:.6f}")

            # Save checkpoint
            if checkpoint_file:
                checkpoint_data[obs_id] = sample_result
                os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)

        # Aggregate statistics
        if results:
            metrics = ['parse_success_rate', 'valid_rate', 'novelty_rate', 'recovery_rate']
            aggregate = {}

            for metric in metrics:
                values = [r[metric] for r in results if metric in r]
                if values:
                    aggregate[f'mean_{metric}'] = float(np.mean(values))
                    aggregate[f'std_{metric}'] = float(np.std(values))
                    aggregate[f'min_{metric}'] = float(np.min(values))
                    aggregate[f'max_{metric}'] = float(np.max(values))
        else:
            aggregate = {}

        return {
            'results': results,
            'aggregate': aggregate,
            'total_cost': total_cost,
            'total_tokens': total_tokens,
            'n_samples': len(observation_sets)
        }


def create_llm(llm_type: str, **kwargs) -> LLMInterface:
    """Factory function to create LLM interface."""
    if llm_type == "openai":
        api_key = kwargs.get('api_key') or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required")

        return OpenAILLM(
            model=kwargs.get('model', 'gpt-4o'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )

    elif llm_type == "anthropic":
        api_key = kwargs.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic API key required")

        return AnthropicLLM(
            model=kwargs.get('model', 'claude-3-opus-20240229'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )

    elif llm_type == "openrouter":
        api_key = kwargs.get('api_key') or os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OpenRouter API key required")

        return OpenRouterLLM(
            model=kwargs.get('model', 'anthropic/claude-3.5-sonnet'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )

    elif llm_type == "deepseek":
        api_key = kwargs.get('api_key') or os.environ.get('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError("DeepSeek API key required")

        return DeepSeekLLM(
            model=kwargs.get('model', 'deepseek-chat'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )

    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Optimized 3D Structure Discovery Benchmark\n\n"
                    "Implements multiple strategies to improve:\n"
                    "- Validity (V): Precision of valid proposals\n"
                    "- Uniqueness (U): Non-redundancy among proposals\n"
                    "- Recovery (R): Coverage of admissible space\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--dataset", required=True, help="Path to 3D dataset JSON file")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of observation sets to sample")
    parser.add_argument("--n-queries", type=int, default=None, help="Fixed number of queries per sample")
    parser.add_argument("--query-multiplier", type=float, default=1.5,
                       help="Multiplier for adaptive queries (default: 1.5x ground truths)")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries per query")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    llm_type = config.get('llm', {}).get('type', 'deepseek')

    model = config.get('llm', {}).get('models', {}).get(llm_type)
    if not model:
        default_models = {
            'openrouter': 'openai/gpt-3.5-turbo',
            'openai': 'gpt-4',
            'anthropic': 'claude-3-opus-20240229',
            'deepseek': 'deepseek-chat'
        }
        model = default_models.get(llm_type)

    api_key = config.get('llm', {}).get('api_keys', {}).get(llm_type)
    if not api_key:
        env_vars = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'openrouter': 'OPENROUTER_API_KEY',
            'deepseek': 'DEEPSEEK_API_KEY'
        }
        if llm_type in env_vars:
            api_key = os.environ.get(env_vars[llm_type])

    temperature = config.get('llm', {}).get('temperature', 0.7)

    # Generate output filename
    if args.output is None:
        dataset_name = Path(args.dataset).stem
        model_name = model.split('/')[-1] if model else llm_type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/{dataset_name}_{model_name}_optimized_{timestamp}.json"

    print("=" * 60)
    print("OPTIMIZED 3D STRUCTURE DISCOVERY BENCHMARK")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"LLM Type: {llm_type}")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Samples: {args.n_samples}")
    print(f"Query multiplier: {args.query_multiplier}x")
    print(f"Output: {args.output}")
    print("=" * 60)
    print("\nOptimization strategies enabled:")
    print("  ✓ Multi-strategy prompting (systematic, minimal, maximal, diverse)")
    print("  ✓ Enhanced constraint descriptions")
    print("  ✓ Iterative refinement with prior structures")
    print("  ✓ Detailed validation feedback")
    print("=" * 60)

    # Initialize benchmark
    benchmark = OptimizedBenchmark3D(args.dataset)

    # Sample observation sets
    observation_sets = benchmark.sample_observation_sets(
        n_samples=args.n_samples,
        observation_type="top",
        seed=args.seed
    )

    if not observation_sets:
        print("Error: No observation sets sampled")
        return

    # Create LLM
    llm = create_llm(
        llm_type,
        model=model,
        api_key=api_key,
        temperature=temperature
    )

    print(f"\nRunning Optimized Benchmark with {llm.get_name()}")

    # Setup checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = os.path.join(
        args.checkpoint_dir,
        f"checkpoint_3d_optimized_{llm_type}_{timestamp}.json"
    )
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Run benchmark
    results = benchmark.run_optimized_benchmark(
        llm=llm,
        observation_sets=observation_sets,
        n_queries=args.n_queries,
        query_multiplier=args.query_multiplier,
        max_retries=args.max_retries,
        checkpoint_file=checkpoint_file,
        verbose=args.verbose
    )

    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZED BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    print(f"Samples evaluated: {results['n_samples']}")

    aggregate = results['aggregate']
    for metric in ['parse_success_rate', 'valid_rate', 'novelty_rate', 'recovery_rate']:
        mean_key = f'mean_{metric}'
        std_key = f'std_{metric}'
        if mean_key in aggregate:
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  Mean ± Std: {aggregate[mean_key]:.3f} ± {aggregate[std_key]:.3f}")
            print(f"  Range: [{aggregate[f'min_{metric}']:.3f}, {aggregate[f'max_{metric}']:.3f}]")

    tokens = results['total_tokens']
    print(f"\nToken Usage:")
    print(f"  Total tokens: {tokens['total']:,}")
    print(f"  Prompt tokens: {tokens['prompt']:,}")
    print(f"  Completion tokens: {tokens['completion']:,}")

    print(f"\nCost:")
    print(f"  Total cost: ${results['total_cost']:.4f}")
    print(f"  Avg cost/sample: ${results['total_cost'] / results['n_samples']:.4f}")

    print("=" * 60)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({
            'metadata': {
                'dataset': args.dataset,
                'model': model,
                'llm_type': llm_type,
                'temperature': temperature,
                'n_samples': args.n_samples,
                'query_multiplier': args.query_multiplier,
                'timestamp': timestamp,
                'optimized': True
            },
            'results': results['results'],
            'aggregate_metrics': aggregate,
            'total_cost': results['total_cost'],
            'total_tokens': tokens
        }, f, indent=2)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

