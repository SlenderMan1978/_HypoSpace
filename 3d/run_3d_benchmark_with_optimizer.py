"""
3D Benchmark with Algorithmic Optimization
集成了约束修复、局部搜索和多样性增强算法的3D重建基准测试
不依赖提示词工程，通过后处理算法优化LLM输出
"""

import sys
import json
import argparse
import os
import yaml
import random
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from datetime import datetime
import traceback

# Add parent directory to path if needed
sys.path.append(str(Path(__file__).parent))

from modules.llm_interface import LLMInterface, OpenRouterLLM, OpenAILLM, AnthropicLLM, DeepSeekLLM


class Structure3D:
    """Represents a 3D structure."""

    def __init__(self, layers: List[List[List[int]]]):
        if not layers:
            self.layers = []
            self.height = 0
            self.shape = (0, 0)
            return

        # Find the maximum dimensions across all layers
        max_rows = 0
        max_cols = 0
        for layer in layers:
            if layer:
                max_rows = max(max_rows, len(layer))
                for row in layer:
                    if row:
                        max_cols = max(max_cols, len(row))

        # Pad all layers to have consistent shape
        self.layers = []
        for layer in layers:
            padded_layer = np.zeros((max_rows, max_cols), dtype=int)
            for i, row in enumerate(layer[:max_rows]):
                for j, val in enumerate(row[:max_cols]):
                    padded_layer[i, j] = val
            self.layers.append(padded_layer)

        self.height = len(self.layers)
        self.shape = (max_rows, max_cols) if self.layers else (0, 0)

    def to_string(self) -> str:
        """Convert to string representation."""
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

        if last_non_zero == -1:
            return Structure3D([self.layers[0] * 0])

        return Structure3D([layer.tolist() for layer in self.layers[:last_non_zero + 1]])

    def get_hash(self) -> str:
        """Get hash for comparison."""
        import hashlib

        normalized = self.normalize()

        if not normalized.layers:
            return "empty000"

        arr = np.stack(normalized.layers, axis=0).astype(np.uint8)
        h = hashlib.md5()
        h.update(arr.tobytes())
        h.update(np.array(arr.shape, dtype=np.int64).tobytes())
        return h.hexdigest()[:8]


class ConstraintFixer:
    """修复LLM生成的无效结构，使其满足物理和投影约束"""

    def fix_structure(self, structure: Structure3D, observations) -> Optional[Structure3D]:
        """修复结构使其满足约束"""
        # 1. 修复物理约束(重力支撑)
        fixed_structure = self._fix_gravity(structure)

        # 2. 修复投影约束
        fixed_structure = self._fix_projections(fixed_structure, observations)

        return fixed_structure

    def _fix_gravity(self, structure: Structure3D) -> Structure3D:
        """确保所有方块有支撑"""
        new_layers = [layer.copy() for layer in structure.layers]

        # 从下往上检查
        for z in range(1, len(new_layers)):
            current = new_layers[z]
            below = new_layers[z-1]

            for x in range(current.shape[0]):
                for y in range(current.shape[1]):
                    # 如果当前层有方块但下层没有,移除方块
                    if current[x, y] == 1 and below[x, y] == 0:
                        current[x, y] = 0

        return Structure3D([layer.tolist() for layer in new_layers])

    def _fix_projections(self, structure: Structure3D, observations) -> Structure3D:
        """调整结构以匹配投影"""
        observed = self._parse_observation(observations)
        new_layers = [layer.copy() for layer in structure.layers]

        # 获取当前顶视图
        top_view = structure.get_top_view()

        # 找出不匹配的位置
        for x in range(observed.shape[0]):
            for y in range(observed.shape[1]):
                if observed[x, y] == 1 and top_view[x, y] == 0:
                    # 需要添加方块到底层
                    new_layers[0][x, y] = 1
                elif observed[x, y] == 0 and top_view[x, y] == 1:
                    # 需要移除所有层的方块
                    for z in range(len(new_layers)):
                        new_layers[z][x, y] = 0

        return Structure3D([layer.tolist() for layer in new_layers])

    def _parse_observation(self, obs):
        """解析观测"""
        if isinstance(obs, str):
            length = len(obs)
            grid_size = int(length ** 0.5)
            observation = []
            for i in range(grid_size):
                row = [int(obs[i * grid_size + j]) for j in range(grid_size)]
                observation.append(row)
            return np.array(observation)
        return np.array(obs)


class LocalSearchOptimizer:
    """从LLM输出开始进行局部搜索，生成结构变体"""

    def optimize_with_local_search(
        self,
        observations,
        initial_structures: List[Structure3D],
        n_expansions: int = 20,
        validator=None
    ) -> List[Structure3D]:
        """从LLM输出开始局部搜索"""

        all_structures = initial_structures.copy()
        unique_hashes = {s.get_hash() for s in initial_structures}

        # 对每个LLM生成的结构进行局部变异
        for struct in initial_structures:
            variants = self._generate_variants(struct, n_expansions)

            for variant in variants:
                # 检查有效性
                if validator and validator(variant, observations):
                    v_hash = variant.get_hash()
                    if v_hash not in unique_hashes:
                        unique_hashes.add(v_hash)
                        all_structures.append(variant)

        return all_structures

    def _generate_variants(self, structure: Structure3D, n: int) -> List[Structure3D]:
        """生成结构的局部变体"""
        variants = []

        for _ in range(n):
            new_layers = [layer.copy() for layer in structure.layers]

            # 随机选择1-2个位置进行变异
            n_mutations = random.randint(1, 2)

            for _ in range(n_mutations):
                z = random.randint(0, len(new_layers) - 1)
                x = random.randint(0, new_layers[z].shape[0] - 1)
                y = random.randint(0, new_layers[z].shape[1] - 1)

                # 翻转该位置
                if new_layers[z][x, y] == 0:
                    # 检查是否有支撑
                    if z == 0 or new_layers[z-1][x, y] == 1:
                        new_layers[z][x, y] = 1
                else:
                    # 检查是否支撑上层
                    supported_above = False
                    if z < len(new_layers) - 1:
                        supported_above = new_layers[z+1][x, y] == 1

                    if not supported_above:
                        new_layers[z][x, y] = 0

            variant = Structure3D([layer.tolist() for layer in new_layers])
            variants.append(variant.normalize())

        return variants


class DiversityEnhancer:
    """增强LLM输出的多样性"""

    def enhance_diversity(
        self,
        structures: List[Structure3D],
        observations,
        target_count: int = 30,
        validator=None
    ) -> List[Structure3D]:
        """通过插值和变异增加多样性"""

        if len(structures) >= target_count:
            return structures

        enhanced = structures.copy()
        unique_hashes = {s.get_hash() for s in structures}

        # 策略1: 结构间插值
        max_interpolation_attempts = min(len(structures) * (len(structures) - 1) // 2, 50)
        attempts = 0
        for i in range(len(structures)):
            for j in range(i + 1, len(structures)):
                if len(enhanced) >= target_count or attempts >= max_interpolation_attempts:
                    break

                attempts += 1
                interpolated = self._interpolate(structures[i], structures[j])
                if interpolated and validator and validator(interpolated, observations):
                    h = interpolated.get_hash()
                    if h not in unique_hashes:
                        unique_hashes.add(h)
                        enhanced.append(interpolated)

        # 策略2: 随机变异
        max_mutation_attempts = 100
        attempts = 0
        while len(enhanced) < target_count and attempts < max_mutation_attempts:
            attempts += 1
            base = random.choice(structures)
            mutated = self._mutate(base)

            if mutated and validator and validator(mutated, observations):
                h = mutated.get_hash()
                if h not in unique_hashes:
                    unique_hashes.add(h)
                    enhanced.append(mutated)

        return enhanced

    def _interpolate(self, s1: Structure3D, s2: Structure3D) -> Optional[Structure3D]:
        """两个结构的插值"""
        if len(s1.layers) != len(s2.layers):
            return None

        new_layers = []
        for layer1, layer2 in zip(s1.layers, s2.layers):
            # 随机选择每个位置的值
            new_layer = np.where(
                np.random.random(layer1.shape) > 0.5,
                layer1,
                layer2
            )
            new_layers.append(new_layer)

        return Structure3D([layer.tolist() for layer in new_layers]).normalize()

    def _mutate(self, structure: Structure3D) -> Structure3D:
        """随机变异"""
        new_layers = [layer.copy() for layer in structure.layers]

        # 随机改变1-3个位置
        n_mutations = random.randint(1, 3)

        for _ in range(n_mutations):
            z = random.randint(0, len(new_layers) - 1)
            x = random.randint(0, new_layers[z].shape[0] - 1)
            y = random.randint(0, new_layers[z].shape[1] - 1)

            # 尝试翻转，保持物理约束
            if new_layers[z][x, y] == 0:
                if z == 0 or new_layers[z-1][x, y] == 1:
                    new_layers[z][x, y] = 1
            else:
                # 检查上层是否依赖此方块
                can_remove = True
                if z < len(new_layers) - 1:
                    if new_layers[z+1][x, y] == 1:
                        can_remove = False
                if can_remove:
                    new_layers[z][x, y] = 0

        return Structure3D([layer.tolist() for layer in new_layers]).normalize()


class Benchmark3DWithOptimizer:
    """集成算法优化的3D结构发现基准测试"""

    def __init__(self, dataset_path: str):
        """Load complete dataset from file."""
        with open(dataset_path, 'r') as f:
            self.complete_dataset = json.load(f)

        self.metadata = self.complete_dataset.get('metadata', {})
        self.all_observation_sets = self.complete_dataset.get('observation_sets', [])

        # 初始化优化器组件
        self.fixer = ConstraintFixer()
        self.local_search = LocalSearchOptimizer()
        self.diversity_enhancer = DiversityEnhancer()

        print(f"Loaded dataset with {len(self.all_observation_sets)} observation sets")
        print(f"Optimization algorithms initialized: ConstraintFixer, LocalSearch, DiversityEnhancer")

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

        if n_available == 0:
            print(f"Error: No observation sets found in dataset")
            return []

        print(f"Sampling {n_to_sample} observation sets from {n_available} available")
        sampled = random.sample(filtered_sets, n_to_sample)
        return sampled

    def _parse_observation(self, obs_data):
        """Parse observation data."""
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

    def create_prompt(self, observations, prior_structures=None) -> str:
        """Create prompt for LLM."""
        # Determine grid size
        if isinstance(observations, str):
            grid_size = int(len(observations) ** 0.5)
        elif isinstance(observations, list) and observations:
            if isinstance(observations[0], dict):
                obs_data = observations[0].get('observation', [])
                grid_size = len(obs_data) if obs_data else 3
            else:
                obs_data = self._parse_observation(observations[0])
                grid_size = obs_data.shape[0] if hasattr(obs_data, 'shape') else 3
        else:
            obs_data = self._parse_observation(observations)
            grid_size = obs_data.shape[0] if hasattr(obs_data, 'shape') else 3

        max_height = self.metadata.get('max_height', 3)

        prompt = f"""You are given observations of a 3D structure made of unit blocks on a {grid_size}x{grid_size} grid.
The maximum height is {max_height} layers.

Observations (Top View):
"""

        if isinstance(observations, str):
            observation = self._parse_observation(observations)
            prompt += f"\nTop view:\n"
            for row in observation:
                prompt += " ".join(str(cell) for cell in row) + "\n"
        else:
            observation = self._parse_observation(observations)
            prompt += f"\nTop view:\n"
            for row in observation:
                prompt += " ".join(str(cell) for cell in row) + "\n"

        if prior_structures and len(prior_structures) > 0:
            prompt += "\n\nPreviously generated structures (generate something different):\n"
            for idx, struct in enumerate(prior_structures[:3], 1):  # Show max 3
                prompt += f"\nStructure {idx}:\n"
                prompt += struct.to_string() + "\n"

        prompt += f"""

Task: Generate a 3D structure that produces these observations.

Constraints:
1. Grid: {grid_size}x{grid_size}, max height: {max_height} layers
2. Layer 1 is the BOTTOM layer
3. Blocks must be supported (block at height h needs support at height h-1)
4. Use spaces between digits

Output format:
Structure:
Layer 1:
1 0 1
0 0 0
1 0 1
Layer 2:
1 0 0
0 0 0
0 0 1
"""

        return prompt

    def validate_structure_matches_observations(self, structure: Structure3D, observations) -> bool:
        """Check if structure matches observations."""
        if isinstance(observations, str):
            observed = self._parse_observation(observations)
            generated = structure.get_top_view()
            if generated.shape != observed.shape:
                return False
            if not np.array_equal(generated, observed):
                return False
        else:
            observed = self._parse_observation(observations)
            generated = structure.get_top_view()
            if generated.shape != observed.shape:
                return False
            if not np.array_equal(generated, observed):
                return False

        # Validate physical support
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

            # Find structure start
            structure_start = -1
            for i, line in enumerate(lines):
                if 'Structure:' in line:
                    structure_start = i
                    break

            if structure_start == -1:
                for i, line in enumerate(lines):
                    if line.strip().startswith('Layer 1:'):
                        structure_start = i
                        break

            if structure_start == -1:
                return None

            # Parse layers
            layers = []
            current_layer = []
            in_layer = False
            layer_num = 0

            for line in lines[structure_start:]:
                line = line.strip()

                if not line or line.startswith('Note:') or line.startswith('Reasoning:'):
                    continue

                if line.startswith('Layer '):
                    import re
                    match = re.match(r'Layer\s+(\d+)', line)
                    if match:
                        new_layer_num = int(match.group(1))
                        if new_layer_num == layer_num + 1 or (new_layer_num == 1 and layer_num == 0):
                            if current_layer:
                                layers.append(current_layer)
                                current_layer = []
                            in_layer = True
                            layer_num = new_layer_num
                        else:
                            break
                elif in_layer and line:
                    reasoning_indicators = ['therefore', 'because', 'since', 'thus', 'reasoning:', 'explanation:']
                    if any(indicator in line.lower() for indicator in reasoning_indicators):
                        break

                    try:
                        if len(line) > 100:
                            continue

                        if ',' in line:
                            parts = [x.strip() for x in line.split(',')]
                        elif '|' in line:
                            parts = [x.strip() for x in line.split('|')]
                        elif any(c in line for c in ['[', ']']):
                            line = line.strip('[]')
                            if ',' in line:
                                parts = [x.strip() for x in line.split(',')]
                            else:
                                parts = line.split()
                        else:
                            parts = line.split()

                        row = []
                        for x in parts:
                            if x.strip():
                                val = int(x)
                                if val not in [0, 1]:
                                    break
                                row.append(val)

                        if row and len(row) > 0:
                            current_layer.append(row)
                    except:
                        pass

            if current_layer:
                layers.append(current_layer)

            if layers:
                return Structure3D(layers)

        except Exception as e:
            pass

        return None

    def evaluate_single_observation_set(
        self,
        llm: LLMInterface,
        observation_set: Dict,
        n_queries: int = 10,
        use_optimizer: bool = True,
        verbose: bool = True,
        max_retries: int = 3
    ) -> Dict:
        """
        评估单个观测集，集成算法优化

        Args:
            llm: LLM接口
            observation_set: 观测集
            n_queries: LLM查询次数
            use_optimizer: 是否启用优化算法
            verbose: 是否详细输出
            max_retries: 最大重试次数
        """

        observations = observation_set.get('observation', observation_set.get('observations', []))
        ground_truth_structures = observation_set.get('ground_truth_structures', [])

        # Get ground truth hashes
        gt_hashes = set()
        for gt in ground_truth_structures:
            if 'layers' in gt and isinstance(gt['layers'][0], str):
                layers = []
                grid_size = int(len(gt['layers'][0]) ** 0.5)
                for layer_str in gt['layers']:
                    layer = []
                    for i in range(grid_size):
                        row = [int(layer_str[i * grid_size + j]) for j in range(grid_size)]
                        layer.append(row)
                    layers.append(layer)
                struct = Structure3D(layers)
                gt_hashes.add(struct.get_hash())
            else:
                struct = Structure3D(gt['layers'])
                gt_hashes.add(struct.get_hash())

        # ========== LLM查询阶段 ==========
        all_hypotheses = []
        parse_success_count = 0
        prior_structures = []

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0

        if verbose:
            obs_id = observation_set.get('observation_id', 'unknown')
            print(f"\n{'='*60}")
            print(f"Observation Set ID: {obs_id}")
            print(f"Ground Truths: {len(ground_truth_structures)}")
            print(f"Querying LLM {n_queries} times...")

        for i in range(n_queries):
            if verbose:
                print(f"  Query {i + 1}/{n_queries}...", end='', flush=True)

            query_start_time = time.time()
            prompt = self.create_prompt(observations, prior_structures=prior_structures)

            structure = None

            for attempt in range(max_retries):
                try:
                    if hasattr(llm, 'query_with_usage'):
                        result = llm.query_with_usage(prompt)
                        response = result['response']

                        usage = result.get('usage', {})
                        total_prompt_tokens += usage.get('prompt_tokens', 0)
                        total_completion_tokens += usage.get('completion_tokens', 0)
                        total_tokens += usage.get('total_tokens', 0)
                        total_cost += result.get('cost', 0.0)
                    else:
                        response = llm.query(prompt)

                    if response and not response.startswith("Error querying"):
                        try:
                            structure = self.parse_llm_response(response)
                            if structure:
                                structure = structure.normalize()
                                parse_success_count += 1
                                query_time = time.time() - query_start_time
                                if verbose:
                                    print(f" ✓ ({query_time:.1f}s)", flush=True)
                                break
                            else:
                                if verbose and attempt == max_retries - 1:
                                    query_time = time.time() - query_start_time
                                    print(f" ✗ (parse failed, {query_time:.1f}s)", flush=True)
                        except Exception as parse_error:
                            if verbose and attempt == max_retries - 1:
                                query_time = time.time() - query_start_time
                                print(f" ✗ (parse error: {str(parse_error)[:30]}, {query_time:.1f}s)", flush=True)
                    else:
                        if verbose and attempt == max_retries - 1:
                            query_time = time.time() - query_start_time
                            error_preview = response[:50] if response else "no response"
                            print(f" ✗ (error: {error_preview}, {query_time:.1f}s)", flush=True)

                except Exception as e:
                    if verbose and attempt == max_retries - 1:
                        query_time = time.time() - query_start_time
                        print(f" ✗ (exception: {str(e)[:50]}, {query_time:.1f}s)", flush=True)

            if not structure and verbose:
                query_time = time.time() - query_start_time
                if query_time < 1.0:  # If very quick, line break wasn't printed
                    print("")

            if structure:
                all_hypotheses.append(structure)
                s_hash = structure.get_hash()
                if s_hash not in {s.get_hash() for s in prior_structures}:
                    prior_structures.append(structure)

        if verbose:
            print(f"LLM generated {len(all_hypotheses)} structures ({parse_success_count}/{n_queries} successful)")

        # ========== 优化阶段 ==========
        n_before_optimization = len(all_hypotheses)

        if use_optimizer and all_hypotheses:
            if verbose:
                print(f"\nApplying algorithmic optimizations...")

            try:
                # 1. 修复无效结构
                fixed_structures = []
                fix_success = 0
                fix_failed = 0

                for struct in all_hypotheses:
                    try:
                        if not self.validate_structure_matches_observations(struct, observations):
                            fixed = self.fixer.fix_structure(struct, observations)
                            if fixed and self.validate_structure_matches_observations(fixed, observations):
                                fixed_structures.append(fixed)
                                fix_success += 1
                            else:
                                fix_failed += 1
                        else:
                            fixed_structures.append(struct)
                    except Exception as e:
                        if verbose:
                            print(f"  Warning: Error fixing structure: {str(e)[:50]}")
                        fix_failed += 1

                if verbose:
                    print(f"  Constraint fixing: {len(fixed_structures)} valid ({fix_success} fixed, {fix_failed} failed)")

                # 2. 局部搜索扩展
                try:
                    expanded_structures = self.local_search.optimize_with_local_search(
                        observations,
                        fixed_structures,
                        n_expansions=15,
                        validator=self.validate_structure_matches_observations
                    )

                    if verbose:
                        print(f"  Local search: {len(expanded_structures)} structures (+{len(expanded_structures) - len(fixed_structures)})")
                except Exception as e:
                    if verbose:
                        print(f"  Local search failed: {str(e)[:50]}, skipping")
                    expanded_structures = fixed_structures

                # 3. 多样性增强
                try:
                    final_structures = self.diversity_enhancer.enhance_diversity(
                        expanded_structures,
                        observations,
                        target_count=30,
                        validator=self.validate_structure_matches_observations
                    )

                    if verbose:
                        print(f"  Diversity enhancement: {len(final_structures)} structures (+{len(final_structures) - len(expanded_structures)})")
                except Exception as e:
                    if verbose:
                        print(f"  Diversity enhancement failed: {str(e)[:50]}, skipping")
                    final_structures = expanded_structures

                all_hypotheses = final_structures

                if verbose:
                    total_gain = len(all_hypotheses) - n_before_optimization
                    print(f"  Total optimization gain: {'+' if total_gain >= 0 else ''}{total_gain} structures")

            except Exception as e:
                if verbose:
                    print(f"  Optimization failed: {str(e)}")
                    print(f"  Continuing with unoptimized results...")
                # Keep original hypotheses if optimization completely fails

        # ========== 评估阶段 ==========
        valid_hypotheses = [h for h in all_hypotheses
                            if self.validate_structure_matches_observations(h, observations)]

        unique_hashes = set()
        unique_structures = []
        for struct in valid_hypotheses:
            s_hash = struct.get_hash()
            if s_hash not in unique_hashes:
                unique_hashes.add(s_hash)
                unique_structures.append(struct)

        recovered_gts = unique_hashes & gt_hashes

        parse_success_rate = parse_success_count / n_queries if n_queries > 0 else 0
        valid_rate = len(valid_hypotheses) / len(all_hypotheses) if all_hypotheses else 0
        novelty_rate = len(unique_structures) / len(all_hypotheses) if all_hypotheses else 0
        recovery_rate = len(recovered_gts) / len(gt_hashes) if gt_hashes else 0

        if verbose:
            print(f"\n{'─'*60}")
            print(f"Results:")
            print(f"  Valid structures: {len(valid_hypotheses)}/{len(all_hypotheses)} ({valid_rate:.1%})")
            print(f"  Unique structures: {len(unique_structures)} ({novelty_rate:.1%})")
            print(f"  Recovered ground truths: {len(recovered_gts)}/{len(gt_hashes)} ({recovery_rate:.1%})")
            print(f"  Parse success rate: {parse_success_rate:.1%}")
            if use_optimizer:
                print(f"  Optimization enabled: ✓ (+{len(all_hypotheses) - n_before_optimization} structures)")
            print(f"{'='*60}")

        obs_id = observation_set.get('observation_id', observation_set.get('observation_set_id', 'unknown'))

        return {
            'observation_set_id': obs_id,
            'n_ground_truths': len(ground_truth_structures),
            'n_queries': n_queries,
            'n_hypotheses_before_optimization': n_before_optimization,
            'n_hypotheses_after_optimization': len(all_hypotheses),
            'n_valid': len(valid_hypotheses),
            'n_unique': len(unique_structures),
            'n_recovered_gts': len(recovered_gts),
            'parse_success_count': parse_success_count,
            'parse_success_rate': parse_success_rate,
            'valid_rate': valid_rate,
            'novelty_rate': novelty_rate,
            'recovery_rate': recovery_rate,
            'optimizer_enabled': use_optimizer,
            'optimizer_gain': len(all_hypotheses) - n_before_optimization if use_optimizer else 0,
            'token_usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens
            },
            'cost': total_cost,
            'unique_structures': [s.to_string() for s in unique_structures]
        }

    def run_benchmark(
        self,
        llm: LLMInterface,
        n_samples: int = 10,
        n_queries_per_sample: int = 10,
        use_optimizer: bool = True,
        observation_type: str = "top",
        seed: Optional[int] = None,
        verbose: bool = True
    ) -> Dict:
        """
        运行完整基准测试

        Args:
            llm: LLM接口
            n_samples: 采样的观测集数量
            n_queries_per_sample: 每个观测集的查询次数
            use_optimizer: 是否启用优化算法
            observation_type: 观测类型
            seed: 随机种子
            verbose: 是否详细输出
        """

        print(f"\n{'='*70}")
        print(f"3D Benchmark with Algorithmic Optimization")
        print(f"{'='*70}")
        print(f"Optimizer: {'ENABLED' if use_optimizer else 'DISABLED'}")
        print(f"Samples: {n_samples}, Queries per sample: {n_queries_per_sample}")
        print(f"{'='*70}\n")

        # Sample observation sets
        sampled_sets = self.sample_observation_sets(n_samples, observation_type, seed)

        if not sampled_sets:
            return {
                'error': 'No observation sets available',
                'results': []
            }

        # Evaluate each observation set
        all_results = []

        for idx, obs_set in enumerate(sampled_sets, 1):
            if verbose:
                print(f"\nProcessing observation set {idx}/{len(sampled_sets)}")

            result = self.evaluate_single_observation_set(
                llm=llm,
                observation_set=obs_set,
                n_queries=n_queries_per_sample,
                use_optimizer=use_optimizer,
                verbose=verbose
            )

            all_results.append(result)

        # Aggregate statistics
        total_queries = sum(r['n_queries'] for r in all_results)
        total_hypotheses_before = sum(r['n_hypotheses_before_optimization'] for r in all_results)
        total_hypotheses_after = sum(r['n_hypotheses_after_optimization'] for r in all_results)
        total_valid = sum(r['n_valid'] for r in all_results)
        total_unique = sum(r['n_unique'] for r in all_results)
        total_recovered = sum(r['n_recovered_gts'] for r in all_results)
        total_gts = sum(r['n_ground_truths'] for r in all_results)

        avg_valid_rate = np.mean([r['valid_rate'] for r in all_results])
        avg_novelty_rate = np.mean([r['novelty_rate'] for r in all_results])
        avg_recovery_rate = np.mean([r['recovery_rate'] for r in all_results])

        total_cost = sum(r.get('cost', 0.0) for r in all_results)
        total_tokens = sum(r['token_usage']['total_tokens'] for r in all_results)

        summary = {
            'benchmark_config': {
                'n_samples': n_samples,
                'n_queries_per_sample': n_queries_per_sample,
                'optimizer_enabled': use_optimizer,
                'observation_type': observation_type,
                'seed': seed
            },
            'aggregate_stats': {
                'total_queries': total_queries,
                'total_hypotheses_before_optimization': total_hypotheses_before,
                'total_hypotheses_after_optimization': total_hypotheses_after,
                'optimizer_gain': total_hypotheses_after - total_hypotheses_before,
                'total_valid': total_valid,
                'total_unique': total_unique,
                'total_recovered_gts': total_recovered,
                'total_ground_truths': total_gts,
                'avg_valid_rate': float(avg_valid_rate),
                'avg_novelty_rate': float(avg_novelty_rate),
                'avg_recovery_rate': float(avg_recovery_rate),
                'total_cost': float(total_cost),
                'total_tokens': int(total_tokens)
            },
            'per_observation_results': all_results,
            'timestamp': datetime.now().isoformat()
        }

        # Print summary
        print(f"\n{'='*70}")
        print(f"BENCHMARK SUMMARY")
        print(f"{'='*70}")
        print(f"Optimizer: {'ENABLED ✓' if use_optimizer else 'DISABLED ✗'}")
        print(f"Samples processed: {len(all_results)}")
        print(f"Total queries: {total_queries}")
        print(f"Hypotheses before optimization: {total_hypotheses_before}")
        print(f"Hypotheses after optimization: {total_hypotheses_after}")
        if use_optimizer:
            print(f"Optimization gain: +{total_hypotheses_after - total_hypotheses_before} structures")
        print(f"\nPerformance Metrics:")
        print(f"  Validity:    {avg_valid_rate:.1%} ({total_valid}/{total_hypotheses_after})")
        print(f"  Uniqueness:  {avg_novelty_rate:.1%} ({total_unique}/{total_hypotheses_after})")
        print(f"  Recovery:    {avg_recovery_rate:.1%} ({total_recovered}/{total_gts})")
        print(f"\nCost: ${total_cost:.4f}, Tokens: {total_tokens:,}")
        print(f"{'='*70}\n")

        return summary


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_llm_from_config(config: Dict) -> LLMInterface:
    """Create LLM interface from config."""
    llm_config = config['llm']
    llm_type = llm_config['type'].lower()

    if llm_type == 'deepseek':
        api_key = llm_config['api_keys']['deepseek']
        model_name = llm_config['models']['deepseek']
        temperature = llm_config.get('temperature', 0.7)

        return DeepSeekLLM(
            api_key=api_key,
            model=model_name,
            temperature=temperature
        )
    elif llm_type == 'openrouter':
        api_key = llm_config['api_keys']['openrouter']
        model_name = llm_config['models']['openrouter']
        temperature = llm_config.get('temperature', 0.7)

        return OpenRouterLLM(
            api_key=api_key,
            model=model_name,
            temperature=temperature
        )
    elif llm_type == 'openai':
        api_key = llm_config['api_keys']['openai']
        model_name = llm_config['models']['openai']
        temperature = llm_config.get('temperature', 0.7)

        return OpenAILLM(
            api_key=api_key,
            model=model_name,
            temperature=temperature
        )
    elif llm_type == 'anthropic':
        api_key = llm_config['api_keys']['anthropic']
        model_name = llm_config['models']['anthropic']
        temperature = llm_config.get('temperature', 0.7)

        return AnthropicLLM(
            api_key=api_key,
            model=model_name,
            temperature=temperature
        )
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")


def main():
    parser = argparse.ArgumentParser(description='3D Benchmark with Algorithmic Optimization')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset JSON file')
    parser.add_argument('--n-samples', type=int, default=10, help='Number of observation sets to sample')
    parser.add_argument('--n-queries', type=int, default=10, help='Number of queries per observation set')
    parser.add_argument('--optimizer', type=str, default='on', choices=['on', 'off'],
                       help='Enable/disable algorithmic optimization')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file path')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create LLM
    llm = create_llm_from_config(config)

    # Create benchmark
    benchmark = Benchmark3DWithOptimizer(args.dataset)

    # Run benchmark
    use_optimizer = (args.optimizer == 'on')

    results = benchmark.run_benchmark(
        llm=llm,
        n_samples=args.n_samples,
        n_queries_per_sample=args.n_queries,
        use_optimizer=use_optimizer,
        seed=args.seed,
        verbose=True
    )

    # Save results
    if args.output:
        output_path = args.output
    else:
        # Generate default output path
        llm_type = config['llm']['type']
        model_name = config['llm']['models'][llm_type].replace('/', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        opt_suffix = 'with_optimizer' if use_optimizer else 'no_optimizer'
        output_path = f"results/3d_complete_{model_name}_{opt_suffix}_{timestamp}.json"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()

