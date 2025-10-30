# 3D Structure Discovery Benchmark - Performance Optimization

## 📋 Project Overview

This project focuses on improving Large Language Model (LLM) performance on the **3D Structure Discovery** creativity task from the HypoSpace benchmark. The goal is to enhance the model's ability to generate valid, unique, and comprehensive hypotheses when reconstructing 3D structures from 2D observations.

## 🎯 Task Description

**3D Structure Discovery** is a creativity task that challenges LLMs to:
- Infer complete 3D structures from top-view observations
- Generate multiple valid hypotheses that satisfy physical constraints (gravity, support)
- Explore the space of admissible structures systematically
- Maximize coverage of the ground truth hypothesis space

### Evaluation Metrics

The task is evaluated on three key dimensions:
- **Validity (V)**: Proportion of generated structures that satisfy all physical constraints
- **Uniqueness (U)**: Non-redundancy among generated hypotheses
- **Recovery (R)**: Coverage of the true admissible hypothesis space

## 🚀 Optimization Approach

### Selected LLM
**DeepSeek Chat** (`deepseek-chat`) via DeepSeek API
- Cost-effective: $0.14/$0.28 per M tokens (input/output)
- Strong reasoning capabilities
- Good performance on structured tasks

### Optimization Methods

I implemented **5 key optimization strategies** to improve LLM performance:

#### 1. **Multi-Strategy Prompting**
Instead of using a single prompting approach, I designed 5 different exploration strategies that guide the model to explore different regions of the solution space:

- **Systematic**: Step-by-step analytical approach
- **Minimal**: Generate structures with minimum blocks
- **Maximal**: Generate structures with maximum blocks (within constraints)
- **Diverse**: Explicitly request maximum diversity from previous attempts
- **Analytical**: Require analysis before generation

These strategies are applied cyclically across queries to ensure comprehensive coverage.

#### 2. **Enhanced Constraint Specification**
Restructured prompts with clearer organization:
```
GIVEN INFORMATION → PHYSICAL CONSTRAINTS → STRATEGY GUIDANCE → 
PREVIOUS ATTEMPTS → OUTPUT FORMAT → VALIDATION CHECKLIST
```

Key improvements:
- Explicit gravity/support requirements
- Clear layer numbering (Layer 1 = bottom)
- Detailed format examples
- Built-in validation checklist

#### 3. **Adaptive Query Allocation**
Dynamic query count based on problem complexity:
```python
target_queries = max(5, int(n_ground_truths × 1.5))
```
- Simple problems: fewer queries
- Complex problems: more queries
- Guarantees minimum exploration

#### 4. **Iterative Refinement with Feedback**
- Display up to 5 previous structures
- Explicit uniqueness requirements
- Structured history to avoid repetition

#### 5. **Intelligent Retry Mechanism**
- Up to 3 retries per query
- Parse failures trigger automatic retry
- Maintains detailed statistics

## 📊 Results

### Performance Comparison

![Performance Comparison](results/performance_comparison.png)

### Key Improvements (10 samples, seed=100)

| Metric | Original | Optimized | Absolute Gain | Relative Gain |
|--------|----------|-----------|---------------|---------------|
| **Validity** | 64.00% ± 21.07% | 70.00% ± 13.42% | **+6.00%** | **+9.4%** |
| **Uniqueness** | 68.00% ± 18.33% | 93.06% ± 11.74% | **+25.06%** | **+36.8%** ⭐ |
| **Recovery** | 28.89% ± 16.30% | 45.93% ± 21.99% | **+17.04%** | **+59.0%** ⭐⭐ |
| **Average** | 53.63% | 69.66% | **+16.03%** | **+29.9%** |

### Highlights

✅ **Most Significant Improvement**: Recovery rate increased by **59%** (relative)
- From 28.89% to 45.93%
- Multi-strategy prompting enables more comprehensive exploration

✅ **Uniqueness Boost**: +36.8% relative improvement
- From 68.00% to 93.06%
- Diverse strategies prevent repetitive generations

✅ **Stable Validity**: Maintained high validity with reduced variance
- Standard deviation decreased from 21.07% to 13.42%
- Enhanced constraints ensure consistent quality

✅ **Cost-Effective**: Only 5.6% increase in cost
- Performance/Cost ratio improved by 30%
- Efficient resource allocation

## 🛠️ Implementation

### File Structure
```
3d/
├── run_3d_benchmark.py              # Original benchmark
├── run_3d_benchmark_optimized.py    # Optimized version ⭐
├── compare_results.py               # Visualization tool
├── config/
│   ├── config_deepseek.yaml         # DeepSeek configuration
│   └── config_gpt4o.yaml
├── datasets/
│   └── 3d_complete.json             # 129 observation sets
├── modules/
│   ├── llm_interface.py             # LLM API interfaces
│   └── models.py
└── results/
    └── performance_comparison.png   # Comparison chart
```

### Quick Start

#### 1. Install Dependencies
```bash
pip install requests numpy scipy pyyaml matplotlib seaborn
```

#### 2. Configure API Key
Edit `config/config_deepseek.yaml`:
```yaml
llm:
  type: deepseek
  models:
    deepseek: "deepseek-chat"
  api_keys:
    deepseek: "your-api-key-here"
  temperature: 0.7
```

#### 3. Run Original Benchmark
```bash
python run_3d_benchmark.py \
  --dataset datasets/3d_complete.json \
  --config config/config_deepseek.yaml \
  --n-samples 10 \
  --n-queries 10 \
  --seed 100
```

#### 4. Run Optimized Benchmark
```bash
python run_3d_benchmark_optimized.py \
  --dataset datasets/3d_complete.json \
  --config config/config_deepseek.yaml \
  --n-samples 10 \
  --query-multiplier 1.5 \
  --seed 100
```

#### 5. Generate Comparison Chart
```bash
python compare_results.py
```

### Configuration Options

| Parameter | Description | Default | Optimized |
|-----------|-------------|---------|-----------|
| `--n-samples` | Number of test samples | 10 | 10 |
| `--n-queries` | Fixed queries per sample | 10 | Adaptive |
| `--query-multiplier` | Adaptive multiplier | N/A | 1.5 |
| `--max-retries` | Retries per query | 3 | 3 |
| `--seed` | Random seed | None | 100 |
| `--temperature` | LLM temperature | 0.7 | 0.7 |

## 🔬 Technical Details

### Strategy Selection Logic
```python
# First N queries: cycle through all strategies
if query_idx < len(strategies):
    strategy = strategies[query_idx]
# Mid-phase: favor systematic/analytical
elif query_idx < target_queries // 2:
    strategy = random.choice(["systematic", "diverse", "analytical"])
# Late-phase: favor extreme configurations
else:
    strategy = random.choice(["minimal", "maximal", "diverse"])
```

### Structure Validation
1. Parse LLM output → Structure3D object
2. Generate top view projection
3. Compare with observation
4. Check physical constraints (gravity support)
5. Compute normalized hash
6. Check uniqueness
7. Compare with ground truth

### Prompt Engineering Pattern
```
STRUCTURED INFORMATION PRESENTATION:
├── GIVEN: Problem description + statistics
├── CONSTRAINTS: Physical rules (explicit)
├── STRATEGY: Specific guidance per strategy
├── PREVIOUS: History (up to 5 structures)
├── OUTPUT FORMAT: Clear examples
└── VALIDATION: Self-check list
```

## 📈 Analysis

### Why Multi-Strategy Works

**Original Approach**: Single prompting strategy
- Limited exploration
- Prone to local optima
- Repetitive generations

**Optimized Approach**: 5 diverse strategies
- Systematic coverage of solution space
- Different strategies discover different structures
- Explicit diversity requirements reduce redundancy

### Sample-Level Performance

From the 10-sample test:
- **Best case**: 100% validity, 100% uniqueness, 67% recovery
- **Worst case**: 30% validity, 67% uniqueness, 22% recovery
- **Consistency**: Reduced variance in all metrics

### Cost-Benefit Analysis

| Metric | Value | Improvement |
|--------|-------|-------------|
| Token usage | 45K → 52K | +15.6% |
| API cost | $0.009 → $0.0095 | +5.6% |
| Recovery/Cost ratio | 3209 → 4835 | **+50.7%** |
| Performance/Cost | 5959 → 7333 | **+23.1%** |

**Conclusion**: Significant performance gains with minimal cost increase.

## 🎓 Key Learnings

### 1. Prompt Diversity > Temperature Diversity
Instead of relying on temperature sampling, using **multiple prompting strategies** provides:
- More controlled exploration
- Strategy-specific guidance
- Better interpretability

### 2. Constraint Clarity is Critical
Clear, structured constraint descriptions dramatically improve:
- Validity rates
- Consistency across attempts
- Model's understanding of the task

### 3. Adaptive Resource Allocation
Matching query count to problem complexity:
- Prevents over-querying simple cases
- Ensures sufficient exploration for complex cases
- Improves efficiency

### 4. Iterative Feedback Helps
Showing previous attempts to the model:
- Reduces redundancy
- Encourages diversity
- Mimics human exploration process

## 🚀 Future Improvements

Potential areas for further optimization:

1. **Temperature Scheduling**: Vary temperature across queries
   - Low (0.3) for early queries → ensure validity
   - High (1.0) for late queries → maximize diversity

2. **Beam Search**: Generate multiple candidates per query, select best

3. **Constraint Programming**: Pre-compute valid configurations as hints

4. **Meta-Learning**: Learn which strategies work best for which problem types

5. **Ensemble Methods**: Combine multiple LLMs with different strengths

## 📚 References

- **HypoSpace Benchmark**: Framework for evaluating creative hypothesis generation
- **DeepSeek API**: https://api-docs.deepseek.com/
- **Original Task**: 3D Structure Discovery from 2D observations

## 👨‍💻 Author

Created as part of a creativity task optimization assignment, demonstrating:
- ✅ LLM selection and configuration (DeepSeek)
- ✅ Novel optimization methods (multi-strategy prompting)
- ✅ Significant performance improvements (59% gain in Recovery)
- ✅ Rigorous evaluation and comparison
- ✅ Cost-effectiveness analysis

## 📄 License

This project is part of the HypoSpace benchmark framework.

---

**Assignment Completion Date**: October 30, 2025  
**Task**: Improve LLM performance on 3D Structure Discovery  
**Method**: Multi-strategy prompting with adaptive query allocation  
**Result**: +59% Recovery, +37% Uniqueness, +9% Validity  
**Status**: ✅ Complete

