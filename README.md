# MathOOD: Probing the Generalization Limits of LLMs in Math Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-preprint-b31b1b.svg)](https://arxiv.org/abs/your-paper-link)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Datasets-yellow)](https://huggingface.co/datasets/sunyiyou)

**Authors:** Yiyou Sun¹, Shawn Hu⁴, Georgia Zhou¹, Ken Zheng¹, Hannaneh Hajishirzi²'³, Nouha Dziri², Dawn Song¹

¹University of California, Berkeley | ²Ai2 | ³University of Washington | ⁴dmodel.ai

## Abstract

Recent large-scale language models (LLMs) with long Chain-of-Thought reasoning—such as DeepSeek-R1—have achieved impressive results on Olympiad-level mathematics benchmarks. However, they often rely on a narrow set of strategies and struggle with problems that require a novel way of thinking. To systematically probe these limitations, we introduce **MathOOD**, a controlled yet diverse benchmark designed to evaluate three axes of out-of-distribution generalization inspired by Boden's typology of creativity:

1. **Exploratory** — applying known problem-solving skills to more complex instances within the same problem domain
2. **Compositional** — combining distinct reasoning skills, previously learned in isolation, to solve novel problems that require integrating these skills in new and coherent ways  
3. **Transformative** — adopting novel, often unconventional strategies by moving beyond familiar approaches to solve problems more effectively

MathOOD consists of programmatically generated training–test pairs derived from templated problem generators across geometry, number theory, algebra, combinatorics, logic, and puzzles, with solutions verified using symbolic, numerical, or graphical methods.

## 🚀 Quick Start

### Generate Problems

Generate problems by difficulty level:
```bash
python generate_basics_by_difficulty.py --difficulty 3 --num_samples 100
```

Generate specific problem types:
```bash
python generate_function_by_difficulty.py --type intersection --count 50
python generate_matrix_by_difficulty.py --type determinant --size 4
python generate_combinatory_by_difficulty.py --type probability --event no_fixed_points
```

## 📊 Benchmark Structure

### Problem Categories

MathOOD covers six major mathematical domains:

- **🔢 Arithmetic** - Matrix operations, GCD, prime factorization
- **📐 Geometry** - Circles, polygons, perpendicular intersections  
- **🔤 Algebra** - Function analysis, polynomial roots, linear equations
- **🎲 Combinatorics** - Probability, pattern matching, distributions
- **🧩 Logic** - Grid-world problems, constraint satisfaction
- **📊 Number Theory** - Quadratic residues, ordered sets

### Generalization Types

#### 1. Exploratory Generalization
Tests scaling to more complex instances within the same domain:
- **Training**: Simple 2x2 matrix determinants
- **Testing**: Complex 5x5 matrix determinants with special structures

#### 2. Compositional Generalization  
Tests combining multiple mathematical skills:
- **Example**: Geometry (circle properties) + Algebra (function intersection)
- **Training**: Each skill learned separately
- **Testing**: Problems requiring both skills simultaneously

#### 3. Transformative Generalization
Tests adoption of novel solution strategies:
- **Training**: Standard polynomial root-finding methods
- **Testing**: Complex analysis techniques (De Moivre's theorem)

## 🎯 Usage Examples

### Generate Matrix Problems

```python
from modules.matrix_computations import generate_matrix_determinant_problem

# Generate a determinant problem
problem = generate_matrix_determinant_problem(size=3, min_val=-5, max_val=5)
print(f"Question: {problem.question}")
print(f"Answer: {problem.answer}")
```

### Generate Function Analysis Problems

```python  
from modules.function_analysis import generate_intersection_problem

# Generate function intersection problem
problem = generate_intersection_problem(difficulty=4)
print(f"Question: {problem.question}")
print(f"Answer: {problem.answer}")
```

### Generate Combinatorial Problems

```python
from modules.combinatories import generate_probability_problem

# Generate probability problem
problem = generate_probability_problem(
    event_type="no_fixed_points",
    length=6,
    letters=['a', 'b', 'c']
)
print(f"Question: {problem.question}")
print(f"Answer: {problem.answer}")
```

## 📚 Citation

If you use MathOOD in your research, please cite:

```bibtex
@article{sun2025mathood,
  title={MathOOD: Probing the Generalization Limits of LLMs in Math Reasoning},
  author={Sun, Yiyou and Hu, Shawn and Zhou, Georgia and Zheng, Ken and Hajishirzi, Hannaneh and Dziri, Nouha and Song, Dawn},
  year={2025},
  
}
```

[//]: # (journal={arXiv preprint arXiv:xxxx.xxxxx})

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


For questions or issues, please open a GitHub issue or contact the authors.