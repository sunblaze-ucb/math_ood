# OMEGA: Can LLMs Reason Outside the Box in Math? Evaluating Exploratory, Compositional, and Transformative Generalization

[![arXiv](https://img.shields.io/badge/arXiv-preprint-b31b1b.svg)](https://arxiv.org/abs/2506.18880)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Datasets-yellow)](https://huggingface.co/datasets/allenai/omega-explorative)

**Authors:** Yiyou Sun¹, Shawn Hu⁴, Georgia Zhou¹, Ken Zheng¹, Hannaneh Hajishirzi²'³, Nouha Dziri², Dawn Song¹

¹University of California, Berkeley | ²Ai2 | ³University of Washington | ⁴dmodel.ai

## Abstract

Recent large-scale language models (LLMs) with long Chain-of-Thought reasoning—such as DeepSeek-R1—have achieved impressive results on Olympiad-level mathematics benchmarks. However, they often rely on a narrow set of strategies and struggle with problems that require a novel way of thinking. To systematically probe these limitations, we introduce **OMEGA**, a controlled yet diverse benchmark designed to evaluate three axes of out-of-distribution generalization inspired by Boden's typology of creativity:

1. **Exploratory** — applying known problem-solving skills to more complex instances within the same problem domain
2. **Compositional** — combining distinct reasoning skills, previously learned in isolation, to solve novel problems that require integrating these skills in new and coherent ways  
3. **Transformative**  — adopting novel, often unconventional strategies by moving beyond familiar approaches to solve problems more effectively

MathOOD consists of programmatically generated training–test pairs derived from templated problem generators across geometry, number theory, algebra, combinatorics, logic, and puzzles, with solutions verified using symbolic, numerical, or graphical methods.

## 🚀 Quick Start

### Generate Problems

Generate problems by difficulty level:
```bash
# problem_types: 
python generate_basics_by_difficulty.py --difficulty 3 --num_samples 10 --problem_types algebra_linear_equation algebra_polynomial_roots arithmetic_mixed arithmetic_list_prime_factors arithmetic_gcd

python generate_matrix_by_difficulty.py --difficulty 3 --num_samples 10  --problem_types inverse multiplication determinant rank eigenvalues power svd

python generate_function_by_difficulty.py --difficulty level_3 --num_samples 10  --problem_types intersection intersection_coords extrema extrema_coords area zeros derivative_sign

python generate_combinatory_by_difficulty.py --difficulty 3 --num_samples 10  --type distribution
python generate_combinatory_by_difficulty.py --difficulty 3 --num_samples 10  --type pattern_matching
for event in no_fixed_points no_specific_letter_fixed exactly_n_specific_fixed at_least_n_specific_fixed; do
  python generate_combinatory_by_difficulty.py --difficulty 3 --num_samples 10  --type probability --event $event
done 
for type in polygon circle triangle basic; do
  python generate_geometry_by_difficulty.py --difficulty 3 --num_samples 10  --problem_types $type
done 
for type in prime_mod triple_count digit_sum; do
  python generate_number_theory_by_difficulty.py --difficulty 3 --num_samples 10  --problem_types $type
done 
for type in polygon_chords polygon_color rotation; do
  python generate_extra_geometry_by_difficulty.py --difficulty level_3 --num_samples 10  --problem_types $type
done 
for type in blocked_grid grid_knight grid_rook zebralogic grid_chip; do
  python generate_logic_puzzles_by_difficulty.py --difficulty level_3 --num_samples 10  --problem_types $type
done 
```

Results will be stored in `problems/specific_difficulty`. 

## 📊 Benchmark Structure

### Problem Categories

MathOOD covers six major mathematical domains:

- **🔢 Arithmetic** - Matrix operations, GCD, prime factorization
- **📐 Geometry** - Circles, polygons, perpendicular intersections  
- **🔤 Algebra** - Function analysis, polynomial roots, linear equations
- **🎲 Combinatorics** - Probability, pattern matching, distributions
- **🧩 Logic** - Grid-world problems, constraint satisfaction
- **📊 Number Theory** - Quadratic residues, ordered sets

### Generalization Types (Example)

#### 1. Exploratory Generalization [![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Datasets-yellow)](https://huggingface.co/datasets/allenai/omega-explorative)
Tests scaling to more complex instances within the same domain:
- **Training**: Simple 2x2 matrix determinants
- **Testing**: Complex 5x5 matrix determinants

#### 2. Compositional Generalization [![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Datasets-yellow)](https://huggingface.co/datasets/allenai/omega-compositional) 
Tests combining multiple mathematical skills:
- **Training**: Geometry (circle properties) + Algebra (function intersection)
- **Testing**: Problems requiring both skills simultaneously

#### 3. Transformative Generalization [![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Datasets-yellow)](https://huggingface.co/datasets/allenai/omega-transformative)
Tests adoption of novel solution strategies:
- **Training**: Standard polynomial root-finding methods
- **Testing**: Complex analysis techniques (De Moivre's theorem)

### Problem Collection Structure

The benchmark includes three specialized problem collections that implement different generalization paradigms:

#### `problems/explorative/`
Contains **exploratory generalization** datasets that test scaling within the same mathematical domain. Models are trained on low-complexity problems and evaluated on progressively harder instances from the same template. Examples:
- **Algebra**: Function analysis (area, derivatives, extrema, intersections, zeros), linear equations, polynomial roots
- **Arithmetic**: Number (GCD, prime factorization), matrix operations (determinant, eigenvalues, inverse, multiplication, power, rank, SVD)
- **Combinatorics**: Probability analysis, pattern matching, distribution problems

#### `problems/compositional/`
Features **compositional generalization** scenarios that combine distinct mathematical domains from training into novel multi-domain problems. Examples:
- Algebra (polynomial roots) + Arithmetic (GCD)
- Geometry (circles) + Algebra (function intersection) 
- Combinatorics (probability) + Matrix operations (rank)
- Geometry (polygon properties) + Pattern matching

#### `problems/transformative/`
Presents **transformative generalization** challenges requiring fundamentally different solution approaches than those seen in training. Examples:
- Matrix rank problems requiring advanced linear algebra techniques
- Function intersection via complex analysis methods
- Polynomial root-finding using De Moivre's theorem
- Spatial reasoning problems requiring geometric transformations

Each folder contains detailed README files with dataset links and problem descriptions.

## 📚 Citation

If you use OMEGA in your research, please cite:

```bibtex
@article{sun2024omega,
  title     = {OMEGA: Can LLMs Reason Outside the Box in Math? Evaluating Exploratory, Compositional, and Transformative Generalization},
  author    = {Yiyou Sun and Shawn Hu and Georgia Zhou and Ken Zheng and Hannaneh Hajishirzi and Nouha Dziri and Dawn Song},
  journal   = {arXiv preprint arXiv:2506.18880},
  year      = {2024},
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


For questions or issues, please open a GitHub issue or contact the authors.