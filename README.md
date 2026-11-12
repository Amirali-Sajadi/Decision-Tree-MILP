## Optimal Decision Trees Using Mixed-Integer Linear Programming

This project focuses on implementing and improving MILP-based optimal decision tree classification models, including both univariate and multivariate formulations. Unlike traditional heuristic approaches such as CART and ID3, these models formulate decision tree construction as an optimization problem, aiming to find globally optimal trees for a given dataset.

To address the computational complexity of these formulations, two acceleration techniques were investigated and implemented:

CART-Based Warm Start: A pipeline that uses a CART-generated tree as an initial feasible solution, guiding the Branch-and-Bound algorithm and reducing its search tree size. This approach achieved approximately a 50% reduction in solution time.

Valid Inequalities (Cuts): Development and integration of additional valid constraints to strengthen the MILP formulation, reduce the feasible search space, and accelerate the optimization process.

The project explores the intersection of mathematical optimization and machine learning, with an emphasis on improving the computational efficiency of exact decision tree learning.
