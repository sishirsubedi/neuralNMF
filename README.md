# neuralNMF

Pilot implementation of multilayer NMF combining ideas from ( neuralNMF package https://pypi.org/project/NeuralNMF/ ) and poisson matrix factorixation (from ASAP https://github.com/causalpathlab/asapp ) packages.

For factorization, $W \sim AS$ then use pytorch autograd for $A$ and least squares or PMF method for $S$ optimization. Use neural networks for multilayer factorization ( W -> $S1$ -> $S2$). 

(Will, Tyler, et al. "Neural Nonnegative Matrix Factorization for Hierarchical Multilayer Topic Modeling." arXiv preprint arXiv:2303.00058 (2023))

Cell type clusters are captured well with both least squares and PMF (needs more training epochs) but clusters are not as good as previous ASAP method.

Needs improvements - resolve loading matrix and capture hierarchical nature of cell types (T cells -> T naive, T reg, TH1/17, ..) .
