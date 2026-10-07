# Mathematical guide

## Proximal optimization and certificates

For Lasso, let `r=Ax-b`. A feasible dual vector satisfies `||Aᵀz||∞<=lambda`; scaling `r` into that domain gives the lower bound `D(z)=-.5||z||²-bᵀz`. The published gap `F(x)-D(z)` bounds objective suboptimality for primal-feasible x. ISTA steps with `L=||A||₂²`; FISTA adds Nesterov momentum. ADMM splits `x=z`, solves the ridge system, soft-thresholds z and updates its scaled multiplier. The deployed ADMM variable is z. Cholesky setup is real work, not a free iteration.

For masked nuclear completion, `F(X)=.5||PΩ(X-Y)||F²+lambda||X||*`. The masked gradient has Lipschitz constant1. The exact proximal operator soft-thresholds singular values. A dual residual must be supported on Ω and have operator norm at most lambda. Its dual value is `-.5||Z||F²-<Z,PΩY>`. Missing entries are excluded before arithmetic, including NaN placeholders.

For cyclic Poisson smoothing, `F(l)=sum(E*l-N*log(l))+rho||Dl||²`, with `l>=0` and `0log0=0`. The likelihood proximal root is

`prox(v)=(v-tE+sqrt((v-tE)²+4tN))/2`.

The negative branch is rationalized to avoid cancellation; N=0 permits an exact boundary solution. The smooth part has `L<=8rho`, and the implementation takes `t=1/max(8rho,1)`. KKT is checked with `l-max(0,l-gradient)`, not by requiring a zero full gradient at an active bound. The conic reference uses Poisson KL with its additive constant removed to match the owned convention.

## Travel-time convexification and duality

For segment distance d and speed s, change variables to `t=d/s`. The original fuel expression becomes `a*d²/t+b*d+c*t`. Speed limits and prefix arrival windows become linear inequalities `Gt<=h`. For nonnegative multipliers y, the separable dual is

`D(y)=b*sum(d)-hᵀy+sum(2sqrt(a*d²*(c+Gᵀy)))`,

provided `c+Gᵀy>=0`. The owned barrier finds a strict interior by a phase-I LP, then applies damped Newton to the logarithmic barrier. Recorded multipliers are repaired into the exact conjugate domain before computing a feasible dual bound. The final nearly active inequalities also supply a nonnegative least-squares KKT multiplier fit; it is retained only if its independently feasible dual bound improves. The central-path history and recovered final witness are distinguished in the plot. No strict interior means this feasible-start method is inapplicable; it is not silently run on an infeasible initialization.

Piecewise revenue is the minimum of two affine sales-revenue pieces per product. Hypograph variables turn its maximization into an LP. HiGHS supplies an independent primal solution and dual capacity prices; one-sided finite differences check local value sensitivity. A finite perturbation need not keep the same active basis.

## Geometry and matrix cones

The supplied circuit program uses `alpha>=0,sum(alpha)=1`, `log(Wmix)=log(W)alpha`. Log-space metric bounds are linear in alpha. Jensen gives conditional upper bounds only if the physical metrics are log-convex in log design. The computed blend is not a measured fabricated circuit or an identified regression model.

Nearest PSD projection first symmetrizes A, then clips negative eigenvalues. The antisymmetric component contributes a constant Frobenius loss. The cone variable is explicitly symmetric; an implicit symmetric-part eigenvalue condition on an unsymmetric variable is not enough to define the intended PSD matrix.

Perspective quadratic forms require a positive denominator; square-root hypographs require a nonnegative radicand. DCP witnesses verify the explicit transformed domains. They do not identify a unique inverse transformation absent the original definitions.

## Dynamics and robust risk

The regulator obeys `x[t+1]=Ax[t]+Bu[t]`, `x[0]=ones`, `x[T]=0` at T=100. Frobenius norm, sum of per-slot Euclidean norms, peak per-slot norm and total elementwise L1 are four different objectives. Squared energy shares the Frobenius minimizer, but objective values and dual scaling differ. A stable terminal-controllability SVD supplies the minimum-energy solution. An independent LP supplies the minimum-L1 objective. Nonunique solutions are compared by objective and feasibility, not identical coefficients.

Evacuation uses `q[t+1]=q[t]+Af[t]`, nonnegative bounded populations and signed bounded flows. Incidence columns sum to zero, so total mass is conserved. Signed edges permit bidirectional transport and within-slot forwarding under the given balance model. The supplied risk objective has no required full terminal evacuation. The terminal-safe-mass program is explicitly a separate extension.

The original portfolio solves `max muᵀw-gamma*max_j(wᵀSigma_j w)` with `sum(w)=1` and shorting allowed. Scenario epigraph multipliers sum to gamma. The common riskless first asset permits eliminating its weight. For `d=mu[risky]-mu[0]` and a dual mixture `S=sum(beta_j Sigma_j[risky,risky])`, a dual upper bound is `mu[0]+.25*dᵀS^-1d`. The code checks covariance PSD, budget, stationarity, primal / dual values and gross exposure. This is a theoretical supplied-data program, not investment guidance or an empirical asset-market study.

## Spectral algorithms

Orthogonal Householder similarity reduces a symmetric matrix to tridiagonal form. Wilkinson-shifted QR then safely deflates negligible subdiagonals and accumulates the orthogonal eigenbasis. Sorting eigenvalues preserves multiplicity; a dictionary keyed by eigenvalue would destroy it. Residual and orthogonality checks accompany convergence flags. For nonsymmetric nonnormal matrices, Schur columns form an invariant orthonormal basis but are not generally individual eigenvectors.

Channelwise rank-k SVD minimizes raw Frobenius approximation error, equal to the sum of discarded singular-value squares. Clipping display values changes that error, so raw optimality and display PSNR are reported separately. Factor parameter counts are not claims about encoded JPEG / PNG byte sizes. Centered PCA uses the right singular vectors of raw-unit or standardized data; sign-invariant projector comparisons verify the subspace.

## Recommendation is nonconvex

With fixed fitting-only mean m, prediction is `m+b_u+c_i+p_uᵀq_i`. The common objective is

`(sum_(u,i in Ω) e_ui² + lambda*(sum_u n_u||p_u||²+sum_i n_i||q_i||²) + beta*(sum_u n_u b_u²+sum_i n_i c_i²))/(2|Ω|)`.

Each ALS block augments neighboring factors with an intercept and solves a positive ridge system, giving monotone full objective values. Uniform rating minibatches give an unbiased objective for SGD / Adam, including degree-scaled penalties and repeated-index gradient accumulation. Block convexity does not give joint convexity. ALS monotonicity does not guarantee the global optimum or generalization, and a validation-selected early checkpoint need not be stationary. Fresh deployment audits record gradient residuals rather than asserting stationarity from a good test score.

## Combinatorial search and proof scope

SA uses Metropolis acceptance and a fixed cooling schedule; finite-budget results are heuristic. Genetic crossovers preserve permutations, but do not by themselves guarantee useful edge structures. AS and ACS differ in local versus global pheromone updates. A separately named 2-opt hybrid spends part of its work budget on local improvement.

Every Hamiltonian cycle is a 1-tree. With node potentials pi, minimum modified-cost 1-tree minus `2sum(pi)` lower-bounds every tour, because a tour gives every node degree 2. Dense Prim computes the non-root MST even when modified costs are negative; the root adds its two cheapest distinct incident edges. Integer potentials and TSPLIB integer distances give an exactly recomputable integer certificate. A connected degree 2 1-tree is a Hamiltonian cycle and certifies its length. Otherwise the gap remains real.

Held–Karp subset dynamic programming has exponential complexity and is limited to small instances. The MTZ mixed-integer formulation gives degree constraints plus order variables excluding subtours. Their agreement and a zero MILP gap certify the small subproblem, not the 150-city problem. Larger heuristic scores are compared to the official TSPLIB proven lengths without loading official tours.
