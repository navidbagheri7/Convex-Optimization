# Experimental protocol

## Evidence and information access

Complete runs write strict finite JSON with configuration, package versions, source SHA256 and elapsed time. Numeric results are saved without Python object serialization. Smoke runs carry `quick=true`, live in a separate output directory and cannot enter the release report. Independent reference solvers receive the full mathematical model; they are not compared as if they had the same per-step information as a sampled optimizer.

Convex checks use objective agreement, feasible primal / dual witnesses, residuals and analytic or LP references as appropriate. Solver status `optimal_inaccurate`, if present, is explicitly retained rather than silently promoted. Tiny negative differences between an owned objective and a numerical solver's objective are solver tolerance effects, not proof of a violated exact lower bound.

## Recommendation

MovieLens 100K `u1.base` / `u1.test` contain 80,000 / 20,000 ratings. A fixed permutation with split seed 314159 reserves 16,000 validation ratings inside the base, leaving 64,000 fitting ratings. All methods and fitting seeds use identical pair-disjoint roles. The user / item ID vocabulary is defined by the official files, but the global mean, degrees, biases and factors are fitted on fitting ratings only. No demographics are used. Missing ratings are never zero targets.

The original-compatible extension uses rank 32, fixed mean, degree-scaled factor penalty and a separate bias penalty 0.03. The initial fixed-penalty 0.01 campaign is retained in `recommendation_fixed`. Its ALS fitting objective falls monotonically while validation deteriorates. That validation failure motivated the documented extension with the common preset factor-penalty grid[0.01,0.03,0.1]. SGD learning rates[0.2,1.0] and Adam rates[0.0003,0.001,0.003] are preset; all methods run80 epochs with batch 512 for stochastic methods. Each method selects its penalty / rate / epoch using validation RMSE before its final test evaluation. The fixed-study test had already been inspected before designing the extension; the test is a reused benchmark, not a pristine confirmatory study. The extension is motivated by validation, and no claim of statistical hypothesis confirmation is made.

The objective is shared at a **given penalty**. Independently selected penalties may differ; reporting a lower selected objective across different penalties as faster optimization would be invalid. Matched-penalty histories are retained for fair optimization comparisons. ALS solves float64 row / column normal equations. SGD is minibatch SGD with momentum 0.9; Adam uses its standard optimizer. Stochastic fitting uses GPU float32, TF32 disabled. Every validation and test prediction uses exported factors and portable float64 NumPy inference. Epoch0 is a legitimate selectable initial model, retained when every fit overfits.

Scoring clips predictions to[1,5] and also reports unclipped RMSE. Unknown IDs and zero-degree blocks use explicit mean / known-bias fallback. Conditional 95% intervals resample users 2,000 times; they describe the finite test users given a fitted model. Across-fit SD describes the three fitting seeds separately. Mean and bias-only baselines are deterministic on the fixed split; their repeated executions do not create three independent learned baselines. Aggregate per-user squared / absolute errors are public; raw test ratings and factor checkpoints are private. The release includes a fresh-checkpoint audit that recomputes these aggregates locally.

## Routing

Berlin52 and ch150 use TSPLIB EUC_2D distance `floor(norm(delta)+0.5)`. Every route is a permutation, and its objective includes the return edge. Eleven original owned recipes, two explicitly separate memetic GA extensions and a library reference run under seeds 0, 1, 2 on both original instances. Owned methods have a 10,000,000distance-edge-work cap, including NN initialization, full-tour scoring, candidate construction weights and four-edge 2-opt checks. NN + 2-opt can terminate before the cap when locally optimal. Budgets are maxima, not claims of identical total FLOPs.

Crossover, sampling, probability arithmetic and dense pheromone updates are not counted as distance-edge work. Elapsed times and objective counts are also recorded. Annealing temperature follows a fixed work-fraction schedule from twice the median positive edge length down to1. Genetic steady-state populations use64 members, three-way tournament selection, mutation probability 0.3, elitist worst replacement and explicit 2% immigration. The four combinations are OX / CX with swap / reversal mutation. AS globally evaporates then deposits all ant tours; ACS locally updates construction edges and globally updates best-tour edges. ACS + 2-opt performs up to two local-search passes per ant, charged to the same cap.

The memetic extension uses 24 NN / perturbed / two-pass2opt initial members, OX or CX, reversal mutation 0.3, two local-improvement passes per child and an explicit immigrant every 20 children, all under the same counted work cap.

OR-Tools uses PATH_CHEAPEST_ARC and guided local search with a 5 s wall-clock limit. Its callback count and time are reported, separately from the owned-work budget. The seed changes its depot, not a claimed stochastic solver seed; wall-clock termination can vary with machine load. Its scores are a practical library reference, not an equal-compute contest.

Published optimum lengths7542 / 6528 are evaluation references from the official TSPLIB table. No published optimal tour is loaded. The owned integer 1-tree bound uses rounded integer potentials, exact int64 arithmetic with range guards, and dense Prim supporting negative modified costs. Float subgradients only propose potentials; the recorded integer bound is independently recomputable. Step sizes use our computed NN + 2-opt incumbent, never the published optimum. DP and MTZ MILP independently certify a 12-city subproblem. Heuristic optimality is claimed only when the tour equals an independently justified optimum / bound.

## Additional tasks

Lasso uses `.5||Ax-b||² + lambda||x||1`, with sklearn `alpha=lambda/n` and no intercept. Nuclear completion uses a uniform nuclear norm, exact SVD and synthetic unobserved truth; it is a different objective from MovieLens. Penalties are preset before heldout scoring. Proximal convergence stops by feasible dual gap.

The supplied Poisson second day is validation only. Three synthetic tasks reserve independent fit / validation / test days with known rates and exposure 10. The fixed rho grid is selected using validation likelihood with an explicitly declared1e-6 predictive-count floor; strict likelihood and legitimately infinite zero-rate scores are separately reported. Rate KKT checks use a projected mapping to handle active boundaries.

Spectral and convex examples are deterministic numerical problems, so fitting-seed error bars would be inappropriate. Their verification uses independent mathematical identities and solver witnesses. Quadratic GD, acceleration, CG and Newton have different curvature access and per-step costs. Budget-limited slow convergence is retained.
