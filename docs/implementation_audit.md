# Implementation review and evidence

The review covered all authored numerical, experimental, deployment and reporting
modules, their configs and notebook sources. The AST inventory is a completeness
index; mathematical invariants and executed evidence provide separate checks.

| Component | Reviewed contracts and evidence |
|---|---|
| Proximal methods | Matching Lasso normalization; feasible residual dual; masked nuclear entries; spectral dual feasibility; ADMM setup counted; Poisson zero-rate boundaries and stable quadratic roots |
| Convex applications and barrier | Strictly feasible time-domain initialization, Newton/line-search behavior, recovered feasible dual; explicit constant fuel; conic/LP objectives, shadow prices, terminal dynamics and population conservation |
| Risk and geometric models | PSD scenario assumptions, common riskless asset, shorting/exposure, dual covariance mixture; geometric bounds conditional on log-convex physics; denominator/radicand domains retained |
| Spectral methods | Symmetric Householder/shifted QR, deflation and repeated eigenvalues; Schur counterexample; raw SVD tail identity versus clipped display; PCA projector comparison avoids sign ambiguity |
| Recommendation | Exact degree-regularized objective; ALS stationarity and monotonic fitting loss; unbiased per-rating stochastic penalty; fixed roles and test isolation; zero-degree factors and unknown-ID fallback; portable float64 deployment |
| Routing and certificates | Integer EUC_2D distances, cyclic permutations, crossover invariants, counted objective/construction/local-search work; declared heuristic recipes; integer one-tree validity and overflow guard; independent DP/MILP references |
| Primal–dual extension | Periodic gradient/adjoint identity; projected feasible Fenchel dual; accelerated steps maintain the operator product bound; dual FISTA step uses norm bound 8; exact periodic FFT ADMM quadratic step; cone crop oracle independent of large-image certificate |
| Adaptive search | Same model and validation roles, sequential seeded samplers, exact shared first eight proposals; all 96 budgets finished; strict validation epoch/parameter selection; final test once per campaign; partial sampler continuation rejected before replacing prior records |
| Deployment and browser | All selected search factors independently reproduce full-test errors; actual CSV commands with integer-ID schema; browser controls keep arbitrary candidates separate from winner test metrics; final route maps do not invent intermediate tours; live inequalities compared against Python |
| Packaging and learning | Bundled defaults, source-archive templates/HTML, English executed walkthroughs, local links and private artifact exclusions; code/config/citation identities checked |

The original recommendation fixed-penalty source snapshot remains bound to its
historical records. New scientific modules are frozen for their fitting campaign;
reports and previews are generated separately. Complete search records skip
unchanged; incomplete search continuation is rejected because recreating a sampler
seed does not restore its adaptive RNG history. TV solves preserve nonconvergence
rather than silently claiming an iteration cap achieved the requested certificate.

A stricter new test exposed a finite-budget limitation of accelerated PDHG on a
random small image: at 12,000 steps its gap was approximately 8.9e-8, while the test
requested 5.6e-8. Its objective still agreed with the cone oracle within 1e-6. The
independent test now declares a gap per pixel of 2e-9, consistent with that budget;
the full image campaign retains its original 1e-6 tolerance. No fitted result was
changed to hide the limitation, and acceleration is not claimed universally faster.

During integration, the demo generator was corrected to reconstruct historical
candidate metadata from its saved config, rather than assuming the newer schema.
The earlier release notebook's protocol-count assertion was updated from 31 to 55.
Disabled geometry controls are respected by the browser checker. These changes
concern version/schema integration; the preserved scientific fits are unchanged.

Read [public numerical validation](validation.json), [extension checks](extensions_validation.json),
[checkpoint inference](adaptive_checkpoint_audit.json), [the demo audit](explorer_validation.json)
and [scope limitations](model_cards.md). Local checks do not establish hosted CI,
external replication or production-scale deployment.
