# Source audit and reconstruction

The supplied archive contains 201 files. Extensions are unreliable: the content audit found 102 PNG images, 42 text files, 30 PDFs, nine other binary files, seven notebooks, six gzip files, three ZIP archives and two JPEG images. `legacy/manifest.json` records hashes and detected content types. All original bytes were preserved and verified; none of the original Python or notebook code was blindly executed.

The author's completed main notebook is the JSON content stored as `lecture1.pdf`. The nominal `Project_Navid_Bagheri_Shouraki.ipynb` is a blank PDF. The two-page main assignment describes regularized MovieLens matrix factorization and simulated annealing, genetic algorithms and ant-colony optimization on Berlin52 / ch150. These are the main applied studies in the reconstruction.

| Original concept | Rebuilt English study | Material correction / extension |
|---|---|---|
| ALS and coordinate optimization | Recommendation | Fit / validation / official test roles; common degree-scaled objective; biases; SGD / Adam; fixed-grid search and all seeds |
| Three annealing neighborhoods | Routing | Swap, reversal and insertion; cyclic tours; real EUC_2D rounding; work accounting |
| Genetic crossovers and mutations | Routing | OX and CX crossed with swap and reversal mutation; elitism and explicit immigration |
| Ant colonies and variants | Routing | AS, ACS and separately named ACS + 2-opt; independent OR-Tools reference |
| Heuristic "optimal" routes | Routing certificates | Published-optimum gaps; exact integer 1-tree bounds; exact DP / MILP on a small subproblem |
| QR / spectral decomposition | Spectral | Tridiagonalization before deflation; repeated eigenvalue multiplicity; nonsymmetric Schur vectors are not eigenvectors |
| Image SVD and Iris PCA | Spectral | Licensed source photograph; [0,1] display; raw tail identity; standardized versus raw units |
| Poisson smoothing | First order | Correct zero-count boundaries; validation day distinguished from independent synthetic test |
| Piecewise revenue and speed | Convex applications | Matched primal / dual LP and sensitivity; full fuel constant and reciprocal travel-time change of variables |
| Geometric circuit blending | Convex applications | Simplex weights over six supplied designs; conditional log-convex performance bounds |
| PSD projection | Convex applications | Explicit symmetry and analytic eigen clipping, including nonsymmetric input |
| Four control norms | Regulator | Original endpoint and horizon; independent minimum-energy / L1 references |
| Population evacuation | Convex applications | Conservation and signed-flow convention; original objective preserved; terminal evacuation added separately |
| Multiple covariance scenarios | Portfolio | Original risk-regularized objective with shorting allowed; dual mixture and gross exposure |
| Transformed feasibility domains | Geometry | Explicit perspective / square-root domains and DCP witnesses; no unsupported inverse transformation |
| Convergence theory | First order extensions | Conditioning, first-order acceleration, CG / Newton information access, sparse and nuclear dual certificates |

Five small numeric parameter scripts were recovered through restricted AST literal / array parsing, without arbitrary execution. `docs/parameter_provenance.json` binds the recovered speed, blend, regulator, evacuation and portfolio numbers to their supplied source hashes. Additional Poisson / revenue values were recovered from the supplied notebook. The public JSON configurations contain numerical problem data, not course scans or reports.

The supplied archive also includes other students' reports and notebooks. Their authorship is not attributed to this repository's author; all public implementation and explanatory text were reconstructed. Raw books, course statements, handwriting, screenshots and reports remain excluded from the release. Every public walkthrough is new and in English.
