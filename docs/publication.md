# Owner-managed publication

This repository is prepared and released locally. No GitHub repository is created and no remote push is performed. The owner chooses their GitHub account and repository name.

The local main branch, local v1.1.0 tag, source distribution / wheel and release validation provide a reviewable starting point. Before publishing, run `git status`, inspect the staged file list, and verify that `legacy/originals`, raw MovieLens ratings, private environments and checkpoint directories remain ignored. MIT covers new code / prose; external asset terms are identified in `docs/third_party.md`.

The numerical reference tables and figures are finite measured experiments. Do not present weak routing variants as optimal, validation-selected models as globally converged, the historical rating benchmark as an online recommender, or the supplied shorting-allowed program as investment advice. CI is configured for public-evidence validation and notebook execution, but a hosted GitHub Actions run is not claimed until the owner publishes and observes it.

Suggested repository name: `numerical-optimization-lab`. Suggested description: “Reproducible convex certificates, first-order methods, MovieLens optimization and TSP algorithm comparisons.” Choose the final URL before adding it to CITATION.cff or release metadata; no placeholder account is invented.
