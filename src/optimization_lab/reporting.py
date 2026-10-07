"""Measured tables and figures regenerated from complete numeric evidence."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .artifacts import write_json

EXPECTED = {
    "convex": 7,
    "first_order": 11,
    "spectral": 1,
    "geometry": 1,
    "exact": 1,
    "recommendation_fixed": 3,
    "recommendation": 3,
    "routing": 2,
    "memetic": 2,
    "primal_dual": 18,
    "adaptive_search": 6,
}


def read(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON in {path}: {value}")

    return json.loads(Path(path).read_text(), parse_constant=invalid)


def records(strict=True):
    rows = []
    counts = {family: 0 for family in EXPECTED}
    for path in sorted(Path("results/reference").rglob("run.json")):
        record = read(path)
        family = path.relative_to("results/reference").parts[0]
        if family not in counts:
            raise ValueError(f"Unexpected reference family {family}")
        if record["status"] != "complete" or record["config"].get("quick", False):
            raise ValueError(f"Incomplete or smoke reference {path}")
        metrics = read(path.with_name("metrics.json"))
        rows.append((family, path, record, metrics))
        counts[family] += 1
    if strict and counts != EXPECTED:
        raise ValueError(f"Incomplete reference campaign: {counts}; expected{EXPECTED}")
    return rows, counts


def report():
    protocols, counts = records()
    directory = Path("assets/figures")
    directory.mkdir(parents=True, exist_ok=True)
    Path("docs").mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
            "figure.constrained_layout.use": True,
        }
    )
    figures = []

    def save(fig, name):
        fig.savefig(directory / f"{name}.png")
        fig.savefig(directory / f"{name}.pdf")
        plt.close(fig)
        figures.append(name)

    def data(family, name):
        return np.load(Path("assets/source_data") / family / f"{name}.npz", allow_pickle=False)

    def metric(family, name=None):
        root = Path("results/reference") / family
        return read((root / name if name else root) / "metrics.json")

    for seed in [0, 1, 2]:
        a = data("first_order", f"lasso_seed{seed}")
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        for method in ["ista", "fista", "admm"]:
            h = a[f"{method}_history"]
            for ax, index, label in zip(axes, [0, 4], ["Iteration", "Seconds incl. setup"]):
                ax.semilogy(h[:, index], np.maximum(h[:, 3], 1e-14), label=method.upper())
                ax.set(xlabel=label, ylabel="Feasible primal–dual gap")
        axes[0].legend()
        save(fig, f"lasso_convergence_seed{seed}")
        a = data("first_order", f"nuclear_seed{seed}")
        fig, axes = plt.subplots(1, 3, figsize=(11, 3))
        for method in ["ista", "fista"]:
            h = a[f"{method}_history"]
            axes[0].semilogy(h[:, 0], h[:, 3], label=method.upper())
        axes[0].set(xlabel="Iteration", ylabel="Feasible dual gap")
        axes[0].legend()
        low, high = np.quantile(a["truth"], [0.01, 0.99])
        for ax, key, title in zip(
            axes[1:], ["truth", "fista_x"], ["Synthetic truth", "FISTA reconstruction"]
        ):
            ax.imshow(a[key], vmin=low, vmax=high, cmap="coolwarm")
            ax.set(title=title)
            ax.axis("off")
        save(fig, f"nuclear_completion_seed{seed}")
    a = data("first_order", "quadratic")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, condition in zip(axes, [10, 1000, 100000]):
        for method in ["gradient", "accelerated", "conjugate_gradient", "newton"]:
            h = a[f"{condition}_{method}_history"]
            ax.semilogy(h[:, 0], np.maximum(h[:, 1], 1e-24), label=method)
        ax.set(
            xlabel="Iteration (different costs)",
            ylabel="Known objective gap",
            title=f"Condition{condition:,}",
        )
    axes[0].legend(fontsize=7)
    save(fig, "quadratic_conditioning")
    for label in ["original", "synthetic_seed0", "synthetic_seed1", "synthetic_seed2"]:
        a = data("first_order", f"poisson_{label}")
        m = metric("first_order", f"poisson_{label}")
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        exposure = 1 if label == "original" else 10
        axes[0].plot(a["counts"] / exposure, ".", label="Fit counts / exposure")
        axes[0].plot(a["validation"] / exposure, "x", alpha=0.5, label="Validation")
        for row in m["rows"]:
            if row["rho"] in [0, 0.1, 10]:
                axes[0].plot(a[f"rho{row['rho']}_rate"], label=f"rho={row['rho']}")
        if "truth" in a.files:
            axes[0].plot(a["truth"], "k--", label="Known truth")
        axes[0].set(xlabel="Hour", ylabel="Rate")
        axes[0].legend(fontsize=6)
        axes[1].plot(
            [r["rho"] + 1e-4 for r in m["rows"]],
            [r["validation"]["declared_predictive_floor_nll"] for r in m["rows"]],
            "o-",
        )
        axes[1].set(
            xscale="log",
            xlabel="rho (0 shown at1e-4)",
            ylabel="Validation NLL; explicit floor",
            title=f"Selected rho={m['selected']['rho']}",
        )
        save(fig, f"poisson_{label}")
    a = data("spectral", "qr")
    fig, ax = plt.subplots(figsize=(6, 3))
    for case in ["repeated", "indefinite", "ill_conditioned"]:
        h = a[f"{case}_history"]
        ax.semilogy(h[:, 0], np.maximum(h[:, 1], 1e-18), label=case)
    ax.set(xlabel="Shifted QR step", ylabel="Largest subdiagonal")
    ax.legend()
    save(fig, "symmetric_qr")
    a = data("spectral", "svd")
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    for ax, key, title in zip(
        axes,
        ["original", "rank8", "rank32", "rank128"],
        ["Original, CC BY2.0", "Rank8", "Rank32", "Rank128"],
    ):
        ax.imshow(a[key])
        ax.set_title(title)
        ax.axis("off")
    save(fig, "svd_gallery")
    m = metric("spectral")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(
        [r["parameter_fraction"] for r in m["compression"]],
        [r["display_psnr"] for r in m["compression"]],
        "o-",
    )
    ax.set(xlabel="Factor parameters / pixel values (not codec bytes)", ylabel="Display PSNR(dB)")
    save(fig, "svd_frontier")
    a = data("spectral", "pca")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, label in zip(axes, ["raw_units", "standardized"]):
        scores = a[f"{label}_scores"]
        ax.scatter(scores[:, 0], scores[:, 1], c=a["species"], s=15, cmap="viridis")
        ax.set(title=label.replace("_", " "), xlabel="PC1", ylabel="PC2")
    save(fig, "pca_units")
    a = data("convex", "speed")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].plot(a["distance"] / a["barrier_time"], label="Owned barrier")
    axes[0].plot(a["distance"] / a["reference_time"], "--", label="CLARABEL")
    axes[0].fill_between(
        np.arange(len(a["distance"])),
        a["speed_min"],
        a["speed_max"],
        alpha=0.15,
        label="Allowed speed",
    )
    axes[0].set(xlabel="Segment", ylabel="Speed")
    axes[0].legend(fontsize=7)
    axes[1].plot(np.cumsum(a["barrier_time"]), label="Arrival")
    axes[1].fill_between(
        np.arange(len(a["distance"])), a["prefix_min"], a["prefix_max"], alpha=0.2, label="Window"
    )
    axes[1].set(xlabel="Segment", ylabel="Cumulative arrival time")
    axes[1].legend()
    save(fig, "speed_schedule")
    fig, ax = plt.subplots(figsize=(6, 3))
    h = a["barrier_history"]
    ax.loglog(h[:, 1], h[:, 4], "o-", label="Central-path witness")
    final = metric("convex", "speed")["owned_certificate"]["gap"]
    ax.scatter(h[-1, 1], final, marker="*", s=80, label="Recovered feasible dual")
    ax.set(xlabel="Barrier weight", ylabel="Feasible dual gap")
    ax.legend()
    save(fig, "speed_certificate")
    a = data("convex", "revenue")
    m = metric("convex", "revenue")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].bar(np.arange(4), a["production"])
    axes[0].set(xlabel="Product", ylabel="Production")
    axes[1].bar(np.arange(5), a["resource_use"] / a["capacity"])
    axes[1].axhline(1, color="k", ls="--")
    axes[1].set(xlabel="Resource", ylabel="Capacity utilization")
    save(fig, "revenue_capacity")
    a = data("convex", "blend")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for offset, name in enumerate(["feasible", "min_power", "max_common_log_margin"]):
        axes[0].bar(np.arange(6) + offset * 0.23, a[name + "_weights"], width=0.23, label=name)
    axes[0].set(xlabel="Supplied design index", ylabel="Simplex weight")
    axes[0].legend(fontsize=6)
    m = metric("convex", "blend")
    for offset, row in enumerate(m["rows"]):
        axes[1].bar(
            np.arange(3) + offset * 0.23,
            np.array(row["performance_upper_bounds"]) / a["specifications"],
            width=0.23,
        )
    axes[1].axhline(1, color="k", ls="--")
    axes[1].set(xlabel="Power / delay / area", ylabel="Conditional upper bound / spec")
    save(fig, "geometric_blending")
    a = data("convex", "psd")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(a["input"], cmap="coolwarm")
    axes[0].set(title="Input")
    axes[1].imshow(a["projection"], cmap="coolwarm")
    axes[1].set(title="PSD projection")
    axes[2].plot(a["eigenvalues_before"], "o-", label="Before")
    axes[2].plot(np.linalg.eigvalsh(a["projection"]), "x-", label="After")
    axes[2].legend()
    axes[2].set(ylabel="Eigenvalue")
    save(fig, "psd_projection")
    a = data("convex", "regulator")
    fig, axes = plt.subplots(2, 4, figsize=(12, 5))
    for j, name in enumerate(["frobenius", "group_sum", "group_peak", "element_l1"]):
        axes[0, j].plot(a[name + "_controls"].T)
        axes[0, j].set(title=name, ylabel="Control")
        axes[1, j].plot(a[name + "_states"].T)
        axes[1, j].set(xlabel="Time", ylabel="State")
    save(fig, "regulator_norms")
    a = data("convex", "evacuation")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, name in zip(axes, ["original", "terminal"]):
        image = ax.imshow(a[name + "_population"], aspect="auto", vmin=0, vmax=0.7, cmap="viridis")
        ax.set(title=name, xlabel="State column", ylabel="Node")
    fig.colorbar(image, ax=axes, label="Population")
    save(fig, "evacuation_population")
    a = data("convex", "portfolio")
    m = metric("convex", "portfolio")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for label in ["mean_covariance", "worst_scenario"]:
        rows = [r for r in m["rows"] if r["risk_model"] == label]
        axes[0].plot(
            [r["worst_variance_across_all_scenarios"] for r in rows],
            [r["mean_return"] for r in rows],
            "o-",
            label=label,
        )
        axes[1].plot(
            [r["gamma"] for r in rows], [r["gross_exposure"] for r in rows], "o-", label=label
        )
    axes[0].set(xlabel="Worst supplied-scenario variance", ylabel="Supplied mean return")
    axes[0].legend(fontsize=7)
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="Risk penalty gamma",
        ylabel="Gross exposure (shorting allowed)",
    )
    save(fig, "scenario_risk")

    recommendation = []
    routing = []
    for family, path, record, metrics in protocols:
        if family.startswith("recommendation"):
            recommendation.extend(dict(protocol=family, **row) for row in metrics["rows"])
        if family in ["routing", "memetic"]:
            routing.extend(
                dict(instance=path.parent.name, protocol=family, **row) for row in metrics["rows"]
            )
    recommendations = pd.DataFrame(
        [
            dict(
                protocol=r["protocol"],
                method=r["method"],
                seed=r["seed"],
                test_rmse=r["test"]["rmse"],
                test_mae=r["test"]["mae"],
                validation_rmse=r["validation_rmse"],
                selected_epoch=r["selected_epoch"],
                selected_lambda=r.get("selected_lambda", 0.01),
                selected_learning_rate=r["selected_learning_rate"],
            )
            for r in recommendation
        ]
    )
    routes = pd.DataFrame(routing)
    recommendations.to_csv("docs/recommendation_fits.csv", index=False)
    routes.to_csv("docs/routing_fits.csv", index=False)
    methods = ["global_mean", "bias_only", "als", "sgd", "adam"]
    fig, ax = plt.subplots(figsize=(9, 3))
    for offset, family in enumerate(["recommendation_fixed", "recommendation"]):
        frame = recommendations[recommendations.protocol == family]
        summary = frame.groupby("method").test_rmse.agg(["mean", "std"]).reindex(methods)
        x = np.arange(len(methods)) + offset * 0.3
        ax.bar(
            x,
            summary["mean"],
            width=0.3,
            yerr=summary["std"],
            label="Initial fixed penalty" if offset == 0 else "Validation search",
            alpha=0.8,
            capsize=3,
        )
        for j, method in enumerate(methods):
            ax.scatter(
                np.full(3, x[j]), frame[frame.method == method].test_rmse, c="k", s=9, zorder=3
            )
    ax.set(
        xticks=np.arange(len(methods)) + 0.15,
        xticklabels=methods,
        ylabel="Official test RMSE (lower better)",
        ylim=(0.85, 1.2),
    )
    ax.legend(fontsize=7)
    save(fig, "recommendation_comparison")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, penalty in zip(axes, [0.01, 0.03, 0.1]):
        for method in ["als", "sgd", "adam"]:
            m = next(
                r
                for r in recommendation
                if r["protocol"] == "recommendation" and r["seed"] == 0 and r["method"] == method
            )
            indexes = [i for i, c in enumerate(m["candidates"]) if c["penalty"] == penalty]
            index = min(indexes, key=lambda i: m["candidate_validation_rmse"][i])
            curve = data("recommendation", f"{method}_0_candidate{index}")["history"]
            ax.plot(curve[:, 0], curve[:, 2], label=method.upper())
        ax.set(
            title=f"Shared penalty={penalty}",
            xlabel="Epoch (different step costs)",
            ylabel="Validation RMSE",
            ylim=(0.88, 1.25),
        )
    axes[0].legend()
    save(fig, "recommendation_matched_penalties")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, penalty in zip(axes, [0.01, 0.03, 0.1]):
        for method in ["als", "sgd", "adam"]:
            row = next(
                r
                for r in recommendation
                if r["protocol"] == "recommendation" and r["seed"] == 0 and r["method"] == method
            )
            indexes = [
                i
                for i, candidate in enumerate(row["candidates"])
                if candidate["penalty"] == penalty
            ]
            index = min(indexes, key=lambda i: row["candidate_validation_rmse"][i])
            curve = data("recommendation", f"{method}_0_candidate{index}")["history"]
            ax.plot(curve[:, 0], curve[:, 1], label=method.upper())
        ax.set(
            title=f"Shared penalty={penalty}",
            xlabel="Epoch (different step costs)",
            ylabel="Normalized fitting objective",
        )
    axes[0].legend()
    save(fig, "recommendation_optimization")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    route_methods = sorted(routes.method.unique())
    for ax, name in zip(axes, ["berlin52", "ch150"]):
        frame = routes[routes.instance == name]
        grouped = (
            frame.groupby("method")
            .gap_to_published_optimum_percent.agg(["mean", "std"])
            .reindex(route_methods)
        )
        ax.barh(np.arange(len(route_methods)), grouped["mean"], xerr=grouped["std"], capsize=2)
        for j, method in enumerate(route_methods):
            ax.scatter(
                frame[frame.method == method].gap_to_published_optimum_percent,
                np.full(3, j),
                c="k",
                s=9,
            )
        ax.set(
            yticks=np.arange(len(route_methods)),
            yticklabels=route_methods,
            title=name,
            xlabel="Gap to published optimum (%)",
        )
    save(fig, "routing_comparison")
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    chosen = ["nn_2opt", "ga_ox_reverse_2opt", "acs_2opt", "ortools_gls"]
    for row, name in enumerate(["berlin52", "ch150"]):
        a = data("routing", name)
        extra = data("memetic", name)
        xy = a["coordinates"]
        for ax, method in zip(axes[row], chosen):
            tour = (extra if method.startswith("ga_") else a)[method + "_seed0_tour"]
            line = xy[np.r_[tour, tour[0]]]
            ax.plot(line[:, 0], line[:, 1], "-", lw=0.8)
            ax.scatter(xy[:, 0], xy[:, 1], s=3)
            length = int(
                routes[
                    (routes.instance == name) & (routes.method == method) & (routes.seed == 0)
                ].length.iloc[0]
            )
            ax.set(title=f"{name} {method}\nSeed0 length{length}", aspect="equal")
            ax.axis("off")
    save(fig, "routing_fixed_seed_tours")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, name in zip(axes, ["berlin52", "ch150"]):
        for family, methods in [
            ("routing", ["sa_reverse", "ga_ox_reverse", "as", "acs_2opt"]),
            ("memetic", ["ga_ox_reverse_2opt"]),
        ]:
            a = data(family, name)
            for method in methods:
                h = a[f"{method}_seed0_history"]
                ax.step(h[:, 0], h[:, 1], where="post", label=method)
        ax.axhline(
            {"berlin52": 7542, "ch150": 6528}[name], ls="--", c="k", label="Published optimum"
        )
        ax.set(
            title=f"{name}, fixed seed 0",
            xlabel="Counted distance-edge work",
            ylabel="Best feasible tour length",
        )
    axes[0].legend(fontsize=6)
    save(fig, "routing_convergence")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, name in zip(axes, ["berlin52", "ch150"]):
        a = data("routing", name)
        h = a["one_tree_history"]
        m = metric("routing", name)
        ax.step(h[:, 0], h[:, 1], where="post", label="Owned integer lower bound")
        ax.axhline(m["published_optimum"], c="k", ls="--", label="Published optimum")
        ax.set(title=name, xlabel="Subgradient proposal", ylabel="Exact integer lower bound")
        ax.legend(fontsize=7)
    save(fig, "routing_lower_bounds")
    lines = [
        "# Measured results",
        "",
        f"The release contains {sum(counts.values())} complete protocols. Figures below are regenerated from included numeric evidence.",
        "",
        "## MovieLens official test",
        "",
        "Means and sample SD across three fitting seeds; repeated deterministic baselines have zero fitting variability. User-bootstrap intervals are separately retained in each model record.",
        "",
        "| Method | Test RMSE, mean ± fit SD | Test MAE | Worst fitted-seed RMSE |",
        "|---|---:|---:|---:|",
    ]
    frame = recommendations[recommendations.protocol == "recommendation"]
    for method in ["global_mean", "bias_only", "als", "sgd", "adam"]:
        values = frame[frame.method == method]
        lines.append(
            f"| {method} | {values.test_rmse.mean():.5f} ± {values.test_rmse.std():.5f} | {values.test_mae.mean():.5f} | {values.test_rmse.max():.5f} |"
        )
    lines += [
        "",
        "![Recommendation](../assets/figures/recommendation_comparison.png)",
        "",
        "The initial penalty 0.01 ALS fit overfits: monotone training loss does not imply good validation. Its epoch0 selection and poor test score remain public. The extension selects regularization and stochastic learning rates using validation, with every candidate recorded. Matched-penalty curves avoid comparing differently regularized objective values.",
        "",
        "## Routing",
        "",
        "| Instance | Method | Mean length ± across-run SD | Best / worst length | Mean optimum gap (%) |",
        "|---|---|---:|---:|---:|",
    ]
    for (instance, method), values in routes.groupby(["instance", "method"]):
        lines.append(
            f"| {instance} | {method} | {values.length.mean():.1f} ± {values.length.std():.1f} | {values.length.min()} / {values.length.max()} | {values.gap_to_published_optimum_percent.mean():.3f} |"
        )
    lines += [
        "",
        "![Routing](../assets/figures/routing_comparison.png)",
        "",
        "OR-Tools uses a separately disclosed 5 s time limit; owned methods use a 10M distance-edge-work cap. These are not equal-FLOP budgets. Fixed-seed route figures avoid selecting the best-looking fit. Main textbook variants and separately named local-search hybrids are all included.",
        "",
        "## Numerical checks",
        "",
        "Lasso feasible dual gaps reach1e-7 and nuclear completion1e-6 on all three synthetic problems. Poisson projected KKT residuals reach1e-7, with the conic objective agreeing within numerical tolerance. The repeated / indefinite / ill-conditioned symmetric QR cases converge with small residuals and retain multiplicity.",
        "",
        f"The speed reference objective is {metric('convex', 'speed')['reference']['objective']:.9f}; the owned barrier's refined feasible dual gap is {metric('convex', 'speed')['owned_certificate']['gap']:.3g}. Control, evacuation and robust-risk checks are stored with their own physical / dual residuals. Held–Karp and MTZ MILP agree at length2248 on the 12-city subproblem; an exact integer1-tree bound matches that value.",
        "",
        "These are finite numerical reference checks, not broad superiority, production-deployment or universally certified heuristic-optimality claims. See [protocol](protocol.md), [model cards](model_cards.md), [validation](validation.json) and the individual CSV tables.",
        "",
    ]
    Path("docs/results.md").write_text("\n".join(lines))
    figures.extend(
        path.stem
        for path in sorted(directory.glob("*.png"))
        if path.stem.startswith(("primal_dual_", "adaptive_search"))
        and path.with_suffix(".pdf").exists()
        and path.stem not in figures
    )
    write_json(
        "docs/figure_manifest.json",
        dict(
            figures=figures,
            count=len(figures),
            scope="Numeric source arrays and complete protocol records; PNG and vector PDF for every figure. Photo attribution in third_party.md.",
        ),
    )
    print("Generated", len(figures), "PNG/PDF pairs")
