"""Exercise every optimization preset, live inequalities and mobile layouts in Chromium."""

from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

from optimization_lab.artifacts import sha256, write_json


def feasible(domain, x, y, s):
    if domain == "simplex":
        return x >= 0 and y >= 0 and x + y <= 1
    if domain == "nonnegative_order":
        return x >= 0 and y >= 0 and x >= y
    if domain == "perspective_quadratic":
        return y > 0 and x * x / y <= x - 2 * y * y + 5
    return x >= 0 and s >= 0 and x - y * y >= 0 and x + y <= s + np.sqrt(max(0, x - y * y))


def main():
    root = Path(__file__).resolve().parents[1]
    preview = root / "assets/previews"
    preview.mkdir(exist_ok=True)
    errors = []
    checks = 0
    geometry_checks = 0
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport=dict(width=1280, height=1060))
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto((root / "docs/explorer.html").as_uri())
        page.wait_for_function('document.querySelector("#content svg")!==null')
        assert page.locator("#concepts button").count() == 8

        def select(title, value):
            page.locator(f'select[data-control="{title}"]').select_option(str(value))

        def values(title):
            return page.locator(f'select[data-control="{title}"] option').evaluate_all(
                "(options)=>options.map(o=>o.value)"
            )

        def images():
            page.wait_for_function(
                '[...document.querySelectorAll("#content img")].every(i=>i.complete&&i.naturalWidth>0)'
            )

        def clean():
            assert not errors, errors
            assert "NaN" not in page.locator("#content").inner_text()
            assert "undefined" not in page.locator("#content").inner_text()

        for section in page.locator("#concepts button").all_text_contents():
            page.get_by_role("tab", name=section, exact=True).click()
            assert page.locator("#content").get_attribute("data-section") == section
            if section == "Routing":
                for dataset in values("Instance"):
                    select("Instance", dataset)
                    for method in values("Method"):
                        select("Method", method)
                        for seed in [0, 1, 2]:
                            select("Fitting seed", seed)
                            checks += 1
                            clean()
                            row = page.evaluate(
                                "evidence.routing.find(r=>r.instance===document.querySelector('select[data-control=\"Instance\"]').value&&r.method===document.querySelector('select[data-control=\"Method\"]').value&&r.seed===Number(document.querySelector('select[data-control=\"Fitting seed\"]').value))"
                            )
                            assert (
                                int(
                                    page.locator("[data-route-length]").get_attribute(
                                        "data-route-length"
                                    )
                                )
                                == row["length"]
                            )
                select("Instance", "berlin52")
                select("Method", "acs_2opt")
                select("Fitting seed", 0)
                page.screenshot(path=str(preview / "routing_desktop.png"), full_page=False)
            elif section == "Primal–dual images":
                for dataset in values("Image"):
                    select("Image", dataset)
                    for method in values("Solver"):
                        select("Solver", method)
                        for seed in [0, 1, 2]:
                            select("Noise seed", seed)
                            for penalty in values("TV penalty"):
                                select("TV penalty", penalty)
                                images()
                                clean()
                                checks += 1
                                cursor = page.locator('input[data-control="Recorded iteration"]')
                                for value in ["0", cursor.get_attribute("max")]:
                                    cursor.fill(value)
                                    cursor.dispatch_event("input")
                                    images()
                                assert (
                                    page.locator("[data-iteration]").get_attribute("data-iteration")
                                    is not None
                                )
                select("Image", "photograph")
                select("Solver", "pdhg")
                select("Noise seed", 0)
                select("TV penalty", 0.06)
                images()
                page.screenshot(path=str(preview / "primal_dual_desktop.png"), full_page=True)
            elif section == "Composite optimization":
                for problem in values("Composite problem"):
                    select("Composite problem", problem)
                    for seed in [0, 1, 2]:
                        select("Problem seed", seed)
                        for condition in values("Quadratic condition number"):
                            select("Quadratic condition number", condition)
                            clean()
                            checks += 1
            elif section == "Recommendation":
                for protocol in values("Protocol"):
                    select("Protocol", protocol)
                    for method in values("Method"):
                        select("Method", method)
                        for seed in [0, 1, 2]:
                            select("Fitting seed", seed)
                            for candidate in values("Validation candidate"):
                                select("Validation candidate", candidate)
                                clean()
                                checks += 1
            elif section == "Adaptive search":
                for seed in [0, 1, 2]:
                    select("Search/fitting seed", seed)
                    for strategy in values("Search strategy"):
                        select("Search strategy", strategy)
                        for trial in values("Candidate trial"):
                            select("Candidate trial", trial)
                            clean()
                            checks += 1
            elif section == "Spectral geometry":
                for rank in values("SVD rank"):
                    select("SVD rank", rank)
                    for units in values("PCA units"):
                        select("PCA units", units)
                        images()
                        clean()
                        checks += 1
            elif section == "Convex applications":
                for case in values("Application"):
                    select("Application", case)
                    clean()
                    checks += 1
            else:
                domains = values("Domain")
                points = [
                    (-1.0, 0.0, 0.5),
                    (0.0, 0.0, 0.0),
                    (0.3, 0.4, 0.5),
                    (1.0, 1.0, 0.5),
                    (1.0, 0.0, 0.5),
                    (3.0, 3.0, 2.0),
                    (0.25, 0.5, 0.0),
                    (0.25, -0.5, 2.0),
                ]
                for domain in domains:
                    select("Domain", domain)
                    for x, y, s in points:
                        observed = page.evaluate(
                            "([domain,x,y,s])=>geometryFeasible(domain,x,y,s)", [domain, x, y, s]
                        )
                        assert observed == bool(feasible(domain, x, y, s))
                        geometry_checks += 1
                    for title, value in [
                        ("Coordinate x / u", 0.3),
                        ("Coordinate y / t", 0.4),
                        ("Square-root slice s", 0.5),
                    ]:
                        control = page.locator(f'input[data-control="{title}"]')
                        if control.is_enabled():
                            control.fill(str(value))
                            control.dispatch_event("input")
                    status = page.locator("canvas").get_attribute("data-feasible") == "true"
                    assert status == bool(feasible(domain, 0.3, 0.4, 0.5))
                    clean()
                    checks += 1
        page.set_viewport_size(dict(width=390, height=844))
        for section in page.locator("#concepts button").all_text_contents():
            page.get_by_role("tab", name=section, exact=True).click()
            images()
            clean()
            assert page.evaluate("document.documentElement.scrollWidth<=innerWidth+1"), section
        page.get_by_role("tab", name="Primal–dual images", exact=True).click()
        images()
        page.screenshot(path=str(preview / "primal_dual_mobile.png"), full_page=True)
        browser.close()
    write_json(
        root / "docs/explorer_browser_validation.json",
        dict(
            status="passed",
            html_sha256=sha256(root / "docs/explorer.html"),
            sections=8,
            checked_presets=checks,
            live_geometry_checks=geometry_checks,
            mobile_width=390,
            javascript_errors=errors,
        ),
    )
    print("Browser passed", checks, "presets and", geometry_checks, "live inequality checks")


if __name__ == "__main__":
    main()
