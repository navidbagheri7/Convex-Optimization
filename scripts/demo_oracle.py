"""Source-bound numerical assertions for the currently selected demo controls."""

import math
import re

import numpy as np

_CACHE = None


def close(text, expected):
    if expected is None:
        assert text == "—", (text, expected)
        return
    if isinstance(expected, str):
        assert text == expected, (text, expected)
        return
    found = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?", text, re.I)
    assert found, (text, expected)
    token = found.group().replace(",", "")
    places = len(token.split("e")[0].partition(".")[2])
    exponent = int(token.lower().partition("e")[2] or "0")
    tolerance = 0.501 * 10 ** (exponent - places)
    assert math.isclose(float(token), expected, rel_tol=1e-12, abs_tol=tolerance), (text, expected)


def cards(page, selector, expected):
    actual = page.locator(selector).all_text_contents()
    assert len(actual) == len(expected), (actual, expected)
    for text, value in zip(actual, expected, strict=True):
        close(text, value)


def image_sources(page, selector, expected):
    actual = page.locator(selector).evaluate_all('xs=>xs.map(x=>x.getAttribute("src"))')
    assert actual == expected, (actual, expected)


def check(page):
    global _CACHE
    if _CACHE is None:
        _CACHE = page.evaluate("evidence")
    e = _CACHE
    c = page.locator("#content select").evaluate_all(
        "xs=>Object.fromEntries(xs.map(x=>[x.dataset.control,x.value]))"
    )
    section = page.locator("#content").get_attribute("data-section")
    expected = None
    if section == "Routing":
        r = next(
            r
            for r in e["routing"]
            if r["instance"] == c["Instance"]
            and r["method"] == c["Method"]
            and r["seed"] == int(c["Fitting seed"])
        )
        expected = [
            r["length"],
            r["certificate"]["bound"],
            r["length"] - r["certificate"]["bound"],
            r["gap_to_published_optimum_percent"],
        ]
    elif section == "Primal–dual images":
        r = next(
            r
            for r in e["tv"]
            if r["dataset"] == c["Image"]
            and r["method"] == c["Solver"]
            and r["seed"] == int(c["Noise seed"])
            and r["penalty"] == float(c["TV penalty"])
        )
        expected = [
            r["psnr"],
            r["certificate"]["gap"] / r["pixels"],
            r["iterations"],
            r["oracle_objective_difference"],
        ]
        i = int(page.locator('input[data-control="Recorded iteration"]').input_value())
        v = r["images"][i]
        image_sources(page, "#content img", [r["observed"], v["path"], r["truth"]])
        assert (
            int(page.locator("[data-iteration]").get_attribute("data-iteration")) == v["iteration"]
        )
    elif section == "Recommendation":
        r = next(
            r
            for r in e["recommendation"]
            if r["protocol"] == c["Protocol"]
            and r["method"] == c["Method"]
            and r["seed"] == int(c["Fitting seed"])
        )
        expected = [
            r["test"]["rmse"],
            r["validation_rmse"],
            int(np.argmin(r["candidate_validation_rmse"])),
            r["selected_epoch"],
        ]
    elif section == "Adaptive search":
        r = next(
            r
            for r in e["adaptive"]
            if r["strategy"] == c["Search strategy"] and r["seed"] == int(c["Search/fitting seed"])
        )
        expected = [
            r["test"]["rmse"],
            r["validation_rmse"],
            r["selected_trial"],
            r["trials"][int(c["Candidate trial"])]["validation_rmse"],
        ]
    elif section == "Spectral geometry":
        r = next(r for r in e["spectral"]["compression"] if r["rank"] == int(c["SVD rank"]))
        expected = [
            r["display_psnr"],
            r["parameter_fraction"],
            r["raw_squared_error"],
            abs(r["raw_squared_error"] - r["eckart_young_tail_sum"]),
        ]
        image_sources(page, "#content img", [r["original"], r["prediction"]])
    if expected is not None:
        cards(page, "#content .metric strong", expected)
