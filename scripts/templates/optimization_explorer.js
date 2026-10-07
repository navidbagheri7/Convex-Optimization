"use strict";
const evidence = JSON.parse(document.getElementById("evidence").textContent);
const content = document.getElementById("content");
const colors = ["#087b94", "#d48216", "#a54e8b", "#4a9161", "#a35b4c", "#5c70a4", "#bd6b94", "#899442", "#886040", "#538a9c"];

function name(value) {
  const names = {
    pdhg: "PDHG",
    accelerated_pdhg: "Accelerated PDHG",
    dual_fista: "Dual FISTA",
    admm: "ADMM",
    ista: "ISTA",
    fista: "FISTA",
    als: "ALS",
    sgd: "SGD",
    adam: "Adam",
    tpe: "TPE"
  };
  return names[value] || String(value).replaceAll("_", " ");
}

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function fmt(value, digits = 3) {
  return value === null || value === undefined ? "—" : Number(value).toFixed(digits);
}

function mean(values) {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function sd(values) {
  return values.length < 2 ? 0 : Math.sqrt(values.reduce((a, b) => a + (b - mean(values)) ** 2, 0) / (values.length - 1));
}

function note(text) {
  content.append(el("p", text, "note"));
}

function panel(title) {
  const box = el("div", undefined, "panel");
  if (title) box.append(el("h2", title));
  return box;
}

function metrics(items) {
  const box = el("div", undefined, "metrics");
  items.forEach(([label, value]) => {
    const card = el("div", undefined, "metric");
    card.append(el("strong", String(value)), el("span", label));
    box.append(card);
  });
  content.append(box);
}

function select(title, values, selected) {
  const label = el("label", title);
  const input = el("select");
  input.dataset.control = title;
  values.forEach(value => {
    const option = el("option", name(String(value)));
    option.value = String(value);
    input.append(option);
  });
  if (selected !== undefined) input.value = String(selected);
  label.append(input);
  return [label, input];
}

function slider(title, min, max, value, step = 1) {
  const label = el("label", title);
  const input = el("input");
  input.type = "range";
  Object.assign(input, {
    min,
    max,
    step
  });
  input.value = String(value);
  input.dataset.control = title;
  const output = el("span", String(value));
  input.addEventListener("input", () => output.textContent = input.value);
  label.append(input, output);
  return [label, input];
}

function table(headers, rows) {
  const wrap = el("div", undefined, "table-scroll");
  const node = el("table");
  const head = el("thead");
  const hr = el("tr");
  headers.forEach(h => hr.append(el("th", h)));
  head.append(hr);
  const body = el("tbody");
  rows.forEach(row => {
    const tr = el("tr");
    row.forEach(value => tr.append(el("td", String(value))));
    body.append(tr);
  });
  node.append(head, body);
  wrap.append(node);
  return wrap;
}

function link(path, text = "Source metrics") {
  const a = el("a", text);
  a.href = path;
  return a;
}

function aggregate(rows, key, group = "model") {
  return [...new Set(rows.map(r => r[group]))].map(method => {
    const fits = rows.filter(r => r[group] === method);
    return {
      method,
      fits,
      mean: mean(fits.map(r => r[key])),
      sd: sd(fits.map(r => r[key]))
    };
  });
}

function svgNode(tag, attributes, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes || {}).forEach(([k, v]) => node.setAttribute(k, v));
  if (text !== undefined) node.textContent = text;
  return node;
}

function plot(series, {
  xlabel = "",
  ylabel = "",
  points = false,
  logY = false,
  intervals = false,
  equalAspect = false
} = {}) {
  const svg = svgNode("svg", {
    viewBox: "0 0 740 360",
    class: "plot",
    role: "img",
    "aria-label": `${ylabel} versus ${xlabel}`
  });
  const all = series.flatMap(s => s.points);
  if (!all.length) return svg;
  const yTransform = y => logY ? Math.log10(Math.max(y, 1e-12)) : y;
  let xmin = Math.min(...all.map(p => p.x)),
    xmax = Math.max(...all.map(p => p.x));
  let ymin = Math.min(...all.map(p => yTransform(intervals ? p.lo ?? p.y : p.y))),
    ymax = Math.max(...all.map(p => yTransform(intervals ? p.hi ?? p.y : p.y)));
  if (xmax === xmin) xmax = xmin + 1;
  if (ymax === ymin) ymax = ymin + 1;
  const pad = (ymax - ymin) * .05;
  ymin -= pad;
  ymax += pad;
  const equalScale = Math.min(640 / (xmax - xmin), 260 / (ymax - ymin));
  const sx = equalAspect ? equalScale : 640 / (xmax - xmin),
    sy = equalAspect ? equalScale : 260 / (ymax - ymin);
  const px = x => 380 + sx * (x - (xmin + xmax) / 2),
    py = y => 170 - sy * (yTransform(y) - (ymin + ymax) / 2);
  for (let i = 0; i <= 4; i++) {
    const x = xmin + (xmax - xmin) * i / 4,
      y = ymin + (ymax - ymin) * i / 4;
    svg.append(svgNode("line", {
      x1: 60,
      x2: 700,
      y1: py(logY ? 10 ** y : y),
      y2: py(logY ? 10 ** y : y),
      stroke: "#dbe6ed"
    }));
    svg.append(svgNode("text", {
      x: 4,
      y: py(logY ? 10 ** y : y) + 4
    }, fmt(logY ? 10 ** y : y, logY ? 3 : 2)));
    svg.append(svgNode("text", {
      x: px(x) - 12,
      y: 325
    }, fmt(x, 1)));
  }
  series.forEach((s, index) => {
    const color = s.color || colors[index % colors.length];
    if (points) {
      s.points.forEach(p => {
        if (intervals && p.lo !== undefined) svg.append(svgNode("line", {
          x1: px(p.x),
          x2: px(p.x),
          y1: py(p.lo),
          y2: py(p.hi),
          stroke: color,
          "stroke-opacity": .12
        }));
        const dot = svgNode("circle", {
          cx: px(p.x),
          cy: py(p.y),
          r: p.radius || 2.5,
          fill: p.color || color,
          "fill-opacity": .65
        });
        dot.append(svgNode("title", {}, `${s.label}: x=${fmt(p.x)}, y=${fmt(p.y)}`));
        svg.append(dot);
      });
    } else {
      const line = svgNode("path", {
        d: s.points.map((p, i) => `${i?"L":"M"}${px(p.x)},${py(p.y)}`).join(" "),
        stroke: color,
        "stroke-width": 2.5,
        fill: "none"
      });
      line.append(svgNode("title", {}, s.label));
      svg.append(line);
    }
  });
  svg.append(svgNode("text", {
    x: 320,
    y: 355
  }, xlabel), svgNode("text", {
    x: 60,
    y: 20
  }, ylabel + (logY ? " (log scale)" : "")));
  return svg;
}

function legend(series) {
  const box = el("div", undefined, "legend");
  series.forEach((s, i) => {
    const label = el("span", s.label);
    label.style.setProperty("--color", s.color || colors[i % colors.length]);
    box.append(label);
  });
  return box;
}

function heading(title, text) {
  content.replaceChildren(el("h2", title), el("p", text, "muted"));
}

function controls(...inputs) {
  const box = el('div', undefined, 'controls');
  inputs.forEach(pair => box.append(pair[0]));
  content.append(box);
}

function kpis(items) {
  const box = el('div', undefined, 'metrics');
  items.forEach(([label, value]) => {
    const card = el('div', undefined, 'metric');
    card.append(el('strong', String(value)), el('span', label));
    box.append(card);
  });
  return box;
}

function imageFigure(path, label) {
  const figure = el('figure'),
    image = el('img');
  image.src = path;
  image.alt = label;
  figure.append(image, el('figcaption', label));
  return figure;
}

function lines(rows, x, y, label = 'method') {
  return rows.map(row => ({
    label: name(row[label]),
    points: row.history.map(h => ({
      x: h[x],
      y: h[y]
    }))
  }));
}

function seriesPanel(title, series, options) {
  const box = panel(title);
  box.append(plot(series, options), legend(series));
  return box;
}

function routing() {
  heading('Routes, search and rigorous lower bounds', 'Compare every computed tour on two official TSPLIB instances. The cursor inspects measured best-length history; the map always shows the final tour.');
  const dataset = select('Instance', ['berlin52', 'ch150']),
    method = select('Method', []),
    seed = select('Fitting seed', [0, 1, 2]);
  controls(dataset, method, seed);
  const results = el('div');
  content.append(results);

  function update() {
    results.replaceChildren();
    const row = evidence.routing.find(r => r.instance === dataset[1].value && r.method === method[1].value && r.seed === Number(seed[1].value));
    results.append(kpis([
      ['Feasible final length', row.length],
      ['Certified lower bound', row.certificate.bound],
      ['Upper/lower width', row.length - row.certificate.bound],
      ['Published optimum gap', fmt(row.gap_to_published_optimum_percent, 2) + '%']
    ]));
    const grid = el('div', undefined, 'grid'),
      map = panel('Final cyclic tour'),
      curves = panel('Measured incumbent history');
    const points = row.tour.map(i => ({
      x: row.coordinates[i][0],
      y: row.coordinates[i][1]
    }));
    points.push(points[0]);
    const route = plot([{
      label: 'Final tour',
      points
    }], {
      xlabel: 'TSPLIB coordinate x',
      ylabel: 'TSPLIB coordinate y',
      equalAspect: true
    });
    route.dataset.routeLength = row.length;
    map.append(route);
    map.append(el('p', 'Distances use TSPLIB nearest-integer Euclidean rules. The plotted coordinates are display geometry; reported lengths sum those integer distances.', 'muted'));
    const series = [{
      label: 'Final incumbent progression',
      points: row.history.map(h => ({
        x: h[0],
        y: h[1]
      }))
    }];
    if (row.history.length) {
      curves.append(plot(series, {
        xlabel: 'Counted distance-edge work',
        ylabel: 'Best feasible length'
      }));
      const [label, cursor] = slider('Recorded incumbent index', 0, row.history.length - 1, row.history.length - 1), readout = el('p');

      function inspect() {
        const h = row.history[Number(cursor.value)];
        readout.textContent = 'At ' + h[0].toLocaleString() + ' counted edges: incumbent length ' + h[1] + '; elapsed ' + fmt(h[2], 3) + ' s. Intermediate tours were not saved.';
      }
      cursor.addEventListener('input', inspect);
      inspect();
      curves.append(label, readout);
    } else curves.append(el('p', 'OR-Tools uses its separately declared five-second limit. The public run stores its final tour and callback count; an intermediate trace is unavailable.'));
    grid.append(map, curves);
    results.append(grid);
    results.append(seriesPanel('Exact integer one-tree bound proposals', [{
      label: 'Feasible lower bound',
      points: row.lower_bound_history.map(h => ({
        x: h[0],
        y: h[1]
      }))
    }], {
      xlabel: 'Subgradient proposal',
      ylabel: 'Integer lower bound'
    }));
    results.append(table(['Method', 'Seed', 'Length', 'Optimum gap %', 'Seconds'], evidence.routing.filter(r => r.instance === dataset[1].value).map(r => [name(r.method), r.seed, r.length, fmt(r.gap_to_published_optimum_percent, 2), fmt(r.elapsed_seconds, 3)])));
  }

  function refresh() {
    const values = [...new Set(evidence.routing.filter(r => r.instance === dataset[1].value).map(r => r.method))];
    method[1].replaceChildren(...values.map(v => {
      const option = el('option', name(v));
      option.value = v;
      return option;
    }));
    method[1].value = 'acs_2opt';
    update();
  }
  dataset[1].onchange = refresh;
  method[1].onchange = seed[1].onchange = update;
  refresh();
  note('Owned methods share a ten-million counted-edge budget; their other arithmetic is not equal FLOPs. OR-Tools has a distinct wall-clock budget. A heuristic route is an upper bound; the independently verified one-tree is a lower bound.');
}

function tv() {
  heading('Primal–dual image reconstruction', 'Inspect four solvers for the same isotropic TV problem, with feasible certificates and stored real iteration images. Display clipping does not enter the numerical objective.');
  const dataset = select('Image', ['photograph', 'piecewise']),
    method = select('Solver', ['pdhg', 'accelerated_pdhg', 'dual_fista', 'admm']),
    seed = select('Noise seed', [0, 1, 2]),
    penalty = select('TV penalty', [0.02, 0.06, 0.15], 0.06);
  controls(dataset, method, seed, penalty);
  const results = el('div');
  content.append(results);

  function update() {
    results.replaceChildren();
    const row = evidence.tv.find(r => r.dataset === dataset[1].value && r.method === method[1].value && r.seed === Number(seed[1].value) && r.penalty === Number(penalty[1].value));
    results.append(kpis([
      ['Final float PSNR', fmt(row.psnr, 2) + ' dB'],
      ['Final feasible gap/pixel', row.certificate.gap.toExponential(2) + ' total'],
      ['Iterations', row.iterations],
      ['Small oracle objective error', row.oracle_objective_difference.toExponential(2)]
    ]));
    // Show the final gap per pixel rather than a display-image statistic.
    results.querySelectorAll('.metric strong')[1].textContent = (row.certificate.gap / row.pixels).toExponential(2);
    const box = panel('Stored optimization iterates'),
      figures = el('div', undefined, 'images');
    const reconstructed = imageFigure(row.images.at(-1).path, 'Reconstruction at iteration ' + row.images.at(-1).iteration);
    figures.append(imageFigure(row.observed, 'Observed float image, clipped for display'), reconstructed, imageFigure(row.truth, 'Reference image, fixed before noise'));
    const [label, cursor] = slider('Recorded iteration', 0, row.images.length - 1, row.images.length - 1);

    function inspect() {
      const sample = row.images[Number(cursor.value)];
      reconstructed.querySelector('img').src = sample.path;
      reconstructed.querySelector('figcaption').textContent = 'Reconstruction at iteration ' + sample.iteration;
      reconstructed.dataset.iteration = sample.iteration;
      label.querySelector('span').textContent = String(sample.iteration);
    }
    cursor.addEventListener('input', inspect);
    inspect();
    box.append(figures, label);
    results.append(box);
    const matched = evidence.tv.filter(r => r.dataset === row.dataset && r.seed === row.seed && r.penalty === row.penalty),
      series = lines(matched, 0, 3);
    results.append(seriesPanel('Same problem, feasible certificate progress', series, {
      xlabel: 'Iterations, unequal step cost',
      ylabel: 'Absolute primal–dual gap',
      logY: true
    }));
    results.append(table(['Solver', 'Iterations', 'Gap / pixel', 'PSNR', 'Seconds'], matched.map(r => [name(r.method), r.iterations, (r.certificate.gap / r.pixels).toExponential(2), fmt(r.psnr, 3), fmt(r.elapsed_seconds, 4)])));
  }
  [dataset, method, seed, penalty].forEach(pair => pair[1].onchange = update);
  update();
  note('Opposite image edges connect under the declared periodic boundary. Penalties are fixed comparisons, not selected using reference-image PSNR. The 16×16 conic oracle solves a separate crop; the large-image certificate is its own feasible dual gap.');
}

function composite() {
  heading('Composite objectives and conditioning', 'Separate solver convergence from statistical reconstruction. Feasible dual gaps certify Lasso and nuclear-norm objectives; quadratic solvers have different information and per-step costs.');
  const problem = select('Composite problem', ['lasso', 'nuclear']),
    seed = select('Problem seed', [0, 1, 2]),
    condition = select('Quadratic condition number', [10, 1000, 100000], 1000);
  controls(problem, seed, condition);
  const results = el('div');
  content.append(results);

  function update() {
    results.replaceChildren();
    const rows = evidence.composite.filter(r => r.problem === problem[1].value && r.seed === Number(seed[1].value));
    results.append(seriesPanel('Feasible composite gaps', lines(rows, 0, 3), {
      xlabel: 'Iterations',
      ylabel: 'Feasible primal–dual gap',
      logY: true
    }));
    results.append(table(['Method', 'Iterations', 'Primal objective', 'Dual objective', 'Gap'], rows.map(r => [name(r.method), r.iterations, fmt(r.certificate.objective, 5), fmt(r.certificate.dual_objective, 5), r.certificate.gap.toExponential(2)])));
    const quad = evidence.quadratic.filter(r => r.condition === Number(condition[1].value));
    results.append(seriesPanel('Quadratic objective gap', lines(quad, 0, 1), {
      xlabel: 'Iterations, unequal information/cost',
      ylabel: 'Known objective gap',
      logY: true
    }));
    results.append(table(['Method', 'Steps', 'Tolerance reached', 'Gradient norm'], quad.map(r => [name(r.method), r.iterations, r.reached_tolerance, r.gradient_norm.toExponential(2)])));
  }
  [problem, seed, condition].forEach(pair => pair[1].onchange = update);
  update();
  note('GD/acceleration use spectral bounds; conjugate gradient uses Hessian products; Newton solves with the full Hessian. Slow and budget-limited runs are retained. Certificates concern the fitted convex objective, not prediction quality on future data.');
}

function recommendation() {
  heading('Recommendation, regularization and overfitting', 'Inspect every validation candidate while keeping the deployed winner and its official test result separate. All methods use the same degree-regularized biased factor objective.');
  const profile = select('Protocol', ['recommendation', 'recommendation_fixed']),
    method = select('Method', ['global_mean', 'bias_only', 'als', 'sgd', 'adam'], 'als'),
    seed = select('Fitting seed', [0, 1, 2]),
    candidate = select('Validation candidate', []);
  controls(profile, method, seed, candidate);
  const results = el('div');
  content.append(results);

  function row() {
    return evidence.recommendation.find(r => r.protocol === profile[1].value && r.method === method[1].value && r.seed === Number(seed[1].value));
  }

  function update() {
    results.replaceChildren();
    const r = row(),
      i = Number(candidate[1].value),
      c = r.candidates[i],
      winning = r.candidate_validation_rmse.indexOf(Math.min(...r.candidate_validation_rmse));
    results.append(kpis([
      ['Deployed winner test RMSE', fmt(r.test.rmse, 4)],
      ['Winner validation RMSE', fmt(r.validation_rmse, 4)],
      ['Winner candidate', winning],
      ['Winner selected epoch', r.selected_epoch]
    ]));
    const grid = el('div', undefined, 'grid');
    grid.append(seriesPanel('Selected candidate fitting objective', [{
      label: 'Candidate ' + i,
      points: c.history.map(h => ({
        x: h[0],
        y: h[1]
      }))
    }], {
      xlabel: 'Epoch',
      ylabel: 'Degree-regularized objective'
    }), seriesPanel('Selected candidate validation RMSE', [{
      label: 'Candidate ' + i,
      points: c.history.map(h => ({
        x: h[0],
        y: h[2]
      }))
    }], {
      xlabel: 'Epoch',
      ylabel: 'Validation RMSE'
    }));
    results.append(grid);
    const box = panel('Candidate parameters and deployment uncertainty');
    box.append(el('p', 'Inspected candidate ' + i + ': factor penalty ' + c.penalty + ', learning rate ' + (c.learning_rate ?? 'not applicable') + '. Its best validation RMSE is ' + fmt(r.candidate_validation_rmse[i], 5) + '. Test metrics above always belong to candidate ' + winning + '.'));
    box.append(el('p', 'Winner conditional user-bootstrap test RMSE interval: [' + r.test.conditional_user_bootstrap_rmse_ci95.map(v => fmt(v, 4)).join(', ') + ']. Warm-item ratings: ' + r.test.warm.n_ratings + '; cold-item ratings: ' + r.test.cold_item.n_ratings + '.'));
    results.append(box);
  }

  function refresh() {
    const r = row();
    candidate[1].replaceChildren(...r.candidates.map((c, i) => {
      const option = el('option', 'Candidate ' + i + ' · λ ' + c.penalty);
      option.value = i;
      return option;
    }));
    candidate[1].value = r.candidate_validation_rmse.indexOf(Math.min(...r.candidate_validation_rmse));
    update();
  }
  [profile, method, seed].forEach(pair => pair[1].onchange = refresh);
  candidate[1].onchange = update;
  refresh();
  note('Official MovieLens-100K provides 64,000 fitting, 16,000 validation and 20,000 test ratings. RMSE uses bounded 1–5 predictions. These random rating splits do not estimate future chronological or production recommendation performance.');
}

function adaptive() {
  heading('Adaptive hyperparameter search', 'Compare sequential TPE with random search using the same candidate and epoch budgets. The first eight proposals match exactly, and no trial uses test scores for selection.');
  const seed = select('Search/fitting seed', [0, 1, 2]),
    strategy = select('Search strategy', ['random', 'tpe'], 'tpe'),
    trial = select('Candidate trial', []);
  controls(seed, strategy, trial);
  const results = el('div');
  content.append(results);

  function row() {
    return evidence.adaptive.find(r => r.seed === Number(seed[1].value) && r.strategy === strategy[1].value);
  }

  function update() {
    results.replaceChildren();
    const r = row(),
      t = r.trials[Number(trial[1].value)];
    results.append(kpis([
      ['Selected campaign test RMSE', fmt(r.test.rmse, 4)],
      ['Selected validation RMSE', fmt(r.validation_rmse, 4)],
      ['Deployed trial', r.selected_trial],
      ['Inspected trial validation', fmt(t.validation_rmse, 4)]
    ]));
    const curves = evidence.adaptive.filter(v => v.seed === r.seed).map(v => {
      let best = Infinity;
      return {
        label: v.strategy,
        points: v.trials.map(t => ({
          x: t.number + 1,
          y: best = Math.min(best, t.validation_rmse)
        }))
      };
    });
    results.append(seriesPanel('Best validation result after each completed fit', curves, {
      xlabel: 'Completed candidate fits',
      ylabel: 'Best validation RMSE'
    }));
    results.append(seriesPanel('Inspected trial learning curve', [{
      label: 'Trial ' + t.number,
      points: t.history.map(h => ({
        x: h[0],
        y: h[2]
      }))
    }], {
      xlabel: 'Epoch',
      ylabel: 'Validation RMSE'
    }));
    const box = panel('Inspected trial parameters');
    box.append(table(['Parameter', 'Value'], Object.entries(t.parameters)));
    box.append(el('p', 'Selected epoch ' + t.selected_epoch + '; ' + t.generator_updates + ' optimizer updates; ' + fmt(t.elapsed_seconds, 3) + ' s. Test results above are the campaign winner, not this arbitrary trial.'));
    results.append(box);
    results.append(table(['Search', 'Seed', 'Selected validation RMSE', 'Official test RMSE'], evidence.adaptive.map(v => [v.strategy, v.seed, fmt(v.validation_rmse, 5), fmt(v.test.rmse, 5)])));
  }

  function refresh() {
    const r = row();
    trial[1].replaceChildren(...r.trials.map(t => {
      const option = el('option', 'Trial ' + t.number);
      option.value = t.number;
      return option;
    }));
    trial[1].value = r.selected_trial;
    update();
  }
  seed[1].onchange = strategy[1].onchange = refresh;
  trial[1].onchange = update;
  refresh();
  note('Six campaigns contain 96 full fits. Rank and proposed parameters change runtime, so equal trials are not equal FLOPs. Three seeds combine initialization, batch order and search variability. More sophisticated search need not win within a small budget.');
}

function spectral() {
  heading('Spectral geometry and approximation', 'Separate truncated-SVD mathematical optimality from clipped visual quality, and observe how feature units change PCA geometry.');
  const rank = select('SVD rank', evidence.spectral.compression.map(r => r.rank), 32),
    units = select('PCA units', ['raw_units', 'standardized']);
  controls(rank, units);
  const results = el('div');
  content.append(results);

  function update() {
    results.replaceChildren();
    const r = evidence.spectral.compression.find(v => v.rank === Number(rank[1].value));
    results.append(kpis([
      ['Clipped display PSNR', fmt(r.display_psnr, 2) + ' dB'],
      ['Factor/pixel parameter ratio', fmt(r.parameter_fraction, 3)],
      ['Raw Frobenius squared error', fmt(r.raw_squared_error, 3)],
      ['Tail-identity residual', Math.abs(r.raw_squared_error - r.eckart_young_tail_sum).toExponential(2)]
    ]));
    const box = panel('Original and truncated reconstruction'),
      figures = el('div', undefined, 'images');
    figures.append(imageFigure(r.original, 'Attributed original photograph'), imageFigure(r.prediction, 'Channel SVD rank ' + r.rank));
    box.append(figures);
    results.append(box);
    const data = evidence.spectral.pca,
      scores = data[units[1].value + '_scores'];
    const groups = [0, 1, 2].map(species => ({
      label: 'Iris species ' + species,
      points: scores.filter((p, i) => data.species[i] === species).map(p => ({
        x: p[0],
        y: p[1]
      }))
    }));
    results.append(seriesPanel('Iris PCA descriptive geometry', groups, {
      xlabel: 'PC1',
      ylabel: 'PC2',
      points: true
    }));
    results.append(table(['Symmetric QR case', 'Converged', 'Eigen residual', 'Orthogonality residual'], evidence.spectral.qr.map(v => [v.case, v.converged, v.relative_eigen_residual.toExponential(2), v.orthogonality_residual.toExponential(2)])));
  }
  rank[1].onchange = units[1].onchange = update;
  update();
  note('Factor counts are not compressed-file codec sizes. The SVD tail identity holds for unclipped reconstruction; display PSNR uses clipped pixels. PCA is descriptive, with no held-out prediction claim. Schur columns for a nonnormal matrix are invariant-basis vectors, not generally individual eigenvectors.');
}

function applications() {
  heading('Convex models, physical constraints and certificates', 'Inspect seven reconstructed applications. Each model preserves its original assumptions and identifies separately declared extensions.');
  const problem = select('Application', evidence.applications.map(r => r.case), 'speed');
  controls(problem);
  const results = el('div');
  content.append(results);

  function update() {
    results.replaceChildren();
    const r = evidence.applications.find(v => v.case === problem[1].value),
      a = r.arrays,
      m = r.metrics;
    let series = [],
      xlabel = '',
      ylabel = '';
    if (r.case === 'speed') {
      series = [{
        label: 'Owned barrier speed',
        points: a.distance.map((d, i) => ({
          x: i,
          y: d / a.barrier_time[i]
        }))
      }, {
        label: 'CLARABEL speed',
        points: a.distance.map((d, i) => ({
          x: i,
          y: d / a.reference_time[i]
        }))
      }];
      xlabel = 'Segment';
      ylabel = 'Speed';
      results.append(kpis([
        ['Owned feasible dual gap', m.owned_certificate.gap.toExponential(2)],
        ['Constraint violation', m.owned_certificate.constraint_violation.toExponential(2)],
        ['Reference objective', fmt(m.reference.objective, 6)],
        ['Max time difference', m.max_time_difference.toExponential(2)]
      ]));
    } else if (r.case === 'revenue') {
      series = [{
        label: 'Resource capacity use',
        points: a.resource_use.map((v, i) => ({
          x: i,
          y: v / a.capacity[i]
        }))
      }];
      xlabel = 'Resource';
      ylabel = 'Utilization';
    } else if (r.case === 'blend') {
      series = ['feasible', 'min_power', 'max_common_log_margin'].map(v => ({
        label: name(v),
        points: a[v + '_weights'].map((y, i) => ({
          x: i,
          y
        }))
      }));
      xlabel = 'Measured design index';
      ylabel = 'Simplex weight';
    } else if (r.case === 'psd') {
      series = [{
        label: 'Input eigenvalues',
        points: a.eigenvalues_before.map((y, i) => ({
          x: i,
          y
        }))
      }, {
        label: 'Projection eigenvalues',
        points: a.eigenvalues_before.map((y, i) => ({
          x: i,
          y: Math.max(0, y)
        }))
      }];
      xlabel = 'Eigenvalue index';
      ylabel = 'Eigenvalue';
    } else if (r.case === 'regulator') {
      series = ['frobenius', 'group_sum', 'group_peak', 'element_l1'].map(v => ({
        label: name(v),
        points: a[v + '_controls'][0].map((y, i) => ({
          x: i,
          y
        }))
      }));
      xlabel = 'Control step';
      ylabel = 'First control component';
    } else if (r.case === 'evacuation') {
      series = ['original', 'terminal'].map(v => ({
        label: name(v),
        points: a[v + '_population'][0].map((y, i) => ({
          x: i,
          y
        }))
      }));
      xlabel = 'State time column';
      ylabel = 'Population of first node';
    } else if (r.case === 'portfolio') {
      series = ['mean_covariance', 'worst_scenario'].map(v => ({
        label: name(v),
        points: m.rows.filter(t => t.risk_model === v).map(t => ({
          x: t.worst_variance_across_all_scenarios,
          y: t.mean_return
        }))
      }));
      xlabel = 'Worst supplied-scenario variance';
      ylabel = 'Supplied mean return';
    }
    results.append(seriesPanel('Saved solution behavior', series, {
      xlabel,
      ylabel
    }));
    const box = panel('Declared assumptions and numerical evidence');
    box.append(el('p', m.scope), link('../results/reference/convex/' + r.case+'/metrics.json'), el('pre', JSON.stringify(Object.fromEntries(Object.entries(m).filter(([k]) => k !== 'scope')), null, 2)));
    box.querySelector('pre').style.cssText = 'white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:360px;overflow:auto';
    results.append(box);
  }
  problem[1].onchange = update;
  update();
  note('The risk example uses supplied course covariances and allows shorting; it is a numerical model, not measured investment performance. Different control norms, terminal evacuation constraints and blending objectives define different problems.');
}

function geometryFeasible(domain, x, y, s) {
  if (domain === 'simplex') return x >= 0 && y >= 0 && x + y <= 1;
  if (domain === 'nonnegative_order') return x >= 0 && y >= 0 && x >= y;
  if (domain === 'perspective_quadratic') return y > 0 && x * x / y <= x - 2 * y * y + 5;
  const radicand = x - y * y;
  return x >= 0 && s >= 0 && radicand >= 0 && x + y <= s + Math.sqrt(radicand);
}

function geometry() {
  heading('Convex domain geometry — live analytical illustration', 'Move a point through four transformed feasibility domains. These browser calculations evaluate the stated inequalities; they do not solve a new optimization problem.');
  const domain = select('Domain', evidence.geometry.rows.map(r => r.domain)),
    x = slider('Coordinate x / u', -1, 3, 0.3, 0.01),
    y = slider('Coordinate y / t', -1, 3, 0.4, 0.01),
    s = slider('Square-root slice s', 0, 2, 0.5, 0.01);
  controls(domain, x, y, s);
  const results = el('div');
  content.append(results);

  function update() {
    results.replaceChildren();
    s[1].disabled = domain[1].value !== 'concave_sqrt_hypograph';
    const xv = Number(x[1].value),
      yv = Number(y[1].value),
      sv = Number(s[1].value),
      feasible = geometryFeasible(domain[1].value, xv, yv, sv);
    results.append(kpis([
      ['Live point status', feasible ? 'Feasible' : 'Outside domain'],
      ['x / u', fmt(xv, 2)],
      ['y / t', fmt(yv, 2)],
      ['s slice', fmt(sv, 2)]
    ]));
    const box = panel('Feasibility slice on [−1, 3]²'),
      canvas = el('canvas');
    canvas.width = canvas.height = 480;
    canvas.style.cssText = 'max-width:100%;height:auto;display:block;margin:auto';
    canvas.setAttribute('aria-label', 'Feasible domain shaded teal; live point marked orange');
    canvas.dataset.feasible = String(feasible);
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#f2f7fb';
    ctx.fillRect(0, 0, 480, 480);
    for (let i = 0; i < 160; i++)
      for (let j = 0; j < 160; j++)
        if (geometryFeasible(domain[1].value, -1 + 4 * i / 160, -1 + 4 * j / 160, sv)) {
          ctx.fillStyle = '#bfe1e8';
          ctx.fillRect(3 * i, 477 - 3 * j, 3, 3);
        }
    ctx.strokeStyle = '#607586';
    ctx.beginPath();
    ctx.moveTo(120, 0);
    ctx.lineTo(120, 480);
    ctx.moveTo(0, 360);
    ctx.lineTo(480, 360);
    ctx.stroke();
    ctx.fillStyle = '#d48216';
    ctx.beginPath();
    ctx.arc((xv + 1) * 120, 480 - (yv + 1) * 120, 6, 0, 2 * Math.PI);
    ctx.fill();
    box.append(canvas, el('p', 'Horizontal coordinate increases right; vertical coordinate increases up. Orange marks the live point.'));
    results.append(box);
    const witness = evidence.geometry.rows.find(r => r.domain === domain[1].value),
      details = panel('Independent conic witness');
    details.append(el('p', 'Saved full-domain witness: ' + witness.witness.map(v => fmt(v, 6)).join(', ') + '. Maximum violation: ' + witness.maximum_violation.toExponential(2) + '. A three-variable witness need not belong to a different displayed s slice.'), link('../results/reference/geometry/metrics.json'));
    results.append(details);
  }
  domain[1].onchange = update;
  [x, y, s].forEach(pair => pair[1].oninput = update);
  update();
  note(evidence.geometry.scope);
}
const sections = {
  'Routing': routing,
  'Primal–dual images': tv,
  'Composite optimization': composite,
  'Recommendation': recommendation,
  'Adaptive search': adaptive,
  'Spectral geometry': spectral,
  'Convex applications': applications,
  'Feasibility geometry': geometry
};
const navigation = document.getElementById('concepts');
Object.entries(sections).forEach(([title, render]) => {
  const button = el('button', title);
  button.setAttribute('role', 'tab');
  button.setAttribute('aria-selected', 'false');
  button.onclick = () => {
    navigation.querySelectorAll('button').forEach(b => b.setAttribute('aria-selected', String(b === button)));
    content.dataset.section = title;
    render();
  };
  navigation.append(button);
});
navigation.querySelector('button').click();
