'use strict';

const statusText = document.getElementById('board-status');
const tbody = document.getElementById('standings');
const pagesOwner = location.hostname.match(/^([a-z0-9-]+)\.github\.io$/i);
const pagesRepo = location.pathname.split('/').filter(Boolean)[0];
if (pagesOwner && pagesRepo && /^[a-z0-9._-]+$/i.test(pagesRepo)) {
  const repo = 'https://github.com/' + pagesOwner[1] + '/' + encodeURIComponent(pagesRepo) + '/blob/HEAD/';
  document.getElementById('enter').href = repo + 'docs/runs.md#enter-the-shared-scoreboard';
  document.getElementById('github-board').href = repo + 'lineage/scoreboard.md';
}

function link(label, href) {
  const a = document.createElement('a');
  a.href = href;
  a.textContent = label;
  return a;
}

function drawRow({id, parent, score}) {
  const row = tbody.insertRow();
  if (score.place === 1) row.className = 'board-leader';
  row.insertCell().textContent = score.place ?? '—';
  const runner = document.createElement('th');
  runner.scope = 'row';
  runner.className = 'board-runner';
  runner.textContent = id;
  if (parent) {
    const sub = document.createElement('small');
    sub.append('From ', link(parent, './?run=' + encodeURIComponent(parent)));
    runner.append(sub);
  }
  row.append(runner);
  const finishes = row.insertCell();
  finishes.className = 'numeric';
  finishes.textContent = score.finishes + '/16';
  for (const key of ['median_ms', 'best_ms']) {
    const cell = row.insertCell();
    cell.className = key === 'best_ms' ? 'numeric best-lap' : 'numeric';
    cell.textContent = score[key] === null ? '—' : (score[key] / 1000).toFixed(3) + ' s';
  }
  const watch = link('Watch', './?run=' + encodeURIComponent(id));
  watch.setAttribute('aria-label', 'Watch ' + id + ' in 3D');
  row.insertCell().append(watch);
}

async function loadBoard() {
  try {
    const response = await fetch('lineage/index.json');
    if (!response.ok) throw new Error('Archive unavailable');
    const entries = await response.json();
    const rows = entries.filter(entry => entry.score);
    // Rank is assigned once by run_tree.py; the site and GitHub share ties.
    rows.sort((a,b) => (a.score.place ?? Infinity) - (b.score.place ?? Infinity) || a.id.localeCompare(b.id));
    rows.forEach(drawRow);
    document.getElementById('table-wrap').hidden = rows.length === 0;
    statusText.textContent = rows.length ? '' : 'No drivers have completed the standard evaluation yet.';
  } catch {
    statusText.textContent = 'Standings could not load. The GitHub scoreboard is linked below.';
  }
}
loadBoard();
