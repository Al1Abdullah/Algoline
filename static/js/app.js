/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
   APPLICATION STATE
   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
window.APP = {
  columns: [],
  numericColumns: [],
  target: '',
  task: '',
  leaderboard: null,
  metrics: [],
  bestModel: '',
  nPredictions: 0
};

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
   NAVIGATION
   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
function showSection(name) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const sec = document.getElementById('sec-' + name);
  if (sec) sec.classList.add('active');
  const nav = document.querySelector('.nav-item[data-section="' + name + '"]');
  if (nav) nav.classList.add('active');
}

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
   TABLE RENDERER
   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
function renderTable(containerId, rows, columns) {
  const el = document.getElementById(containerId);
  if (!el || !rows || !columns) { if (el) el.innerHTML = ''; return; }
  let html = '<table class="data-table"><thead><tr>';
  columns.forEach(c => { html += '<th>' + escHtml(String(c)) + '</th>'; });
  html += '</tr></thead><tbody>';
  rows.forEach(r => {
    html += '<tr>';
    columns.forEach(c => {
      const val = r[c] !== undefined && r[c] !== null ? String(r[c]) : '';
      html += '<td>' + escHtml(val) + '</td>';
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  el.innerHTML = html;
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
   PLOTLY HELPER
   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
function renderPlotly(divId, figure) {
  const el = document.getElementById(divId);
  if (!el) return;
  // Always clear any existing content (spinner, old chart) first
  el.innerHTML = '';
  if (!figure || !figure.data) return;
  const style = getComputedStyle(document.documentElement);
  const textColor = style.getPropertyValue('--chart-text').trim() || '#71717a';
  const gridColor = style.getPropertyValue('--chart-grid').trim() || 'rgba(30,27,75,0.15)';
  const layout = Object.assign({}, figure.layout || {}, {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter, sans-serif', size: 12, color: textColor },
    margin: figure.layout && figure.layout.margin ? figure.layout.margin : { l: 40, r: 20, t: 40, b: 40 }
  });
  layout.xaxis = Object.assign({}, layout.xaxis || {}, { gridcolor: gridColor, zeroline: false });
  layout.yaxis = Object.assign({}, layout.yaxis || {}, { gridcolor: gridColor, zeroline: false });
  try {
    Plotly.newPlot(divId, figure.data, layout, { responsive: true, displayModeBar: false });
  } catch (e) {
    el.innerHTML = '<p style="color:#ef4444;padding:20px;text-align:center">Chart rendering failed</p>';
  }
}

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
   FETCH WRAPPER
   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
async function apiFetch(url, options) {
  try {
    const res = await fetch(url, options);
    if (!res.ok) {
      let msg = 'Request failed';
      try { const e = await res.json(); msg = e.detail || e.error || e.message || msg; } catch (_) {}
      throw new Error(msg);
    }
    return res;
  } catch (err) {
    console.error('API Error:', url, err);
    throw err;
  }
}

async function apiJson(url, options) {
  const res = await apiFetch(url, options);
  return res.json();
}

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
   FILE UPLOAD & DRAG-DROP
   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
(function initUpload() {
  const area = document.getElementById('upload-area');
  const input = document.getElementById('file-input');

  area.addEventListener('dragover', e => {
    e.preventDefault(); e.stopPropagation();
    area.classList.add('dragover');
  });
  area.addEventListener('dragleave', e => {
    e.preventDefault(); e.stopPropagation();
    area.classList.remove('dragover');
  });
  area.addEventListener('drop', e => {
    e.preventDefault(); e.stopPropagation();
    area.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      input.files = e.dataTransfer.files;
      uploadFile(e.dataTransfer.files[0]);
    }
  });
  input.addEventListener('change', () => {
    if (input.files.length) uploadFile(input.files[0]);
  });
})();

async function uploadFile(file) {
  const area = document.getElementById('upload-area');
  area.innerHTML = `
    <div class="spinner" style="margin:0 auto 12px"></div>
    <div class="upload-area-title">Uploading ${escHtml(file.name)}...</div>
    <div class="upload-area-sub">Please wait</div>
  `;

  // Step 1: Upload the file
  let data;
  try {
    const fd = new FormData();
    fd.append('file', file);
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    if (!res.ok) {
      let msg = 'Upload failed';
      try { const e = await res.json(); msg = e.detail || e.error || e.message || msg; } catch (_) {}
      throw new Error(msg);
    }
    data = await res.json();
  } catch (err) {
    console.error('Upload error:', err);
    area.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:36px;height:36px;margin-bottom:10px">
        <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
      </svg>
      <div class="upload-area-title" style="color:#ef4444">Upload failed</div>
      <div class="upload-area-sub">${escHtml(err.message)}</div>
    `;
    area.onclick = () => document.getElementById('file-input').click();
    return;
  }

  // Step 2: Process the response (errors here should NOT show "Upload failed")
  try {
    APP.columns = data.columns || [];
    APP.numericColumns = data.numeric_columns || [];

    area.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:space-between;width:100%">
        <div style="display:flex;align-items:center;gap:12px">
          <svg viewBox="0 0 24 24" fill="none" stroke="#6366f1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:28px;height:28px;flex-shrink:0">
            <path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>
          </svg>
          <div>
            <div style="font-size:14px;font-weight:600;color:var(--accent)">${escHtml(file.name)}</div>
            <div style="font-size:12px;color:var(--text-muted);margin-top:2px">${data.rows || 0} rows, ${(data.columns || []).length} columns</div>
          </div>
        </div>
        <button onclick="event.stopPropagation();document.getElementById('file-input').click()" class="btn btn-secondary" style="padding:6px 14px;font-size:12px;white-space:nowrap">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:4px"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
          Replace
        </button>
      </div>
    `;
    area.onclick = null;

    // Reset Build section (clear previous training results)
    const resultsEl = document.getElementById('build-results');
    if (resultsEl) resultsEl.style.display = 'none';
    const evalPlots = document.getElementById('eval-plots');
    if (evalPlots) evalPlots.innerHTML = '';
    const lbTable = document.getElementById('leaderboard-table');
    if (lbTable) lbTable.innerHTML = '';
    const metricChart = document.getElementById('metric-chart');
    if (metricChart) metricChart.innerHTML = '';
    // Reset Explore chart
    const exploreChart = document.getElementById('explore-chart');
    if (exploreChart) exploreChart.innerHTML = '';
    // Reset Export
    const exportModel = document.getElementById('export-model');
    if (exportModel) exportModel.textContent = '--';
    const exportTask = document.getElementById('export-task');
    if (exportTask) exportTask.textContent = '--';

    renderMetrics(data);
    populateTargetSelect(data.columns || []);

    if (data.target) {
      document.getElementById('target-select').value = data.target;
      APP.target = data.target;
      APP.task = data.task || 'classification';
      document.getElementById('task-badge-wrap').innerHTML =
        '<span class="task-badge">' + escHtml(APP.task) + '</span>';
      if (APP.task === 'regression') { selectTaskByValue('regression'); }
      else { selectTaskByValue('classification'); }
    }

    if (data.preview) renderTable('preview-table', data.preview.rows, data.preview.columns);
    if (data.types) renderTable('types-table', data.types.rows, data.types.columns);
    if (data.stats) renderTable('stats-table', data.stats.rows, data.stats.columns);

    document.getElementById('data-metrics').style.display = '';
    document.getElementById('target-section').style.display = '';
    document.getElementById('data-tabs-section').style.display = '';

    // Render auto-insights
    if (data.insights && data.insights.length) {
      renderInsights(data.insights);
      document.getElementById('data-insights').style.display = '';
    }

    populateExploreDropdowns();
  } catch (err2) {
    console.error('Post-upload processing error:', err2);
  }
}

function renderMetrics(data) {
  const grid = document.getElementById('data-metrics');
  const items = [
    { label: 'Rows', value: (data.rows || 0).toLocaleString() },
    { label: 'Columns', value: data.columns ? data.columns.length : 0 },
    { label: 'Missing', value: (data.missing || 0).toLocaleString() },
    { label: 'Duplicates', value: (data.duplicates || 0).toLocaleString() }
  ];
  grid.innerHTML = items.map(m => `
    <div class="card">
      <div class="card-label">${m.label}</div>
      <div class="card-value">${m.value}</div>
    </div>
  `).join('');
}

function populateTargetSelect(cols) {
  const sel = document.getElementById('target-select');
  sel.innerHTML = '<option value="">Select target...</option>';
  cols.forEach(c => {
    sel.innerHTML += '<option value="' + escHtml(c) + '">' + escHtml(c) + '</option>';
  });
}

async function setTarget(col) {
  if (!col) { document.getElementById('task-badge-wrap').innerHTML = ''; return; }
  try {
    const fd = new FormData();
    fd.append('target', col);
    const data = await apiJson('/api/target', { method: 'POST', body: fd });
    APP.target = col;
    APP.task = data.task || 'Classification';
    document.getElementById('task-badge-wrap').innerHTML =
      '<span class="task-badge">' + escHtml(APP.task) + '</span>';

    // Sync build section radio
    if (APP.task.toLowerCase() === 'regression') {
      selectTaskByValue('regression');
    } else {
      selectTaskByValue('classification');
    }
  } catch (err) {
    document.getElementById('task-badge-wrap').innerHTML =
      '<span class="task-badge" style="color:#ef4444;border-color:rgba(239,68,68,0.2);background:rgba(239,68,68,0.1)">Error</span>';
  }
}

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
   DATA TABS
   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
function switchDataTab(tab) {
  document.querySelectorAll('#data-tab-bar .tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('#data-tabs-section .tab-panel').forEach(p => p.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('data-tab-' + tab).classList.add('active');
}

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
   EXPLORE SECTION
   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

// Plot type configuration: which features each plot needs
const PLOT_CONF = {
  histogram:       { feat1: true,  feat2: false, endpoint: '/api/explore/distribution' },
  kde:             { feat1: true,  feat2: false, endpoint: '/api/explore/kde' },
  boxplot:         { feat1: true,  feat2: false, endpoint: '/api/explore/boxplot' },
  violin:          { feat1: true,  feat2: false, endpoint: '/api/explore/violin' },
  missing_bar:     { feat1: false, feat2: false, endpoint: '/api/explore/missing' },
  missing_heatmap: { feat1: false, feat2: false, endpoint: '/api/explore/missing_heatmap' },
  correlation:     { feat1: false, feat2: false, endpoint: '/api/explore/correlation' },
  pairplot:        { feat1: false, feat2: false, endpoint: '/api/explore/pairplot' },
  scatter:         { feat1: true,  feat2: true,  endpoint: '/api/explore/scatter_xy' },
  jointplot:       { feat1: true,  feat2: true,  endpoint: '/api/explore/jointplot' },
  countplot:       { feat1: true,  feat2: false, endpoint: '/api/explore/countplot' },
  pie:             { feat1: false, feat2: false, endpoint: '/api/explore/pie' },
  class_dist:      { feat1: false, feat2: false, endpoint: '/api/explore/target' },
  target_hist:     { feat1: false, feat2: false, endpoint: '/api/explore/target' },
  mean_target:     { feat1: true,  feat2: false, endpoint: '/api/explore/mean_target' },
  scatter_index:   { feat1: true,  feat2: false, endpoint: '/api/explore/scatter_index' },
  grouped_box:     { feat1: true,  feat2: false, endpoint: '/api/explore/grouped_box' },
  facetgrid:       { feat1: true,  feat2: false, endpoint: '/api/explore/facetgrid' }
};

function populateExploreDropdowns() {
  const cols = APP.columns;
  ['plot-feat1', 'plot-feat2'].forEach(id => {
    const sel = document.getElementById(id);
    if (!sel) return;
    sel.innerHTML = '<option value="">Select...</option>';
    cols.forEach(c => {
      sel.innerHTML += '<option value="' + escHtml(c) + '">' + escHtml(c) + '</option>';
    });
  });
}

function switchExploreMain(tab) {
  document.querySelectorAll('#explore-main-tabs .tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('#sec-explore .tab-panel').forEach(p => p.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('explore-panel-' + tab).classList.add('active');
  if (tab === 'quality') loadQuality();
}

function onPlotTypeChange() {
  const type = document.getElementById('plot-type').value;
  const conf = PLOT_CONF[type] || {};
  document.getElementById('feat1-group').style.display = conf.feat1 ? '' : 'none';
  document.getElementById('feat2-group').style.display = conf.feat2 ? '' : 'none';

  // Reset chart visibility state
  const chart = document.getElementById('explore-chart');
  const msg = document.getElementById('explore-msg');
  chart.style.display = '';

  // Auto-generate if no feature selection needed
  if (!conf.feat1 && !conf.feat2) {
    generatePlot();
  } else {
    // Check if a feature is already selected â€” auto-generate
    const f1 = document.getElementById('plot-feat1').value;
    const f2 = document.getElementById('plot-feat2').value;
    if (conf.feat1 && f1 && (!conf.feat2 || f2)) {
      generatePlot();
    } else {
      chart.innerHTML = '';
      msg.style.display = '';
      msg.textContent = 'Select a feature to generate the plot';
    }
  }
}

async function generatePlot() {
  const type = document.getElementById('plot-type').value;
  const conf = PLOT_CONF[type];
  if (!conf) return;

  const f1 = document.getElementById('plot-feat1').value;
  const f2 = document.getElementById('plot-feat2').value;

  // Validate required features
  if (conf.feat1 && !f1) return;
  if (conf.feat2 && !f2) return;

  const chart = document.getElementById('explore-chart');
  const msg = document.getElementById('explore-msg');

  // Always reset visibility before loading
  chart.style.display = '';
  msg.style.display = 'none';
  chart.innerHTML = '<div style="display:flex;justify-content:center;padding:60px 0"><div class="spinner"></div></div>';

  try {
    const fd = new FormData();
    if (conf.feat1 && f1) fd.append('feature', f1);
    if (conf.feat2 && f2) fd.append('feature2', f2);

    const data = await apiJson(conf.endpoint, { method: 'POST', body: fd });

    if (data.figure) {
      chart.style.display = '';
      msg.style.display = 'none';
      renderPlotly('explore-chart', data.figure);
    } else {
      chart.innerHTML = '';
      chart.style.display = 'none';
      msg.style.display = '';
      msg.textContent = data.message || 'No data available for this plot';
    }
  } catch (err) {
    chart.style.display = '';
    chart.innerHTML = '<p style="color:#ef4444;padding:20px;text-align:center">' + escHtml(err.message) + '</p>';
    msg.style.display = 'none';
  }
}

async function loadQuality() {
  try {
    const data = await apiJson('/api/explore/quality', { method: 'POST' });
    if (data.rows && data.rows.length) {
      const cols = Object.keys(data.rows[0]);
      renderTable('quality-table', data.rows, cols);
    } else {
      document.getElementById('quality-table').innerHTML = '<p style="color:var(--text-muted);padding:20px;text-align:center">No quality data available</p>';
    }
  } catch (err) {
    document.getElementById('quality-table').innerHTML = '<p style="color:#ef4444;padding:20px">' + escHtml(err.message) + '</p>';
  }
}

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
   BUILD SECTION
   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
function selectTask(el, value) {
  document.querySelectorAll('#task-radio-group .radio-item').forEach(r => r.classList.remove('active'));
  el.classList.add('active');
  el.querySelector('input').checked = true;
  APP.task = value.charAt(0).toUpperCase() + value.slice(1);
}

function selectTaskByValue(value) {
  const items = document.querySelectorAll('#task-radio-group .radio-item');
  items.forEach(item => {
    const input = item.querySelector('input');
    if (input.value === value) {
      item.classList.add('active');
      input.checked = true;
    } else {
      item.classList.remove('active');
    }
  });
}

function toggleCollapsible(header) {
  header.classList.toggle('open');
  header.nextElementSibling.classList.toggle('open');
}

async function trainModels() {
  const btn = document.getElementById('train-btn');
  const loading = document.getElementById('train-loading');
  const statusEl = document.getElementById('train-status');

  btn.disabled = true;
  loading.classList.remove('hidden');
  statusEl.textContent = 'Initializing pipeline...';
  document.getElementById('build-results').style.display = 'none';

  const fd = new FormData();
  const task = document.querySelector('input[name="task"]:checked').value;
  fd.append('task', task);
  fd.append('test_size', document.getElementById('test-size').value);
  fd.append('cv_folds', document.getElementById('cv-folds').value);

  // Preprocessing options
  document.querySelectorAll('input[name="prep"]').forEach(cb => {
    fd.append(cb.value, cb.checked ? 'true' : 'false');
  });

  // Progress simulation
  const phases = [
    'Preprocessing data...',
    'Setting up experiment...',
    'Comparing models...',
    'Evaluating best model...',
    'Generating plots...'
  ];
  let pi = 0;
  const progressTimer = setInterval(() => {
    if (pi < phases.length) { statusEl.textContent = phases[pi++]; }
  }, 4000);

  try {
    const data = await apiJson('/api/train', { method: 'POST', body: fd });

    clearInterval(progressTimer);
    loading.classList.add('hidden');
    btn.disabled = false;

    // Store state
    APP.bestModel = data.model_name || '--';
    APP.leaderboard = data.leaderboard || null;
    APP.metrics = data.metrics || [];
    APP.nPredictions = data.n_predictions || 0;

    // Show results
    document.getElementById('build-results').style.display = '';
    document.getElementById('best-model-name').textContent = APP.bestModel;

    // Training Summary Banner
    if (data.summary) renderTrainingSummary(data.summary);

    // Leaderboard
    if (data.leaderboard) {
      renderTable('leaderboard-table', data.leaderboard.rows, data.leaderboard.columns);
    }

    // Metric dropdown
    const metricSel = document.getElementById('metric-select');
    metricSel.innerHTML = '';
    (data.metrics || []).forEach(m => {
      metricSel.innerHTML += '<option value="' + escHtml(m) + '">' + escHtml(m) + '</option>';
    });
    renderMetricChart();

    // Eval plots
    renderEvalPlots(data.plots || []);

    // Update export
    document.getElementById('export-model').textContent = APP.bestModel;
    document.getElementById('export-task').textContent = APP.task || task;
    document.getElementById('export-preds').textContent = APP.nPredictions.toLocaleString();

  } catch (err) {
    clearInterval(progressTimer);
    loading.classList.add('hidden');
    btn.disabled = false;
    statusEl.textContent = 'Training failed: ' + err.message;
    setTimeout(() => loading.classList.add('hidden'), 2000);
    alert('Training failed: ' + err.message);
  }
}

function renderMetricChart() {
  if (!APP.leaderboard) return;
  const metric = document.getElementById('metric-select').value;
  if (!metric) return;

  const lb = APP.leaderboard;
  const modelCol = lb.columns.find(c => c.toLowerCase() === 'model') || lb.columns[0];
  const metricIdx = lb.columns.indexOf(metric);
  if (metricIdx === -1) return;

  const models = lb.rows.map(r => r[modelCol] || '');
  const values = lb.rows.map(r => parseFloat(r[metric]) || 0);

  const cs = getComputedStyle(document.documentElement);
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  // Color palette: top 3 get strong indigo, rest get medium indigo (NEVER faded/transparent)
  const barPalette = ['#4f46e5', '#6366f1', '#818cf8', '#a5b4fc', '#93a0f5', '#7c8cf0', '#6b7de8', '#8b96f2', '#7988ed', '#6a7ae6'];
  const colors = values.map((_, i) => barPalette[i % barPalette.length]);
  const barOutline = isLight ? '#4338ca' : 'rgba(255,255,255,0.15)';
  const barOutlineWidth = isLight ? 1.5 : 1;
  const valTextColor = cs.getPropertyValue('--chart-text').trim() || '#a1a1aa';

  const figure = {
    data: [{
      type: 'bar',
      x: values,
      y: models,
      orientation: 'h',
      marker: { color: colors, line: { width: barOutlineWidth, color: barOutline } },
      text: values.map(v => v.toFixed(4)),
      textposition: 'outside',
      textfont: { family: 'Inter', size: 11, color: valTextColor }
    }],
    layout: {
      yaxis: { autorange: 'reversed', tickfont: { size: 11 } },
      xaxis: { title: metric },
      margin: { l: 160, r: 60, t: 20, b: 40 },
      height: Math.max(300, models.length * 36 + 80)
    }
  };
  renderPlotly('metric-chart', figure);
}

function renderEvalPlots(plots) {
  const container = document.getElementById('eval-plots');
  if (!plots || !plots.length) {
    container.innerHTML = '<p style="color:var(--text-muted);font-size:13px">No evaluation plots available</p>';
    return;
  }
  container.innerHTML = plots.map((p, i) => `
    <div class="plot-card">
      <div class="plot-card-title">${escHtml(p.label || 'Plot')}</div>
      <div id="eval-plot-${i}" style="width:100%;min-height:350px"></div>
    </div>
  `).join('');
  // Render each Plotly figure
  plots.forEach((p, i) => {
    if (p.figure && p.figure.data) {
      renderPlotly('eval-plot-' + i, p.figure);
    }
  });
}

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
   TUNING
   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
async function tuneModel() {
  const btn = document.getElementById('tune-btn');
  const loading = document.getElementById('tune-loading');

  btn.disabled = true;
  loading.classList.remove('hidden');
  document.getElementById('tune-results').style.display = 'none';

  const fd = new FormData();
  fd.append('method', document.getElementById('tune-method').value);
  fd.append('iterations', document.getElementById('tune-iter').value);

  try {
    const data = await apiJson('/api/tune', { method: 'POST', body: fd });

    loading.classList.add('hidden');
    btn.disabled = false;

    document.getElementById('tune-results').style.display = '';

    const improvedEl = document.getElementById('tune-improved');
    if (data.improved) {
      improvedEl.innerHTML = '<span style="color:var(--accent)">Model improved after tuning: ' + escHtml(data.model_name || '') + '</span>';
      APP.bestModel = data.model_name || APP.bestModel;
      document.getElementById('best-model-name').textContent = APP.bestModel;
      document.getElementById('export-model').textContent = APP.bestModel;
    } else {
      improvedEl.innerHTML = '<span style="color:#f59e0b">No improvement found. Keeping original model.</span>';
    }

    if (data.tune_results) {
      renderTable('tune-table', data.tune_results.rows, data.tune_results.columns);
    }

    // Update eval plots if provided
    if (data.plots && data.plots.length) {
      renderEvalPlots(data.plots);
    }

  } catch (err) {
    loading.classList.add('hidden');
    btn.disabled = false;
    alert('Tuning failed: ' + err.message);
  }
}

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
   EXPORT
   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
function downloadExport(type) {
  const url = '/api/export/' + type;
  const a = document.createElement('a');
  a.href = url;
  a.download = '';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
   AUTO INSIGHTS
   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
const INSIGHT_ICONS = {
  success: '<svg class="insight-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
  warning: '<svg class="insight-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
  info: '<svg class="insight-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
};

function renderInsights(insights) {
  const el = document.getElementById('data-insights');
  if (!el || !insights || !insights.length) return;
  el.innerHTML = '<div class="insights-grid">' + insights.map((ins, i) => `
    <div class="insight-card ${ins.type || 'info'}" style="animation-delay:${i * 0.06}s">
      ${INSIGHT_ICONS[ins.type] || INSIGHT_ICONS.info}
      <div>
        <div class="insight-title">${escHtml(ins.title || '')}</div>
        <div class="insight-detail">${escHtml(ins.detail || '')}</div>
      </div>
    </div>
  `).join('') + '</div>';
}

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
   TRAINING SUMMARY
   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
function renderTrainingSummary(summary) {
  if (!summary) return;
  const container = document.getElementById('training-summary');
  if (!container) return;

  const featurePills = (summary.top_features || []).map(f =>
    `<span class="feature-pill">${escHtml(f)}</span>`
  ).join('');

  container.innerHTML = `
    <div class="summary-banner">
      <div class="card-label">Training Summary</div>
      <div class="summary-stats">
        <div class="summary-stat">
          <div class="summary-stat-label">${escHtml(summary.key_metric || '')}</div>
          <div class="summary-stat-value" style="color:var(--accent)">${summary.key_metric_value || '--'}</div>
        </div>
        <div class="summary-stat">
          <div class="summary-stat-label">Models Compared</div>
          <div class="summary-stat-value">${summary.n_models || '--'}</div>
        </div>
        <div class="summary-stat">
          <div class="summary-stat-label">Best Model</div>
          <div class="summary-stat-value">${escHtml(summary.model || '--')}</div>
        </div>
      </div>
      ${featurePills ? '<div style="margin-top:14px"><div class="summary-stat-label" style="margin-bottom:6px">Top Predictive Features</div>' + featurePills + '</div>' : ''}
      <div class="summary-text">${escHtml(summary.text || '')}</div>
    </div>
  `;
  container.style.display = '';
}

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
   THEME TOGGLE
   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
function toggleTheme() {
  const html = document.documentElement;
  const isLight = html.getAttribute('data-theme') === 'light';
  const newTheme = isLight ? 'dark' : 'light';
  html.setAttribute('data-theme', newTheme);
  localStorage.setItem('automl-theme', newTheme);
  updateThemeIcon(newTheme);
  reThemePlotly(newTheme);
}

function updateThemeIcon(theme) {
  const moon = document.getElementById('icon-moon');
  const sun = document.getElementById('icon-sun');
  if (theme === 'light') {
    moon.style.display = 'none';
    sun.style.display = '';
  } else {
    moon.style.display = '';
    sun.style.display = 'none';
  }
}

function reThemePlotly(theme) {
  // Small delay to let CSS variables update
  requestAnimationFrame(() => {
    const style = getComputedStyle(document.documentElement);
    const textColor = style.getPropertyValue('--chart-text').trim() || '#71717a';
    const gridColor = style.getPropertyValue('--chart-grid').trim() || 'rgba(30,27,75,0.15)';
    document.querySelectorAll('.js-plotly-plot').forEach(el => {
      try {
        Plotly.relayout(el, {
          'font.color': textColor,
          'xaxis.gridcolor': gridColor,
          'yaxis.gridcolor': gridColor,
        });
      } catch (_) {}
    });
  });
}

// Restore saved theme
(function() {
  const saved = localStorage.getItem('automl-theme');
  if (saved === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
    updateThemeIcon('light');
  }
})();

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
   INIT
   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
showSection('data');
