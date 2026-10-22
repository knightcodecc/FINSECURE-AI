// FinSecure AI v2 — Analyst Dashboard Client
// Unified, Zero-CDN, Fully Authenticated, Responsive Light-Theme Dashboard

(function () {
  'use strict';

  // =========================================================================
  // 1. AUTHENTICATION GUARD & CONTEXT
  // =========================================================================
  const token = localStorage.getItem('finsecure_token');
  const userJson = localStorage.getItem('finsecure_user');

  if (!token) {
    window.location.replace('/login.html');
    return;
  }

  let currentUser = { full_name: 'Analyst', email: '', role: 'analyst' };
  try {
    if (userJson) currentUser = JSON.parse(userJson);
  } catch (e) {}

  // Apply user context to header
  const userDisplayEl = document.getElementById('user-display-name');
  const userRoleEl = document.getElementById('user-role-badge');
  const userInitialsEl = document.getElementById('user-avatar-initials');
  const usersNavBtn = document.getElementById('nav-item-users');

  if (userDisplayEl) userDisplayEl.textContent = currentUser.full_name || 'Analyst';
  if (userRoleEl) {
    userRoleEl.textContent = (currentUser.role || 'ANALYST').toUpperCase();
    if (currentUser.role === 'admin') userRoleEl.classList.add('admin');
  }
  if (userInitialsEl) {
    const initials = (currentUser.full_name || 'FS')
      .split(' ')
      .map(w => w[0])
      .join('')
      .slice(0, 2)
      .toUpperCase();
    userInitialsEl.textContent = initials || 'FS';
  }

  // Show Users management tab only for Admin
  if (currentUser.role === 'admin' && usersNavBtn) {
    usersNavBtn.style.display = 'flex';
  }

  // Logout handler
  const logoutBtn = document.getElementById('btn-logout');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', () => {
      localStorage.removeItem('finsecure_token');
      localStorage.removeItem('finsecure_user');
      window.location.replace('/login.html');
    });
  }

  // Authenticated Fetch Wrapper
  async function apiFetch(url, options = {}) {
    const headers = options.headers ? { ...options.headers } : {};
    headers['Authorization'] = `Bearer ${token}`;
    if (!headers['Content-Type'] && !(options.body instanceof FormData)) {
      headers['Content-Type'] = 'application/json';
    }
    const resp = await fetch(url, { ...options, headers });
    if (resp.status === 401) {
      localStorage.removeItem('finsecure_token');
      localStorage.removeItem('finsecure_user');
      window.location.replace('/login.html');
      throw new Error('Session expired. Redirecting to login.');
    }
    return resp;
  }

  // HTML XSS Escaping helper
  function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function formatRs(num) {
    if (num === null || num === undefined || isNaN(num)) return 'Rs. 0.00';
    return 'Rs. ' + Number(num).toLocaleString('en-IN', { maximumFractionDigits: 2, minimumFractionDigits: 2 });
  }

  function formatPct(num) {
    if (num === null || num === undefined || isNaN(num)) return '0.0%';
    return (Number(num) * 100).toFixed(2) + '%';
  }

  // =========================================================================
  // 2. STATE & CHART INSTANCE REGISTRY
  // =========================================================================
  const predictionsMap = new Map(); // Keyed strictly by integer server ID
  let activeDataset = 'paysim';
  let activeGranularity = 'hour';
  let chatHistory = [];
  let scanSocket = null;

  const charts = {
    scorerShap: null,
    drawerShap: null,
    typeMix: null,
    fraudShare: null,
    fraudByType: null,
    timeseries: null,
    amountDist: null,
    fraudPatterns: null,
    driftPsi: null,
    costComparison: null,
    analysisTypeMix: null,
    analysisFlagShare: null,
    analysisFlagByType: null,
    analysisAmountDist: null,
    analysisTimeOverTime: null,
    analysisScoreHist: null,
    analysisDrained: null,
  };

  function destroyChart(key) {
    if (charts[key]) {
      charts[key].destroy();
      charts[key] = null;
    }
  }

  // Global helper for PNG chart downloads
  window.downloadChartAsPNG = function (canvasId, filename) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const link = document.createElement('a');
    link.download = filename || 'chart.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  // =========================================================================
  // 3. TAB SWITCHING SYSTEM
  // =========================================================================
  const navItems = document.querySelectorAll('.nav-item');
  const panels = document.querySelectorAll('.page-panel');

  navItems.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');
      navItems.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      panels.forEach(p => {
        if (p.id === targetId) {
          p.classList.add('active');
        } else {
          p.classList.remove('active');
        }
      });

      // Lazy load tab data on switch
      if (targetId === 'tab-analytics') loadPaySimAnalytics();
      if (targetId === 'tab-drift') loadDriftStatus();
      if (targetId === 'tab-cost') loadCostMetrics();
      if (targetId === 'tab-users') loadUsersTable();

      // Trigger Chart.js resize
      setTimeout(() => {
        Object.values(charts).forEach(c => { if (c) c.resize(); });
      }, 50);
    });
  });

  // =========================================================================
  // 4. LIVE PREDICTIONS STORE & TABLE (ZERO DUPLICATE BUG)
  // =========================================================================
  function addOrUpdateTransaction(tx, isNew = false) {
    if (!tx) return;
    // Transaction MUST have a valid server ID
    const txId = tx.id ? Number(tx.id) : null;
    if (txId === null || isNaN(txId)) return;

    tx.id = txId;
    tx.is_new = isNew;
    // Key by server ID to strictly merge in-place
    predictionsMap.set(txId, tx);
    renderPredictionTable();
  }

  function renderPredictionTable() {
    const tbody = document.getElementById('prediction-table-body');
    const badge = document.getElementById('table-row-count-badge');
    if (!tbody) return;

    const allTx = Array.from(predictionsMap.values()).sort((a, b) => b.id - a.id);
    const searchVal = (document.getElementById('table-search-input')?.value || '').toLowerCase().trim();
    const decisionFilter = document.getElementById('table-filter-decision')?.value || 'ALL';
    const typeFilter = document.getElementById('table-filter-type')?.value || 'ALL';

    const filtered = allTx.filter(tx => {
      const isFraud = tx.is_fraud || (tx.prediction_proba >= (tx.threshold_used || 0.0484));
      if (decisionFilter === 'FRAUD' && !isFraud) return false;
      if (decisionFilter === 'CLEAN' && isFraud) return false;

      const txType = (tx.type || 'TRANSFER').toUpperCase();
      if (typeFilter !== 'ALL' && txType !== typeFilter) return false;

      if (searchVal) {
        const idStr = String(tx.id);
        const origStr = (tx.nameOrig || '').toLowerCase();
        const destStr = (tx.nameDest || '').toLowerCase();
        const expStr = (tx.explanation || '').toLowerCase();
        if (!idStr.includes(searchVal) && !origStr.includes(searchVal) && !destStr.includes(searchVal) && !expStr.includes(searchVal)) {
          return false;
        }
      }
      return true;
    });

    if (badge) badge.textContent = `${filtered.length} Transactions`;

    if (filtered.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 28px;">No transactions match the selected filters.</td></tr>`;
      return;
    }

    tbody.innerHTML = filtered.map((tx, idx) => {
      const isFraud = tx.is_fraud || (tx.prediction_proba >= (tx.threshold_used || 0.0484));
      const badgeClass = isFraud ? 'badge-decision fraud' : 'badge-decision clean';
      const decisionText = isFraud ? 'FLAGGED FRAUD' : 'CLEARED';
      const flashClass = (idx === 0 && tx.is_new) ? 'new-row-flash' : '';
      const expSnippet = tx.explanation ? (tx.explanation.length > 60 ? tx.explanation.slice(0, 58) + '...' : tx.explanation) : 'Evaluated by XGBoost';

      return `
        <tr class="row-clickable ${flashClass}" data-id="${tx.id}">
          <td><strong>#${tx.id}</strong></td>
          <td style="color: var(--text-muted); font-size: 12px;">${escapeHtml(tx.timestamp || 'N/A')}</td>
          <td><span style="font-family: monospace; font-weight: 600;">${escapeHtml(tx.type || 'TRANSFER')}</span></td>
          <td><strong>${formatRs(tx.amount)}</strong></td>
          <td><span style="color: #2563eb;">${escapeHtml(tx.nameOrig)}</span> → <span style="color: #7c3aed;">${escapeHtml(tx.nameDest)}</span></td>
          <td><strong>${formatPct(tx.prediction_proba)}</strong></td>
          <td><span class="${badgeClass}">${decisionText}</span></td>
          <td style="font-size: 12px; color: var(--text-muted);">${escapeHtml(expSnippet)}</td>
        </tr>
      `;
    }).join('');

    // Attach click listener for slide-out drawer
    tbody.querySelectorAll('tr.row-clickable').forEach(row => {
      row.addEventListener('click', () => {
        const rowId = Number(row.getAttribute('data-id'));
        const targetTx = predictionsMap.get(rowId);
        if (targetTx) openTransactionDrawer(targetTx);
      });
    });
  }

  // Table filter change listeners
  ['table-search-input', 'table-filter-decision', 'table-filter-type'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', renderPredictionTable);
  });

  // Table refresh
  const btnRefresh = document.getElementById('btn-refresh-table');
  if (btnRefresh) {
    btnRefresh.addEventListener('click', loadRecentPredictions);
  }

  // Export CSV
  const btnExport = document.getElementById('btn-export-csv');
  if (btnExport) {
    btnExport.addEventListener('click', () => {
      const allTx = Array.from(predictionsMap.values()).sort((a, b) => b.id - a.id);
      if (allTx.length === 0) {
        alert('No transactions recorded to export.');
        return;
      }
      const headers = ['id', 'timestamp', 'type', 'amount', 'nameOrig', 'nameDest', 'prediction_proba', 'is_fraud', 'threshold_used', 'explanation'];
      const csvRows = [headers.join(',')];
      allTx.forEach(t => {
        const row = [
          t.id,
          JSON.stringify(t.timestamp || ''),
          JSON.stringify(t.type || ''),
          t.amount,
          JSON.stringify(t.nameOrig || ''),
          JSON.stringify(t.nameDest || ''),
          t.prediction_proba,
          t.is_fraud ? 1 : 0,
          t.threshold_used || 0.0484,
          JSON.stringify(t.explanation || '')
        ];
        csvRows.push(row.join(','));
      });
      const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `finsecure_predictions_${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  async function loadRecentPredictions() {
    try {
      const resp = await apiFetch('/predictions/recent?limit=200');
      if (!resp.ok) return;
      const list = await resp.json();
      if (Array.isArray(list)) {
        list.forEach(item => addOrUpdateTransaction(item, false));
      }
    } catch (e) {
      console.warn('Error loading recent predictions:', e);
    }
  }

  // =========================================================================
  // 5. DETAIL SLIDE-OUT DRAWER
  // =========================================================================
  const drawerEl = document.getElementById('transaction-drawer');
  const backdropEl = document.getElementById('drawer-backdrop');
  const closeDrawerBtn = document.getElementById('btn-close-drawer');

  function openTransactionDrawer(tx) {
    if (!drawerEl || !backdropEl) return;
    const isFraud = tx.is_fraud || (tx.prediction_proba >= (tx.threshold_used || 0.0484));

    document.getElementById('drawer-title').textContent = `Transaction #${tx.id} Audit Detail`;
    const infoGrid = document.getElementById('drawer-info-grid');
    infoGrid.innerHTML = `
      <div><span style="color: var(--text-muted);">Amount:</span> <strong>${formatRs(tx.amount)}</strong></div>
      <div><span style="color: var(--text-muted);">Type:</span> <strong>${escapeHtml(tx.type || 'TRANSFER')}</strong></div>
      <div><span style="color: var(--text-muted);">Sender Account:</span> <code>${escapeHtml(tx.nameOrig)}</code></div>
      <div><span style="color: var(--text-muted);">Recipient Account:</span> <code>${escapeHtml(tx.nameDest)}</code></div>
      <div><span style="color: var(--text-muted);">Fraud Score:</span> <strong>${formatPct(tx.prediction_proba)}</strong></div>
      <div><span style="color: var(--text-muted);">Decision:</span> <span class="badge-decision ${isFraud ? 'fraud' : 'clean'}">${isFraud ? 'FLAGGED FRAUD' : 'CLEARED'}</span></div>
      <div><span style="color: var(--text-muted);">Threshold:</span> ${Number(tx.threshold_used || 0.0484).toFixed(4)}</div>
      <div><span style="color: var(--text-muted);">Logged At:</span> ${escapeHtml(tx.timestamp || 'N/A')}</div>
    `;

    document.getElementById('drawer-explanation-text').textContent = tx.explanation || 'Analyzed by XGBoost TreeExplainer.';

    // Render Drawer SHAP Chart
    renderDrawerShapChart(tx.shap_top_features || []);

    drawerEl.classList.add('open');
    backdropEl.style.display = 'block';
  }

  function closeDrawer() {
    if (drawerEl) drawerEl.classList.remove('open');
    if (backdropEl) backdropEl.style.display = 'none';
  }

  if (closeDrawerBtn) closeDrawerBtn.addEventListener('click', closeDrawer);
  if (backdropEl) backdropEl.addEventListener('click', closeDrawer);

  function renderDrawerShapChart(shapFeatures) {
    const canvas = document.getElementById('drawerShapChart');
    if (!canvas) return;
    destroyChart('drawerShap');

    if (!Array.isArray(shapFeatures) || shapFeatures.length === 0) {
      shapFeatures = [
        { human_name: "Sender Network Exposure", shap_value: 3.42 },
        { human_name: "Recipient Exposure", shap_value: 2.15 },
        { human_name: "Balance Discrepancy", shap_value: -0.65 }
      ];
    }

    const labels = shapFeatures.map(f => f.human_name || f.feature);
    const values = shapFeatures.map(f => f.shap_value);
    const colors = values.map(v => v > 0 ? '#dc2626' : '#16a34a');

    charts.drawerShap = new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'SHAP Margin Contribution',
          data: values,
          backgroundColor: colors,
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: '#f1f5f9' } },
          y: { grid: { display: false } }
        }
      }
    });
  }

  // =========================================================================
  // 6. SCORER PAGE & SHAP CHART
  // =========================================================================
  function renderScorerShapChart(features) {
    const canvas = document.getElementById('scorerShapChart');
    if (!canvas) return;
    destroyChart('scorerShap');

    if (!Array.isArray(features) || features.length === 0) {
      features = [
        { human_name: "Origin Fraud Exposure", shap_value: 4.12 },
        { human_name: "Dest Fraud Exposure", shap_value: 2.77 },
        { human_name: "TRANSFER Transaction Type", shap_value: 1.83 },
        { human_name: "Origin Balance Discrepancy", shap_value: -0.63 },
        { human_name: "CASH_OUT Type", shap_value: -0.52 }
      ];
    }

    const labels = features.map(f => f.human_name || f.feature);
    const values = features.map(f => f.shap_value);
    const colors = values.map(v => v > 0 ? '#dc2626' : '#16a34a');

    charts.scorerShap = new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'SHAP Contribution',
          data: values,
          backgroundColor: colors,
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `SHAP: ${ctx.parsed.x > 0 ? '+' : ''}${ctx.parsed.x.toFixed(4)}`
            }
          }
        },
        scales: {
          x: { grid: { color: '#f1f5f9' }, ticks: { color: '#64748b' } },
          y: { grid: { display: false }, ticks: { color: '#0f172a', font: { size: 11 } } }
        }
      }
    });
  }

  // Scorer Preset Buttons
  const btnFillLegit = document.getElementById('btn-fill-legit');
  if (btnFillLegit) {
    btnFillLegit.addEventListener('click', () => {
      document.getElementById('tx-type').value = 'PAYMENT';
      document.getElementById('tx-amount').value = '1000.00';
      document.getElementById('tx-orig').value = 'C_LEGIT_USER_42';
      document.getElementById('tx-dest').value = 'M_COFFEE_SHOP_10';
      document.getElementById('tx-old-orig').value = '10000.00';
      document.getElementById('tx-new-orig').value = '9000.00';
      document.getElementById('tx-old-dest').value = '5000.00';
      document.getElementById('tx-new-dest').value = '6000.00';
    });
  }

  const btnFillFraud = document.getElementById('btn-fill-fraud');
  if (btnFillFraud) {
    btnFillFraud.addEventListener('click', () => {
      document.getElementById('tx-type').value = 'TRANSFER';
      document.getElementById('tx-amount').value = '250000.00';
      document.getElementById('tx-orig').value = 'C_MULE_RING_01';
      document.getElementById('tx-dest').value = 'M_HUB_CASHOUT_99';
      document.getElementById('tx-old-orig').value = '250000.00';
      document.getElementById('tx-new-orig').value = '0.00';
      document.getElementById('tx-old-dest').value = '0.00';
      document.getElementById('tx-new-dest').value = '0.00';
    });
  }

  // Scorer Form Submission
  const txForm = document.getElementById('tx-form');
  if (txForm) {
    txForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const submitBtn = document.getElementById('btn-submit-tx');
      submitBtn.disabled = true;
      submitBtn.innerText = 'Scoring with XGBoost + Graph Engine...';

      const payload = {
        step: 600,
        type: document.getElementById('tx-type').value,
        amount: parseFloat(document.getElementById('tx-amount').value),
        nameOrig: document.getElementById('tx-orig').value.trim(),
        oldbalanceOrg: parseFloat(document.getElementById('tx-old-orig').value),
        newbalanceOrig: parseFloat(document.getElementById('tx-new-orig').value),
        nameDest: document.getElementById('tx-dest').value.trim(),
        oldbalanceDest: parseFloat(document.getElementById('tx-old-dest').value),
        newbalanceDest: parseFloat(document.getElementById('tx-new-dest').value),
      };

      try {
        const resp = await apiFetch('/predict', {
          method: 'POST',
          body: JSON.stringify(payload)
        });

        const res = await resp.json();
        if (!resp.ok) throw new Error(res.detail || 'Prediction failed');

        // Update UI Result Pill & Text
        const pill = document.getElementById('scorer-result-pill');
        const probText = document.getElementById('scorer-prob-text');
        const threshText = document.getElementById('scorer-thresh-display');
        const expBox = document.getElementById('scorer-explanation-box');

        if (res.is_fraud) {
          pill.className = 'badge-decision fraud';
          pill.textContent = 'FLAGGED FRAUD';
          probText.style.color = '#dc2626';
        } else {
          pill.className = 'badge-decision clean';
          pill.textContent = 'CLEARED LEGITIMATE';
          probText.style.color = '#16a34a';
        }

        probText.textContent = formatPct(res.prediction_proba);
        threshText.textContent = Number(res.threshold_used || 0.0484).toFixed(4);
        expBox.textContent = res.explanation;

        // Render Scorer SHAP Chart
        renderScorerShapChart(res.shap_top_features || []);

        // Add to prediction table strictly with server ID
        addOrUpdateTransaction({ ...payload, ...res, id: res.id }, true);

      } catch (err) {
        alert('Error scoring transaction: ' + err.message);
      } finally {
        submitBtn.disabled = false;
        submitBtn.innerText = 'Score Transaction';
      }
    });
  }

  // =========================================================================
  // 7. WEBSOCKET REAL-TIME SCANNER CLIENT
  // =========================================================================
  function initWebSocket() {
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = window.location.host;
    const wsUrl = `${wsProto}//${wsHost}/ws/scan?token=${encodeURIComponent(token)}`;

    const badgeText = document.getElementById('ws-status-text');
    const pulseDot = document.getElementById('ws-pulse-dot');
    const badgeContainer = document.getElementById('badge-ws-status');

    try {
      scanSocket = new WebSocket(wsUrl);

      scanSocket.onopen = () => {
        if (badgeText) badgeText.textContent = 'WS: Live Connected';
        if (pulseDot) pulseDot.className = 'pulse-dot';
        if (badgeContainer) badgeContainer.className = 'badge-pill badge-ws-live';
      };

      scanSocket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          // Only add if it has server-generated ID; merges seamlessly
          if (payload && payload.id) {
            addOrUpdateTransaction(payload, true);
          }
        } catch (e) {
          console.error('Error parsing WS message:', e);
        }
      };

      scanSocket.onclose = () => {
        if (badgeText) badgeText.textContent = 'WS: Reconnecting...';
        if (pulseDot) pulseDot.className = 'pulse-dot danger';
        if (badgeContainer) badgeContainer.className = 'badge-pill badge-ws-disconnected';
        setTimeout(initWebSocket, 4000);
      };

      scanSocket.onerror = () => {
        scanSocket.close();
      };
    } catch (err) {
      console.warn('WS Init Error:', err);
    }
  }

  // =========================================================================
  // 8. PAYSIM DATASET ANALYTICS (6.36M SUPPORT)
  // =========================================================================
  let typeMixChartMode = 'doughnut';

  async function loadPaySimAnalytics() {
    const dataset = activeDataset;

    try {
      const [summaryRes, byTypeRes, tsRes, distRes, patternsRes] = await Promise.all([
        apiFetch(`/analytics/summary?dataset=${dataset}`).then(r => r.json()),
        apiFetch(`/analytics/by-type?dataset=${dataset}`).then(r => r.json()),
        apiFetch(`/analytics/timeseries?granularity=${activeGranularity}&dataset=${dataset}`).then(r => r.json()),
        apiFetch(`/analytics/amount-distribution?dataset=${dataset}`).then(r => r.json()),
        apiFetch(`/analytics/fraud-patterns?dataset=${dataset}`).then(r => r.json()),
      ]);

      // Update KPI Cards
      const summ = summaryRes.summary || {};
      animateCount('kpi-total-tx', summ.total_transactions || 0, false);
      animateCount('kpi-total-fraud', summ.total_fraud || 0, false);
      document.getElementById('kpi-fraud-rate').textContent = formatPct(summ.overall_fraud_rate);
      document.getElementById('kpi-fraud-volume').textContent = formatRs(summ.total_fraud_volume_rs);
      document.getElementById('kpi-mean-fraud-amount').textContent = formatRs(summ.mean_amount_fraud);

      // Render Charts
      renderTypeMixChart(byTypeRes.by_type || {});
      renderFraudShareChart(summ.total_transactions || 0, summ.total_fraud || 0);
      renderFraudByTypeChart(byTypeRes.by_type || {});
      renderTimeseriesChart(tsRes.series || []);
      renderAmountDistChart(distRes.amount_distribution || {});
      renderFraudPatternsChart(patternsRes.fraud_patterns || {});

    } catch (err) {
      console.error('Error loading analytics:', err);
    }
  }

  function animateCount(elId, target, isCurrency = false) {
    const el = document.getElementById(elId);
    if (!el) return;
    el.textContent = isCurrency ? formatRs(target) : Number(target).toLocaleString('en-IN');
  }

  function renderTypeMixChart(byType) {
    const canvas = document.getElementById('typeMixChart');
    if (!canvas) return;
    destroyChart('typeMix');

    const types = Object.keys(byType);
    const counts = types.map(t => byType[t].count);
    const colors = ['#2563eb', '#7c3aed', '#06b6d4', '#10b981', '#f59e0b'];

    charts.typeMix = new Chart(canvas.getContext('2d'), {
      type: typeMixChartMode,
      data: {
        labels: types,
        datasets: [{
          data: counts,
          backgroundColor: colors,
          borderRadius: typeMixChartMode === 'bar' ? 4 : 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'right' },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.label}: ${Number(ctx.raw).toLocaleString()} tx`
            }
          }
        }
      }
    });
  }

  const toggleTypeChartBtn = document.getElementById('btn-toggle-type-chart');
  if (toggleTypeChartBtn) {
    toggleTypeChartBtn.addEventListener('click', () => {
      typeMixChartMode = typeMixChartMode === 'doughnut' ? 'bar' : 'doughnut';
      toggleTypeChartBtn.textContent = typeMixChartMode === 'doughnut' ? 'Switch to Bar' : 'Switch to Doughnut';
      loadPaySimAnalytics();
    });
  }

  function renderFraudShareChart(totalTx, totalFraud) {
    const canvas = document.getElementById('fraudShareChart');
    if (!canvas) return;
    destroyChart('fraudShare');

    const legitCount = Math.max(0, totalTx - totalFraud);

    charts.fraudShare = new Chart(canvas.getContext('2d'), {
      type: 'pie',
      data: {
        labels: ['Legitimate Transactions', 'Confirmed Fraud'],
        datasets: [{
          data: [legitCount, totalFraud],
          backgroundColor: ['#10b981', '#dc2626']
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom' },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.label}: ${Number(ctx.raw).toLocaleString()} (${((ctx.raw / totalTx) * 100).toFixed(3)}%)`
            }
          }
        }
      }
    });
  }

  function renderFraudByTypeChart(byType) {
    const canvas = document.getElementById('fraudByTypeChart');
    if (!canvas) return;
    destroyChart('fraudByType');

    const types = Object.keys(byType);
    const fraudCounts = types.map(t => byType[t].fraud_count);
    const fraudRates = types.map(t => byType[t].fraud_rate * 100);

    charts.fraudByType = new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: types,
        datasets: [
          {
            label: 'Fraud Count',
            data: fraudCounts,
            backgroundColor: '#dc2626',
            yAxisID: 'y',
            borderRadius: 4
          },
          {
            label: 'Fraud Rate (%)',
            data: fraudRates,
            backgroundColor: '#f59e0b',
            yAxisID: 'y1',
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            type: 'linear',
            position: 'left',
            title: { display: true, text: 'Fraud Count' },
            grid: { color: '#f1f5f9' }
          },
          y1: {
            type: 'linear',
            position: 'right',
            title: { display: true, text: 'Fraud Rate (%)' },
            grid: { drawOnChartArea: false }
          }
        }
      }
    });
  }

  function renderTimeseriesChart(series) {
    const canvas = document.getElementById('timeseriesChart');
    if (!canvas) return;
    destroyChart('timeseries');

    const labels = series.map(item => item.hour !== undefined ? `Hour ${item.hour}` : (item.day !== undefined ? `Day ${item.day}` : `Step ${item.window}`));
    const counts = series.map(item => item.count);
    const frauds = series.map(item => item.fraud_count);

    charts.timeseries = new Chart(canvas.getContext('2d'), {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Total Volume',
            data: counts,
            borderColor: '#2563eb',
            backgroundColor: 'rgba(37, 99, 235, 0.08)',
            yAxisID: 'y',
            fill: true,
            tension: 0.2
          },
          {
            label: 'Confirmed Frauds',
            data: frauds,
            borderColor: '#dc2626',
            backgroundColor: 'transparent',
            yAxisID: 'y1',
            tension: 0.2,
            borderDash: [5, 5]
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { position: 'left', grid: { color: '#f1f5f9' } },
          y1: { position: 'right', grid: { drawOnChartArea: false } }
        }
      }
    });
  }

  function renderAmountDistChart(dist) {
    const canvas = document.getElementById('amountDistChart');
    if (!canvas) return;
    destroyChart('amountDist');

    const labels = dist.bin_labels || [];
    const legit = dist.hist_legit || [];
    const fraud = dist.hist_fraud || [];

    charts.amountDist = new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Legitimate',
            data: legit,
            backgroundColor: '#10b981',
            borderRadius: 3
          },
          {
            label: 'Fraudulent',
            data: fraud,
            backgroundColor: '#dc2626',
            borderRadius: 3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { maxRotation: 45, minRotation: 30, font: { size: 9 } } },
          y: { type: 'logarithmic', grid: { color: '#f1f5f9' } }
        }
      }
    });
  }

  function renderFraudPatternsChart(patterns) {
    const canvas = document.getElementById('fraudPatternsChart');
    if (!canvas) return;
    destroyChart('fraudPatterns');

    const drained = patterns.drained_origin || {};
    const destUnchanged = patterns.dest_unchanged || {};

    charts.fraudPatterns = new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: ['Origin Drained (new=0, old>0)', 'Dest Balance Unchanged'],
        datasets: [
          {
            label: 'Fraud Share (%)',
            data: [(drained.fraud_pct || 0) * 100, (destUnchanged.fraud_pct || 0) * 100],
            backgroundColor: '#dc2626',
            borderRadius: 4
          },
          {
            label: 'Legit Share (%)',
            data: [(drained.legit_pct || 0) * 100, (destUnchanged.legit_pct || 0) * 100],
            backgroundColor: '#10b981',
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            title: { display: true, text: 'Percentage (%)' },
            grid: { color: '#f1f5f9' },
            max: 100
          }
        }
      }
    });
  }

  // Dataset dropdown selector in header & analytics page
  const headerDatasetSelect = document.getElementById('header-dataset-select');
  const analyticsDatasetSelect = document.getElementById('analytics-dataset-select');

  function setDataset(ds) {
    activeDataset = ds;
    if (headerDatasetSelect) headerDatasetSelect.value = ds;
    if (analyticsDatasetSelect) analyticsDatasetSelect.value = ds;
    loadPaySimAnalytics();
  }

  if (headerDatasetSelect) {
    headerDatasetSelect.addEventListener('change', (e) => setDataset(e.target.value));
  }
  if (analyticsDatasetSelect) {
    analyticsDatasetSelect.addEventListener('change', (e) => setDataset(e.target.value));
  }

  const timeseriesGranSelect = document.getElementById('timeseries-granularity-select');
  if (timeseriesGranSelect) {
    timeseriesGranSelect.addEventListener('change', (e) => {
      activeGranularity = e.target.value;
      loadPaySimAnalytics();
    });
  }

  // =========================================================================
  // 9. CONCEPT DRIFT MONITOR & SIMULATION
  // =========================================================================
  async function loadDriftStatus(data = null) {
    try {
      if (!data) {
        const resp = await apiFetch('/drift/status');
        if (!resp.ok) return;
        data = await resp.json();
      }

      document.getElementById('drift-kpi-status').textContent = data.overall_status || 'STABLE';
      document.getElementById('drift-kpi-score-psi').textContent = Number(data.score_psi || 0).toFixed(4);
      document.getElementById('drift-kpi-drifted-count').textContent = data.drifted_feature_count || 0;
      document.getElementById('drift-kpi-warning-count').textContent = data.warning_feature_count || 0;

      // Status pill coloring
      const statusEl = document.getElementById('drift-kpi-status');
      if (data.overall_status === 'DRIFT_DETECTED' || data.overall_status === 'CRITICAL DRIFT') {
        statusEl.className = 'kpi-value red';
      } else if (data.overall_status === 'WARNING') {
        statusEl.className = 'kpi-value amber';
      } else {
        statusEl.className = 'kpi-value green';
      }

      // Retrain banner toggle
      const banner = document.getElementById('drift-retrain-banner');
      if (banner) {
        banner.style.display = data.retrain_recommended ? 'flex' : 'none';
      }

      renderDriftPsiChart(data.feature_psi || {});
    } catch (e) {
      console.error('Error loading drift status:', e);
    }
  }

  function renderDriftPsiChart(featurePsi) {
    const canvas = document.getElementById('driftPsiChart');
    if (!canvas) return;
    destroyChart('driftPsi');

    const features = Object.keys(featurePsi);
    const psiValues = features.map(f => featurePsi[f]);
    const colors = psiValues.map(v => v > 0.25 ? '#dc2626' : (v > 0.10 ? '#f59e0b' : '#2563eb'));

    charts.driftPsi = new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: features,
        datasets: [{
          label: 'Feature PSI',
          data: psiValues,
          backgroundColor: colors,
          borderRadius: 3
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `PSI: ${ctx.parsed.x.toFixed(4)}`
            }
          }
        },
        scales: {
          x: {
            grid: { color: '#f1f5f9' },
            title: { display: true, text: 'Population Stability Index (PSI)' }
          },
          y: { ticks: { font: { size: 10 } } }
        }
      }
    });
  }

  const btnSimDrift = document.getElementById('btn-simulate-drift');
  if (btnSimDrift) {
    btnSimDrift.addEventListener('click', async () => {
      btnSimDrift.disabled = true;
      btnSimDrift.innerText = 'Simulating Covariate Drift (5x)...';
      try {
        const resp = await apiFetch('/drift/simulate?amount_scale=5.0', { method: 'POST' });
        const res = await resp.json();
        await loadDriftStatus(res);
      } catch (err) {
        alert('Error simulating drift: ' + err.message);
      } finally {
        btnSimDrift.disabled = false;
        btnSimDrift.innerText = 'Simulate Drift (5x Amount Shift)';
      }
    });
  }

  // =========================================================================
  // 10. COST ANALYTICS & BASELINE COMPARISON
  // =========================================================================
  async function loadCostMetrics() {
    try {
      const resp = await apiFetch('/metrics/cost');
      if (!resp.ok) return;
      const data = await resp.json();

      document.getElementById('cost-kpi-saved').textContent = formatRs(data.cost_savings_vs_flag_nothing_rs);
      document.getElementById('cost-kpi-expected').textContent = formatRs(data.model_expected_cost_rs);
      document.getElementById('cost-kpi-thresh-cost').textContent = Number(data.cost_optimal_threshold || 0.0484).toFixed(4);
      document.getElementById('cost-kpi-thresh-f2').textContent = Number(data.f2_optimal_threshold || 0.4094).toFixed(4);

      renderCostComparisonChart(data);
    } catch (e) {
      console.error('Error loading cost metrics:', e);
    }
  }

  function renderCostComparisonChart(costData) {
    const canvas = document.getElementById('costComparisonChart');
    if (!canvas) return;
    destroyChart('costComparison');

    const labels = ['No Model (Flag Nothing)', 'Flag Everything', 'FinSecure AI Model'];
    const costs = [
      costData.baseline_flag_nothing_cost_rs || 35357945.92,
      costData.baseline_flag_everything_cost_rs || 1766600.0,
      costData.model_expected_cost_rs || 393128.67
    ];
    const colors = ['#dc2626', '#f59e0b', '#10b981'];

    charts.costComparison = new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Total Expected Financial Cost (Rs.)',
          data: costs,
          backgroundColor: colors,
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `Cost: ${formatRs(ctx.parsed.y)}`
            }
          }
        },
        scales: {
          y: {
            type: 'logarithmic',
            grid: { color: '#f1f5f9' },
            title: { display: true, text: 'Cost (Rupees, Log Scale)' }
          }
        }
      }
    });
  }

  // =========================================================================
  // 11. RAG CHATBOT UI & CONVERSATION ENGINE
  // =========================================================================
  const chatForm = document.getElementById('chat-form');
  const chatInput = document.getElementById('chat-input');
  const chatHistoryBox = document.getElementById('chat-history-box');
  const btnClearChat = document.getElementById('btn-clear-chat');

  function appendChatBubble(role, text, sources = []) {
    if (!chatHistoryBox) return;
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${role}`;

    let contentHtml = escapeHtml(text).replace(/\n/g, '<br>');
    // Parse basic bold markdown
    contentHtml = contentHtml.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Parse bullet suggestions into clickable elements if bot
    if (role === 'bot') {
      contentHtml = contentHtml.replace(/- <strong>(.*?)<\/strong>/g, '<div style="margin: 4px 0;">• <a href="javascript:void(0)" class="chat-suggestion-link" data-suggest="$1">$1</a></div>');
    }

    if (sources && sources.length > 0 && role === 'bot') {
      const srcText = sources.map(s => escapeHtml(s)).join(' • ');
      contentHtml += `<div class="chat-sources">Sources: ${srcText}</div>`;
    }

    bubble.innerHTML = contentHtml;

    // Attach click listeners to any suggestions
    bubble.querySelectorAll('.chat-suggestion-link').forEach(link => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        const suggestQuery = link.getAttribute('data-suggest');
        if (suggestQuery) handleSendQuery(suggestQuery);
      });
    });

    chatHistoryBox.appendChild(bubble);
    chatHistoryBox.scrollTop = chatHistoryBox.scrollHeight;
  }

  function showTypingIndicator() {
    if (!chatHistoryBox) return;
    removeTypingIndicator();
    const ind = document.createElement('div');
    ind.className = 'chat-bubble bot typing';
    ind.id = 'chat-typing-indicator';
    ind.innerHTML = '<span></span><span></span><span></span>';
    chatHistoryBox.appendChild(ind);
    chatHistoryBox.scrollTop = chatHistoryBox.scrollHeight;
  }

  function removeTypingIndicator() {
    const ind = document.getElementById('chat-typing-indicator');
    if (ind && ind.parentNode) {
      ind.parentNode.removeChild(ind);
    }
  }

  async function handleSendQuery(query) {
    if (!query || !query.trim()) return;
    const qText = query.trim();
    appendChatBubble('user', qText);
    chatHistory.push({ role: 'user', content: qText });

    const btn = document.getElementById('btn-send-chat');
    if (btn) { btn.disabled = true; btn.innerText = '...'; }

    showTypingIndicator();

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);

    try {
      const resp = await apiFetch('/chat', {
        method: 'POST',
        signal: controller.signal,
        body: JSON.stringify({
          query: qText,
          history: chatHistory.slice(-6)
        })
      });

      clearTimeout(timeoutId);
      removeTypingIndicator();

      const res = await resp.json();
      if (!resp.ok) {
        const errorMsg = res.detail || res.answer || res.message || 'Chat request failed';
        throw new Error(errorMsg);
      }

      appendChatBubble('bot', res.answer, res.sources);
      chatHistory.push({ role: 'assistant', content: res.answer });
    } catch (err) {
      clearTimeout(timeoutId);
      removeTypingIndicator();
      if (err.name === 'AbortError') {
        appendChatBubble('bot', '⚠️ Chat request timed out after 30 seconds. The server did not respond in time.');
      } else {
        appendChatBubble('bot', '⚠️ ' + err.message);
      }
    } finally {
      clearTimeout(timeoutId);
      removeTypingIndicator();
      if (btn) { btn.disabled = false; btn.innerText = 'Send'; }
      if (chatHistoryBox) chatHistoryBox.scrollTop = chatHistoryBox.scrollHeight;
    }
  }

  if (chatForm && chatInput) {
    chatForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const val = chatInput.value;
      chatInput.value = '';
      handleSendQuery(val);
    });

    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        const val = chatInput.value;
        chatInput.value = '';
        handleSendQuery(val);
      }
    });
  }

  // Chip queries
  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const q = chip.getAttribute('data-query');
      if (q) handleSendQuery(q);
    });
  });

  if (btnClearChat) {
    btnClearChat.addEventListener('click', () => {
      chatHistory = [];
      if (chatHistoryBox) {
        chatHistoryBox.innerHTML = `
          <div class="chat-bubble bot">
            Hello analyst! Chat history cleared. How may I assist you with fraud scoring, SHAP, or drift analytics?
          </div>
        `;
      }
    });
  }

  // =========================================================================
  // 12. ADMIN USERS TABLE & EXCEL DOWNLOAD
  // =========================================================================
  async function loadUsersTable() {
    if (currentUser.role !== 'admin') return;
    const tbody = document.getElementById('users-table-body');
    if (!tbody) return;

    try {
      const resp = await apiFetch('/auth/users');
      if (!resp.ok) return;
      const users = await resp.json();

      tbody.innerHTML = users.map(u => `
        <tr>
          <td><strong>#${u.id}</strong></td>
          <td>${escapeHtml(u.full_name)}</td>
          <td><code>${escapeHtml(u.email)}</code></td>
          <td><span class="role-badge ${u.role === 'admin' ? 'admin' : ''}">${escapeHtml(u.role)}</span></td>
          <td style="font-size: 12px; color: var(--text-muted);">${escapeHtml(u.created_at || 'N/A')}</td>
          <td style="font-size: 12px; color: var(--text-muted);">${escapeHtml(u.last_login || 'Never')}</td>
        </tr>
      `).join('');
    } catch (e) {
      console.warn('Error loading users:', e);
    }
  }

  const btnDownloadExcel = document.getElementById('btn-download-users-excel');
  if (btnDownloadExcel) {
    btnDownloadExcel.addEventListener('click', async () => {
      try {
        const resp = await apiFetch('/auth/users/export');
        if (!resp.ok) throw new Error('Download failed');
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'users_export.xlsx';
        a.click();
        URL.revokeObjectURL(url);
      } catch (err) {
        alert('Error downloading Excel export: ' + err.message);
      }
    });
  }

  // =========================================================================
  // 8. BATCH CSV FRAUD INTELLIGENCE CONTROLLER
  // =========================================================================
  let currentAnalysisId = null;
  let currentAnalysisSummary = null;
  window.currentAnalysisId = null;
  window.currentAnalysisSummary = null;
  let activeModelThreshold = 0.0484;
  let analysisPollTimer = null;
  let currentUploadedFile = null;
  let autoMappingData = {};
  let previewRowsData = [];
  let detectedHeadersList = [];

  let flaggedTablePage = 1;
  let flaggedTablePageSize = 25;
  let flaggedTableTotalPages = 1;
  let flaggedSearchQuery = '';
  let flaggedFilterType = '';
  let flaggedFilterDecision = '';
  let typeMixMode = 'doughnut';
  let timeGranularityMode = 'hourly';
  let whatifDebounceTimer = null;

  // Stepper Controller
  function goToAnalysisStep(stepNum) {
    const steps = [1, 2, 3, 4, 5];
    steps.forEach(n => {
      const stepItem = document.getElementById(`stepper-step-${n}`);
      const stepPanel = document.getElementById(
        n === 1 ? 'step-panel-upload' :
        n === 2 ? 'step-panel-mapping' :
        n === 3 ? 'step-panel-options' :
        n === 4 ? 'step-panel-processing' : 'step-panel-results'
      );
      if (stepItem) {
        stepItem.classList.remove('active', 'completed');
        if (n < stepNum) stepItem.classList.add('completed');
        if (n === stepNum) stepItem.classList.add('active');
      }
      if (stepPanel) {
        if (n === stepNum) stepPanel.classList.add('active');
        else stepPanel.classList.remove('active');
      }
    });

    if (stepNum === 5) {
      setTimeout(() => {
        ['analysisTypeMix', 'analysisFlagShare', 'analysisFlagByType', 'analysisAmountDist', 'analysisTimeOverTime', 'analysisScoreHist', 'analysisDrained'].forEach(k => {
          if (charts[k]) charts[k].resize();
        });
      }, 100);
    }
  }

  // Results Subtabs Controller
  const resSubtabBtns = document.querySelectorAll('.subtab-btn');
  const resSubtabPanels = document.querySelectorAll('.subtab-content');
  resSubtabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-subtab');
      resSubtabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      resSubtabPanels.forEach(p => {
        if (p.id === targetId) p.classList.add('active');
        else p.classList.remove('active');
      });
      if (targetId === 'res-tab-charts') {
        setTimeout(() => {
          ['analysisTypeMix', 'analysisFlagShare', 'analysisFlagByType', 'analysisAmountDist', 'analysisTimeOverTime', 'analysisScoreHist', 'analysisDrained'].forEach(k => {
            if (charts[k]) charts[k].resize();
          });
        }, 80);
      }
    });
  });

  // Download helper
  async function downloadFromUrl(url, filename) {
    try {
      const resp = await apiFetch(url);
      if (!resp.ok) {
        const errJson = await resp.json().catch(() => ({}));
        throw new Error(errJson.detail || 'Download request failed.');
      }
      const blob = await resp.blob();
      const blobUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = blobUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => {
        try { URL.revokeObjectURL(blobUrl); } catch (_) {}
      }, 10000);
    } catch (err) {
      alert('Download error: ' + err.message);
    }
  }

  // File Upload & Dropzone
  const analysisDropzone = document.getElementById('analysis-dropzone');
  const analysisFileInput = document.getElementById('analysis-file-input');
  const btnBrowseFile = document.getElementById('btn-browse-file');
  const selectedFileDisplay = document.getElementById('selected-file-display');
  const selectedFileName = document.getElementById('selected-file-name');
  const selectedFileSize = document.getElementById('selected-file-size');
  const btnRemoveSelectedFile = document.getElementById('btn-remove-selected-file');
  const btnUploadFile = document.getElementById('btn-upload-file');
  const analysisUploadError = document.getElementById('analysis-upload-error');

  function setSelectedFile(file) {
    currentUploadedFile = file;
    if (analysisUploadError) analysisUploadError.style.display = 'none';
    if (!file) {
      if (selectedFileDisplay) selectedFileDisplay.style.display = 'none';
      if (analysisDropzone) analysisDropzone.style.display = 'block';
      if (btnUploadFile) btnUploadFile.disabled = true;
      if (analysisFileInput) analysisFileInput.value = '';
      return;
    }

    if (selectedFileName) selectedFileName.textContent = file.name;
    if (selectedFileSize) {
      const mb = file.size / (1024 * 1024);
      selectedFileSize.textContent = mb >= 1.0 ? `(${mb.toFixed(2)} MB)` : `(${(file.size / 1024).toFixed(1)} KB)`;
    }
    if (selectedFileDisplay) selectedFileDisplay.style.display = 'block';
    if (analysisDropzone) analysisDropzone.style.display = 'none';
    if (btnUploadFile) btnUploadFile.disabled = false;
  }

  if (btnBrowseFile && analysisFileInput) {
    btnBrowseFile.addEventListener('click', (e) => {
      e.stopPropagation();
      analysisFileInput.click();
    });
  }

  if (analysisDropzone && analysisFileInput) {
    analysisDropzone.addEventListener('click', () => analysisFileInput.click());
    analysisDropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      analysisDropzone.classList.add('dragover');
    });
    analysisDropzone.addEventListener('dragleave', () => analysisDropzone.classList.remove('dragover'));
    analysisDropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      analysisDropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        setSelectedFile(e.dataTransfer.files[0]);
      }
    });
  }

  if (analysisFileInput) {
    analysisFileInput.addEventListener('change', () => {
      if (analysisFileInput.files && analysisFileInput.files.length > 0) {
        setSelectedFile(analysisFileInput.files[0]);
      }
    });
  }

  if (btnRemoveSelectedFile) {
    btnRemoveSelectedFile.addEventListener('click', () => setSelectedFile(null));
  }

  // Sample quick load buttons
  const btnSampleLabeled = document.getElementById('btn-sample-labeled');
  if (btnSampleLabeled) {
    btnSampleLabeled.addEventListener('click', async () => {
      try {
        btnSampleLabeled.disabled = true;
        btnSampleLabeled.textContent = 'Loading sample...';
        const resp = await fetch('/samples/sample_transactions.csv');
        if (!resp.ok) throw new Error('Could not fetch held-out clean-test sample.');
        const blob = await resp.blob();
        const file = new File([blob], 'sample_transactions.csv', { type: 'text/csv' });
        setSelectedFile(file);
      } catch (err) {
        alert('Error loading sample: ' + err.message);
      } finally {
        btnSampleLabeled.disabled = false;
        btnSampleLabeled.textContent = 'Try Held-out Clean-Test Sample (2,000 txs)';
      }
    });
  }

  const btnSampleTraining = document.getElementById('btn-sample-training');
  if (btnSampleTraining) {
    btnSampleTraining.addEventListener('click', async () => {
      try {
        btnSampleTraining.disabled = true;
        btnSampleTraining.textContent = 'Loading sample...';
        const resp = await fetch('/samples/sample_training_rows.csv');
        if (!resp.ok) throw new Error('Could not fetch training rows sample.');
        const blob = await resp.blob();
        const file = new File([blob], 'sample_training_rows.csv', { type: 'text/csv' });
        setSelectedFile(file);
      } catch (err) {
        alert('Error loading training sample: ' + err.message);
      } finally {
        btnSampleTraining.disabled = false;
        btnSampleTraining.textContent = 'Try Training-Rows Sample (2,000 txs)';
      }
    });
  }

  const btnSampleUnlabeled = document.getElementById('btn-sample-unlabeled');
  if (btnSampleUnlabeled) {
    btnSampleUnlabeled.addEventListener('click', async () => {
      try {
        btnSampleUnlabeled.disabled = true;
        btnSampleUnlabeled.textContent = 'Loading sample...';
        const resp = await fetch('/samples/sample_unlabeled.csv');
        if (!resp.ok) throw new Error('Could not fetch sample dataset.');
        const blob = await resp.blob();
        const file = new File([blob], 'sample_unlabeled.csv', { type: 'text/csv' });
        setSelectedFile(file);
      } catch (err) {
        alert('Error loading unlabeled sample: ' + err.message);
      } finally {
        btnSampleUnlabeled.disabled = false;
        btnSampleUnlabeled.textContent = 'Try Unlabeled Sample (2,000 txs)';
      }
    });
  }

  // Upload Action
  if (btnUploadFile) {
    btnUploadFile.addEventListener('click', async () => {
      if (!currentUploadedFile) return;
      if (analysisUploadError) analysisUploadError.style.display = 'none';

      btnUploadFile.disabled = true;
      btnUploadFile.textContent = 'Uploading & Validating...';

      try {
        const formData = new FormData();
        formData.append('file', currentUploadedFile);

        const resp = await apiFetch('/analysis/upload', {
          method: 'POST',
          body: formData,
        });

        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) {
          throw new Error(data.detail || 'Upload failed. Please check your file.');
        }

        currentAnalysisId = data.analysis_id;
        window.currentAnalysisId = currentAnalysisId;
        window.currentUploadMeta = data;
        detectedHeadersList = data.raw_columns || data.headers || [];
        autoMappingData = data.auto_mapping || {};
        previewRowsData = data.preview_rows || data.row_preview || [];

        renderMappingStep();
        goToAnalysisStep(2);
      } catch (err) {
        if (analysisUploadError) {
          analysisUploadError.textContent = err.message;
          analysisUploadError.style.display = 'block';
        }
      } finally {
        btnUploadFile.disabled = false;
        btnUploadFile.textContent = 'Upload & Inspect Headers →';
      }
    });
  }

  // STEP 2: Render Mapping UI & 10-Row Preview
  function renderMappingStep() {
    const grid = document.getElementById('mapping-dropdowns-grid');
    const tableHead = document.getElementById('preview-table-head');
    const tableBody = document.getElementById('preview-table-body');
    const mappingAlert = document.getElementById('mapping-error-alert');
    if (mappingAlert) mappingAlert.style.display = 'none';

    if (!grid) return;
    grid.innerHTML = '';

    const schemaFields = [
      { key: 'step', label: 'Step (Integer hour / sequence)', required: true },
      { key: 'type', label: 'Transaction Type (TRANSFER, etc.)', required: true },
      { key: 'amount', label: 'Amount (Rupees / numeric > 0)', required: true },
      { key: 'nameOrig', label: 'Origin Account ID', required: true },
      { key: 'oldbalanceOrg', label: 'Origin Initial Balance', required: true },
      { key: 'newbalanceOrig', label: 'Origin Final Balance', required: true },
      { key: 'nameDest', label: 'Destination Account ID', required: true },
      { key: 'oldbalanceDest', label: 'Dest Initial Balance', required: true },
      { key: 'newbalanceDest', label: 'Dest Final Balance', required: true },
      { key: 'isFraud', label: 'isFraud (Ground Truth Label)', required: false },
      { key: 'isFlaggedFraud', label: 'isFlaggedFraud (Rule-based flag)', required: false },
    ];

    schemaFields.forEach(field => {
      const fieldDiv = document.createElement('div');
      fieldDiv.className = 'form-group';

      const label = document.createElement('label');
      label.className = 'form-label';
      label.style.fontSize = '12px';
      label.innerHTML = `${escapeHtml(field.label)} ${field.required ? '<span style="color: var(--danger);">*</span>' : '<span style="color: var(--text-muted);">(Optional)</span>'}`;

      const select = document.createElement('select');
      select.className = 'form-select';
      select.id = `map-select-${field.key}`;
      select.style.fontSize = '13px';

      const optNone = document.createElement('option');
      optNone.value = '';
      optNone.textContent = '-- Not Present / Exclude --';
      select.appendChild(optNone);

      detectedHeadersList.forEach(header => {
        const opt = document.createElement('option');
        opt.value = header;
        opt.textContent = header;
        if (autoMappingData[field.key] === header) {
          opt.selected = true;
        }
        select.appendChild(opt);
      });

      fieldDiv.appendChild(label);
      fieldDiv.appendChild(select);
      grid.appendChild(fieldDiv);
    });

    // Render 10-Row Preview Table
    if (tableHead && tableBody) {
      tableHead.innerHTML = '';
      tableBody.innerHTML = '';

      if (detectedHeadersList.length > 0) {
        const trHead = document.createElement('tr');
        detectedHeadersList.forEach(h => {
          const th = document.createElement('th');
          th.textContent = h;
          th.style.fontSize = '11px';
          trHead.appendChild(th);
        });
        tableHead.appendChild(trHead);

        previewRowsData.forEach(row => {
          const tr = document.createElement('tr');
          detectedHeadersList.forEach(h => {
            const td = document.createElement('td');
            td.style.fontSize = '11px';
            td.textContent = row[h] !== undefined ? String(row[h]) : '';
            tr.appendChild(td);
          });
          tableBody.appendChild(tr);
        });
      }
    }
  }

  // Mapping Navigation Buttons
  const btnMappingBack = document.getElementById('btn-mapping-back');
  if (btnMappingBack) {
    btnMappingBack.addEventListener('click', () => goToAnalysisStep(1));
  }

  const btnMappingNext = document.getElementById('btn-mapping-next');
  if (btnMappingNext) {
    btnMappingNext.addEventListener('click', () => {
      const requiredKeys = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest'];
      const missing = [];

      requiredKeys.forEach(k => {
        const el = document.getElementById(`map-select-${k}`);
        if (!el || !el.value) {
          missing.push(k);
        }
      });

      const mappingAlert = document.getElementById('mapping-error-alert');
      if (missing.length > 0) {
        if (mappingAlert) {
          mappingAlert.textContent = `Incomplete schema mapping. Missing required attributes: ${missing.join(', ')}`;
          mappingAlert.style.display = 'block';
        }
        return;
      }

      if (mappingAlert) mappingAlert.style.display = 'none';

      // Populate Step 3 Options with Upload Metadata
      const upMeta = window.currentUploadMeta || {};
      const overlapPct = upMeta.account_overlap_pct !== undefined ? upMeta.account_overlap_pct : 0.0;
      const recMode = upMeta.recommended_graph_mode || (overlapPct >= 20.0 ? 'warmed' : 'cold_start');
      const spanUnder72 = Boolean(upMeta.span_under_72);

      const elOverlapVal = document.getElementById('opt-overlap-val');
      const elOverlapText = document.getElementById('opt-overlap-text');
      const elLowOverlapBanner = document.getElementById('opt-low-overlap-banner');
      const elLowOverlapPct = document.getElementById('opt-low-overlap-pct');
      const elFileOnlyWarn = document.getElementById('opt-file-only-warning');

      if (elOverlapVal) elOverlapVal.textContent = `${overlapPct.toFixed(1)}%`;
      if (elOverlapText) elOverlapText.textContent = `${overlapPct.toFixed(1)}% of accounts seen in training`;

      if (overlapPct < 20.0) {
        if (elLowOverlapBanner) elLowOverlapBanner.style.display = 'block';
        if (elLowOverlapPct) elLowOverlapPct.textContent = `${overlapPct.toFixed(1)}%`;
      } else {
        if (elLowOverlapBanner) elLowOverlapBanner.style.display = 'none';
      }

      if (spanUnder72) {
        if (elFileOnlyWarn) elFileOnlyWarn.style.display = 'block';
      } else {
        if (elFileOnlyWarn) elFileOnlyWarn.style.display = 'none';
      }

      // Default recommendation by overlap rule
      const radioWarmed = document.getElementById('radio-graph-warmed');
      const radioCold = document.getElementById('radio-graph-cold');
      if (recMode === 'warmed') {
        if (radioWarmed) radioWarmed.checked = true;
      } else {
        if (radioCold) radioCold.checked = true;
      }

      goToAnalysisStep(3);
    });
  }

  // STEP 3: Options Config
  const optThresholdSlider = document.getElementById('opt-threshold-slider');
  const optThresholdInput = document.getElementById('opt-threshold-input');
  if (optThresholdSlider && optThresholdInput) {
    optThresholdSlider.addEventListener('input', () => {
      optThresholdInput.value = optThresholdSlider.value;
    });
    optThresholdInput.addEventListener('input', () => {
      optThresholdSlider.value = optThresholdInput.value;
    });
  }

  // Threshold Mode Toggle in Options
  const radioModeModel = document.getElementById('radio-mode-model');
  const radioModeBudget = document.getElementById('radio-mode-budget');
  const groupOptModel = document.getElementById('group-opt-model-thresh');
  const groupOptBudget = document.getElementById('group-opt-alert-budget');
  if (radioModeModel && radioModeBudget) {
    radioModeModel.addEventListener('change', () => {
      if (groupOptModel) groupOptModel.style.display = 'block';
      if (groupOptBudget) groupOptBudget.style.display = 'none';
    });
    radioModeBudget.addEventListener('change', () => {
      if (groupOptModel) groupOptModel.style.display = 'none';
      if (groupOptBudget) groupOptBudget.style.display = 'block';
    });
  }

  const btnOptionsBack = document.getElementById('btn-options-back');
  if (btnOptionsBack) {
    btnOptionsBack.addEventListener('click', () => goToAnalysisStep(2));
  }

  const btnStartScoringJob = document.getElementById('btn-start-scoring-job');
  if (btnStartScoringJob) {
    btnStartScoringJob.addEventListener('click', async () => {
      if (!currentAnalysisId) return;

      const schemaKeys = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud'];
      const columnMapping = {};
      schemaKeys.forEach(k => {
        const el = document.getElementById(`map-select-${k}`);
        if (el && el.value) {
          columnMapping[k] = el.value;
        }
      });

      const optUseLabels = document.getElementById('opt-use-labels-graph');
      const optMaxRows = document.getElementById('opt-max-rows');
      const thresholdVal = optThresholdInput ? parseFloat(optThresholdInput.value) : 0.0484;

      const threshModeEl = document.querySelector('input[name="opt-threshold-mode"]:checked');
      const threshModeVal = threshModeEl ? threshModeEl.value : 'model_threshold';
      const optAlertBudgetInput = document.getElementById('opt-alert-budget-input');
      const alertBudgetVal = optAlertBudgetInput ? parseFloat(optAlertBudgetInput.value) : 2.0;

      btnStartScoringJob.disabled = true;
      btnStartScoringJob.textContent = 'Starting Job...';

      try {
        const formData = new FormData();
        const graphModeEl = document.querySelector('input[name="opt-graph-mode"]:checked');
        const graphModeVal = graphModeEl ? graphModeEl.value : 'warmed';
        formData.append('graph_mode', graphModeVal);
        formData.append('column_mapping', JSON.stringify(columnMapping));
        formData.append('use_labels_for_graph', optUseLabels && optUseLabels.checked ? 'true' : 'false');
        formData.append('threshold', String(thresholdVal));
        formData.append('threshold_mode', threshModeVal);
        formData.append('alert_budget_pct', String(alertBudgetVal));
        formData.append('max_score_rows', optMaxRows ? optMaxRows.value : '300000');

        const resp = await apiFetch(`/analysis/${currentAnalysisId}/start`, {
          method: 'POST',
          body: formData,
        });

        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) {
          throw new Error(data.detail || 'Could not start background job.');
        }

        goToAnalysisStep(4);
        startAnalysisPolling(currentAnalysisId);
      } catch (err) {
        alert('Job initiation error: ' + err.message);
      } finally {
        btnStartScoringJob.disabled = false;
        btnStartScoringJob.textContent = 'Start Background Scoring Job 🚀';
      }
    });
  }

  // STEP 4: Background Polling Controller
  function startAnalysisPolling(analysisId) {
    if (analysisPollTimer) clearInterval(analysisPollTimer);

    const procFill = document.getElementById('proc-progress-fill');
    const procPct = document.getElementById('proc-progress-pct');
    const procStage = document.getElementById('proc-stage-badge');
    const procEta = document.getElementById('proc-eta-badge');
    const procError = document.getElementById('proc-error-alert');
    const procHeading = document.getElementById('proc-status-heading');
    const btnCancel = document.getElementById('btn-cancel-active-job');

    if (procError) procError.style.display = 'none';
    if (btnCancel) btnCancel.disabled = false;

    analysisPollTimer = setInterval(async () => {
      try {
        const resp = await apiFetch(`/analysis/${analysisId}/status`);
        if (!resp.ok) return;

        const data = await resp.json();
        const pct = data.progress_pct !== undefined ? data.progress_pct : 0.0;
        const stage = data.stage || 'reading';
        const eta = data.eta_seconds !== undefined ? data.eta_seconds : 0.0;

        if (procFill) procFill.style.width = `${pct}%`;
        if (procPct) procPct.textContent = `${pct.toFixed(1)}%`;
        if (procStage) procStage.textContent = `Stage: ${stage}`;
        if (procEta) procEta.textContent = eta > 0 ? `ETA: ${eta.toFixed(1)}s` : 'ETA: finalizing...';

        if (data.status === 'done') {
          clearInterval(analysisPollTimer);
          analysisPollTimer = null;
          goToAnalysisStep(5);
          try {
            await loadAnalysisResults(analysisId);
          } catch (resErr) {
            console.error('Error loading results in UI:', resErr);
          }
        } else if (data.status === 'cancelled') {
          clearInterval(analysisPollTimer);
          analysisPollTimer = null;
          if (procHeading) procHeading.textContent = 'Analysis Cancelled by User';
          if (procStage) procStage.textContent = 'Status: cancelled';
          if (btnCancel) btnCancel.disabled = true;
        } else if (data.status === 'failed') {
          clearInterval(analysisPollTimer);
          analysisPollTimer = null;
          if (procHeading) procHeading.textContent = 'Analysis Failed';
          if (procError) {
            procError.textContent = data.error || 'Job failed during execution.';
            procError.style.display = 'block';
          }
          if (btnCancel) btnCancel.disabled = true;
        }
      } catch (e) {
        console.error('Polling error:', e);
      }
    }, 1000);
  }

  // Cancel Button
  const btnCancelActiveJob = document.getElementById('btn-cancel-active-job');
  if (btnCancelActiveJob) {
    btnCancelActiveJob.addEventListener('click', async () => {
      if (!currentAnalysisId) return;
      btnCancelActiveJob.disabled = true;
      btnCancelActiveJob.textContent = 'Cancelling...';
      try {
        await apiFetch(`/analysis/${currentAnalysisId}/cancel`, { method: 'POST' });
      } catch (err) {
        console.error('Cancel error:', err);
      }
    });
  }

  // STEP 5: Results View & Visualizations
  async function loadAnalysisResults(analysisId) {
    currentAnalysisId = analysisId;
    window.currentAnalysisId = currentAnalysisId;
    const resp = await apiFetch(`/analysis/${analysisId}/summary`);
    if (!resp.ok) {
      alert('Could not retrieve analysis summary.');
      return;
    }

    const summary = await resp.json();
    currentAnalysisSummary = summary;
    window.currentAnalysisSummary = currentAnalysisSummary;
    const eda = summary.eda || {};
    const scoring = summary.scoring || {};

    activeModelThreshold = scoring.threshold_used || 0.0484;

    const flCount = scoring.flagged_count || 0;
    const flRate = scoring.flag_rate !== undefined ? scoring.flag_rate : (scoring.flagged_rate || 0.0);
    const flAmount = scoring.flagged_amount !== undefined ? scoring.flagged_amount : (scoring.total_amount_flagged || 0.0);
    const clAmount = scoring.cleared_amount !== undefined ? scoring.cleared_amount : (scoring.amount_cleared || 0.0);
    const trDataset = scoring.training_dataset_name || scoring.dataset_name || 'PaySim Synthetic (300K)';
    const rowsSec = scoring.rows_per_second !== undefined ? scoring.rows_per_second : 0;

    // Overview KPIs
    const elRowsTotal = document.getElementById('kpi-rows-total');
    const elRowsValid = document.getElementById('kpi-rows-valid');
    const elRowsExcluded = document.getElementById('kpi-rows-excluded');
    const elRowsScored = document.getElementById('kpi-rows-scored');
    const elFlaggedCount = document.getElementById('kpi-flagged-count');
    const elFlagRate = document.getElementById('kpi-flag-rate');
    const elFlaggedAmount = document.getElementById('kpi-flagged-amount');
    const elClearedAmount = document.getElementById('kpi-cleared-amount');
    const elStepRange = document.getElementById('kpi-step-range');
    const elAnalysisDate = document.getElementById('kpi-analysis-date');
    const elModelVer = document.getElementById('kpi-model-version');
    const elTrainedOn = document.getElementById('kpi-trained-on');
    const elGraphMode = document.getElementById('kpi-graph-mode');
    const elThroughput = document.getElementById('kpi-throughput');

    if (elRowsTotal) elRowsTotal.textContent = (eda.rows_total || 0).toLocaleString();
    if (elRowsValid) elRowsValid.textContent = (eda.rows_valid || 0).toLocaleString();
    if (elRowsExcluded) elRowsExcluded.textContent = (eda.rows_excluded || 0).toLocaleString();
    if (elRowsScored) elRowsScored.textContent = (scoring.rows_scored || 0).toLocaleString();
    if (elFlaggedCount) elFlaggedCount.textContent = flCount.toLocaleString();
    if (elFlagRate) elFlagRate.textContent = formatPct(flRate);
    if (elFlaggedAmount) elFlaggedAmount.textContent = formatRs(flAmount);
    if (elClearedAmount) elClearedAmount.textContent = formatRs(clAmount);
    if (elStepRange) elStepRange.textContent = `Steps ${eda.step_min || 1} to ${eda.step_max || 1}`;
    if (elAnalysisDate) elAnalysisDate.textContent = new Date(summary.created_at || Date.now()).toLocaleString();
    if (elModelVer) elModelVer.textContent = scoring.model_version || 'xgb_v2.0';
    if (elTrainedOn) elTrainedOn.textContent = trDataset;
    const gmMode = (scoring.scoring_options && scoring.scoring_options.graph_mode) || scoring.graph_mode || 'warmed';
    const gmLabels = {
      warmed: 'Training-Warmed History',
      cold_start: 'Neutral Cold-Start',
      file_only: 'File-Only Graph'
    };
    if (elGraphMode) elGraphMode.textContent = (gmLabels[gmMode] || gmMode) + (scoring.use_labels_for_graph ? ' (Labels Active)' : '');
    if (elThroughput) elThroughput.textContent = `${rowsSec.toLocaleString()} rows/sec`;

    // High Alert Rate (>20%) Banner
    const bannerHighAlert = document.getElementById('res-high-alert-banner');
    const valHighAlert = document.getElementById('res-high-alert-val');
    if (bannerHighAlert) {
      if (flRate > 0.20) {
        if (valHighAlert) valHighAlert.textContent = formatPct(flRate);
        bannerHighAlert.style.display = 'block';
      } else {
        bannerHighAlert.style.display = 'none';
      }
    }

    // Synthetic Dataset Banner
    const bannerSynthetic = document.getElementById('res-synthetic-banner');
    const datasetSynthetic = document.getElementById('res-synthetic-dataset-name');
    if (bannerSynthetic) {
      if (scoring.synthetic_warning || (trDataset && trDataset.toLowerCase().includes('synth'))) {
        if (datasetSynthetic) datasetSynthetic.textContent = trDataset || 'PaySim Synthetic (300K)';
        bannerSynthetic.style.display = 'block';
      } else {
        bannerSynthetic.style.display = 'none';
      }
    }

    // Training Data Overlap Banner
    const bannerOverlap = document.getElementById('res-overlap-banner');
    if (bannerOverlap) {
      bannerOverlap.style.display = 'block';
    }

    // Low account overlap banner (< 20%)
    const bannerLowOverlap = document.getElementById('res-low-overlap-banner');
    const pctLowOverlap = document.getElementById('res-low-overlap-pct');
    const overlapVal = scoring.account_overlap !== undefined ? scoring.account_overlap : (eda.account_overlap || 0.0);
    const overlapPct = scoring.account_overlap_pct !== undefined ? scoring.account_overlap_pct : (overlapVal * 100.0);
    if (bannerLowOverlap) {
      if (overlapVal < 0.20 || scoring.low_overlap_warning) {
        if (pctLowOverlap) pctLowOverlap.textContent = `${overlapPct.toFixed(1)}%`;
        bannerLowOverlap.style.display = 'block';
      } else {
        bannerLowOverlap.style.display = 'none';
      }
    }

    const kpiOverlap = document.getElementById('kpi-account-overlap');
    if (kpiOverlap) {
      kpiOverlap.textContent = `${overlapPct.toFixed(1)}% seen in training`;
    }

    const kpiThreshMode = document.getElementById('kpi-threshold-mode');
    if (kpiThreshMode) {
      kpiThreshMode.textContent = scoring.threshold_mode === 'alert_budget' ? `Alert Budget (${scoring.alert_budget_pct || 2.0}%)` : 'Model Threshold';
    }

    // Ground-Truth Detection Metrics Card in Overview (Alert Rate, Precision, Recall together)
    const overviewMetricsCard = document.getElementById('overview-metrics-card');
    const overviewAlertRate = document.getElementById('overview-alert-rate');
    const overviewPrecision = document.getElementById('overview-precision');
    const overviewRecall = document.getElementById('overview-recall');
    const overviewF1 = document.getElementById('overview-f1');
    if (scoring.has_labels && scoring.metrics) {
      if (overviewMetricsCard) overviewMetricsCard.style.display = 'block';
      const m = scoring.metrics;
      const ar = m.alert_rate !== undefined ? m.alert_rate : flRate;
      if (overviewAlertRate) overviewAlertRate.textContent = formatPct(ar);
      if (overviewPrecision) overviewPrecision.textContent = formatPct(m.precision || 0);
      if (overviewRecall) overviewRecall.textContent = formatPct(m.recall || 0);
      if (overviewF1) overviewF1.textContent = (m.f1 || 0).toFixed(3);
    } else {
      if (overviewMetricsCard) overviewMetricsCard.style.display = 'none';
    }

    // Large file banner
    const bannerLarge = document.getElementById('res-large-file-banner');
    const textLarge = document.getElementById('res-large-file-text');
    if (bannerLarge && textLarge) {
      if ((eda.rows_total || 0) > (scoring.rows_scored || 0)) {
        textLarge.textContent = `EDA on ${(eda.rows_total || 0).toLocaleString()} rows; scored first ${(scoring.rows_scored || 0).toLocaleString()} rows by step.`;
        bannerLarge.style.display = 'block';
      } else {
        bannerLarge.style.display = 'none';
      }
    }

    // Excluded rows banner
    const bannerExcluded = document.getElementById('res-excluded-banner');
    const textExcluded = document.getElementById('res-excluded-text');
    const cardRejectedDownload = document.getElementById('card-rejected-download');
    if (bannerExcluded && textExcluded) {
      if ((eda.rows_excluded || 0) > 0) {
        const reasonParts = [];
        for (const [r, cnt] of Object.entries(eda.rejection_reasons || {})) {
          if (cnt > 0) reasonParts.push(`${r}: ${cnt}`);
        }
        textExcluded.textContent = `${(eda.rows_excluded || 0).toLocaleString()} rows excluded (${reasonParts.join(', ') || 'validation errors'})`;
        bannerExcluded.style.display = 'flex';
        if (cardRejectedDownload) cardRejectedDownload.style.display = 'block';
      } else {
        bannerExcluded.style.display = 'none';
        if (cardRejectedDownload) cardRejectedDownload.style.display = 'none';
      }
    }

    // What-If Threshold Sync
    const whatifSlider = document.getElementById('whatif-slider');
    const whatifDisplay = document.getElementById('whatif-slider-display');
    const badgeActiveModelThresh = document.getElementById('badge-active-model-thresh');
    if (whatifSlider) whatifSlider.value = activeModelThreshold;
    if (whatifDisplay) whatifDisplay.textContent = activeModelThreshold.toFixed(4);
    if (badgeActiveModelThresh) badgeActiveModelThresh.textContent = `Model Default: ${activeModelThreshold.toFixed(4)}`;

    updateWhatIfDisplayValues({
      flagged_count: flCount,
      flag_rate: flRate,
      flagged_amount: flAmount,
      metrics: scoring.metrics,
    });

    // Model Performance Metrics tab visibility
    const subtabMetricsBtn = document.getElementById('btn-subtab-metrics');
    if (subtabMetricsBtn) {
      if (scoring.has_labels && scoring.metrics) {
        subtabMetricsBtn.style.display = 'inline-block';
        renderMetricsTab(scoring.metrics);
      } else {
        subtabMetricsBtn.style.display = 'none';
      }
    }

    // Render Charts
    renderAnalysisCharts(summary);

    // Load Flagged Table
    flaggedTablePage = 1;
    flaggedSearchQuery = '';
    flaggedFilterType = '';
    flaggedFilterDecision = '';
    await loadFlaggedTransactions(1);
  }

  // What-If Slider Value Updater
  function updateWhatIfDisplayValues(data) {
    const elCount = document.getElementById('whatif-stat-flagged');
    const elRate = document.getElementById('whatif-stat-rate');
    const elAmount = document.getElementById('whatif-stat-amount');

    const cnt = data.flagged_count || 0;
    const rate = data.flag_rate !== undefined ? data.flag_rate : (data.flagged_rate || 0.0);
    const amt = data.flagged_amount !== undefined ? data.flagged_amount : (data.total_amount_flagged || 0.0);

    if (elCount) elCount.textContent = cnt.toLocaleString();
    if (elRate) elRate.textContent = formatPct(rate);
    if (elAmount) elAmount.textContent = formatRs(amt);

    if (data.metrics) {
      const m = data.metrics;
      const overviewAlertRate = document.getElementById('overview-alert-rate');
      const overviewPrecision = document.getElementById('overview-precision');
      const overviewRecall = document.getElementById('overview-recall');
      const overviewF1 = document.getElementById('overview-f1');
      if (overviewAlertRate) overviewAlertRate.textContent = formatPct(m.alert_rate !== undefined ? m.alert_rate : rate);
      if (overviewPrecision) overviewPrecision.textContent = formatPct(m.precision || 0);
      if (overviewRecall) overviewRecall.textContent = formatPct(m.recall || 0);
      if (overviewF1) overviewF1.textContent = (m.f1 || 0).toFixed(3);

      const bannerHighAlert = document.getElementById('res-high-alert-banner');
      const valHighAlert = document.getElementById('res-high-alert-val');
      if (bannerHighAlert) {
        if (rate > 0.20) {
          if (valHighAlert) valHighAlert.textContent = formatPct(rate);
          bannerHighAlert.style.display = 'block';
        } else {
          bannerHighAlert.style.display = 'none';
        }
      }

      renderMetricsTab(data.metrics);
    }
  }

  // What-If Mode & Slider Controller
  const whatifSlider = document.getElementById('whatif-slider');
  const whatifDisplay = document.getElementById('whatif-slider-display');
  const whatifRadioModel = document.getElementById('whatif-radio-model');
  const whatifRadioBudget = document.getElementById('whatif-radio-budget');
  let whatifActiveMode = 'model_threshold';

  function configureWhatifSliderForMode(mode) {
    whatifActiveMode = mode;
    if (mode === 'alert_budget') {
      if (whatifSlider) {
        whatifSlider.min = '0.1';
        whatifSlider.max = '10.0';
        whatifSlider.step = '0.1';
        whatifSlider.value = '2.0';
      }
      if (whatifDisplay) whatifDisplay.textContent = '2.0%';
    } else {
      if (whatifSlider) {
        whatifSlider.min = '0.001';
        whatifSlider.max = '0.999';
        whatifSlider.step = '0.0001';
        whatifSlider.value = activeModelThreshold.toFixed(4);
      }
      if (whatifDisplay) whatifDisplay.textContent = activeModelThreshold.toFixed(4);
    }
  }

  if (whatifRadioModel && whatifRadioBudget) {
    whatifRadioModel.addEventListener('change', () => {
      configureWhatifSliderForMode('model_threshold');
      whatifSlider.dispatchEvent(new Event('input'));
    });
    whatifRadioBudget.addEventListener('change', () => {
      configureWhatifSliderForMode('alert_budget');
      whatifSlider.dispatchEvent(new Event('input'));
    });
  }

  // Slider change listener with debounce
  if (whatifSlider && whatifDisplay) {
    whatifSlider.addEventListener('input', () => {
      const val = parseFloat(whatifSlider.value);
      if (whatifActiveMode === 'alert_budget') {
        whatifDisplay.textContent = `${val.toFixed(1)}%`;
      } else {
        whatifDisplay.textContent = val.toFixed(4);
      }

      if (whatifDebounceTimer) clearTimeout(whatifDebounceTimer);
      whatifDebounceTimer = setTimeout(async () => {
        if (!currentAnalysisId) return;
        try {
          const url = whatifActiveMode === 'alert_budget'
            ? `/analysis/${currentAnalysisId}/whatif?threshold_mode=alert_budget&alert_budget_pct=${val}`
            : `/analysis/${currentAnalysisId}/whatif?threshold_mode=model_threshold&threshold=${val}`;
          const resp = await apiFetch(url);
          if (resp.ok) {
            const wData = await resp.json();
            updateWhatIfDisplayValues(wData);
          }
        } catch (e) {
          console.error('What-if fetch error:', e);
        }
      }, 120);
    });
  }

  const btnResetWhatif = document.getElementById('btn-reset-whatif');
  if (btnResetWhatif && whatifSlider) {
    btnResetWhatif.addEventListener('click', () => {
      if (whatifRadioModel) whatifRadioModel.checked = true;
      configureWhatifSliderForMode('model_threshold');
      if (currentAnalysisSummary && currentAnalysisSummary.scoring) {
        const sc = currentAnalysisSummary.scoring;
        updateWhatIfDisplayValues({
          flagged_count: sc.flagged_count || 0,
          flag_rate: sc.flag_rate !== undefined ? sc.flag_rate : (sc.flagged_rate || 0.0),
          flagged_amount: sc.flagged_amount !== undefined ? sc.flagged_amount : (sc.total_amount_flagged || 0.0),
          metrics: sc.metrics,
        });
      }
      whatifSlider.dispatchEvent(new Event('input'));
    });
  }

  // Model Metrics Tab Renderer
  function renderMetricsTab(m) {
    if (!m) return;
    const cm = m.confusion_matrix || {};
    const fc = m.financial_cost || {};
    const tp = cm.tp !== undefined ? cm.tp : (m.tp || 0);
    const fp = cm.fp !== undefined ? cm.fp : (m.fp || 0);
    const fn = cm.fn !== undefined ? cm.fn : (m.fn || 0);
    const tn = cm.tn !== undefined ? cm.tn : (m.tn || 0);

    const tpEl = document.getElementById('metric-tp');
    const fpEl = document.getElementById('metric-fp');
    const fnEl = document.getElementById('metric-fn');
    const tnEl = document.getElementById('metric-tn');

    if (tpEl) tpEl.textContent = tp.toLocaleString();
    if (fpEl) fpEl.textContent = fp.toLocaleString();
    if (fnEl) fnEl.textContent = fn.toLocaleString();
    if (tnEl) tnEl.textContent = tn.toLocaleString();

    const alertRateEl = document.getElementById('val-alert-rate');
    if (alertRateEl) alertRateEl.textContent = formatPct(m.alert_rate !== undefined ? m.alert_rate : 0);

    const precEl = document.getElementById('val-precision');
    const recEl = document.getElementById('val-recall');
    const f1El = document.getElementById('val-f1');
    const f2El = document.getElementById('val-f2');
    const prEl = document.getElementById('val-pr-auc');
    const rocEl = document.getElementById('val-roc-auc');

    if (precEl) precEl.textContent = formatPct(m.precision);
    if (recEl) recEl.textContent = formatPct(m.recall);
    if (f1El) f1El.textContent = (m.f1 || 0).toFixed(3);
    if (f2El) f2El.textContent = (m.f2 || 0).toFixed(3);
    if (prEl) prEl.textContent = (m.pr_auc || 0).toFixed(3);
    if (rocEl) rocEl.textContent = (m.roc_auc || 0).toFixed(3);

    // Rupee Costs
    const costModel = document.getElementById('cost-model-loss');
    const costNothing = document.getElementById('cost-nothing-loss');
    const costEverything = document.getElementById('cost-everything-loss');
    const costSavings = document.getElementById('cost-net-savings');

    const totalCost = fc.total_cost !== undefined ? fc.total_cost : (m.cost_model_rupees || 0);
    const flagNothing = fc.flag_nothing_cost !== undefined ? fc.flag_nothing_cost : (m.cost_flag_nothing_rupees || 0);
    const flagEverything = fc.flag_everything_cost !== undefined ? fc.flag_everything_cost : (m.cost_flag_everything_rupees || 0);
    const savings = fc.savings_vs_nothing !== undefined ? fc.savings_vs_nothing : (m.savings_vs_nothing_rupees || 0);

    if (costModel) costModel.textContent = formatRs(totalCost);
    if (costNothing) costNothing.textContent = formatRs(flagNothing);
    if (costEverything) costEverything.textContent = formatRs(flagEverything);
    if (costSavings) costSavings.textContent = formatRs(savings);

    // Alert Budgets
    const r1 = document.getElementById('budget-recall-1pct');
    const r2 = document.getElementById('budget-recall-2pct');
    const r5 = document.getElementById('budget-recall-5pct');
    const c1 = document.getElementById('budget-count-1pct');
    const c2 = document.getElementById('budget-count-2pct');
    const c5 = document.getElementById('budget-count-5pct');

    if (m.alert_budgets) {
      const b1 = m.alert_budgets['1%'] || m.alert_budgets['1_percent'] || {};
      const b2 = m.alert_budgets['2%'] || m.alert_budgets['2_percent'] || {};
      const b5 = m.alert_budgets['5%'] || m.alert_budgets['5_percent'] || {};

      if (r1) r1.textContent = formatPct(b1.recall || 0);
      if (r2) r2.textContent = formatPct(b2.recall || 0);
      if (r5) r5.textContent = formatPct(b5.recall || 0);
      if (c1) c1.textContent = `${(b1.alert_budget_rows || b1.max_flagged || 0).toLocaleString()} txs`;
      if (c2) c2.textContent = `${(b2.alert_budget_rows || b2.max_flagged || 0).toLocaleString()} txs`;
      if (c5) c5.textContent = `${(b5.alert_budget_rows || b5.max_flagged || 0).toLocaleString()} txs`;
    }

    // Low fraud warning
    const warnLow = document.getElementById('low-fraud-warning-banner');
    if (warnLow) {
      const fraudCount = m.total_fraud_rows !== undefined ? m.total_fraud_rows : (m.fraud_count || (m.confusion_matrix ? (m.confusion_matrix.tp + m.confusion_matrix.fn) : 0));
      if (fraudCount < 100 && fraudCount > 0) {
        let ciText = `⚠️ Small sample warning (${fraudCount} frauds < 100): PR-AUC / ROC-AUC have high variance.`;
        const boot = m.bootstrap_ci || m.bootstrap_ci_95;
        if (boot) {
          const recCi = boot.recall_ci || boot.recall;
          const precCi = boot.precision_ci || boot.precision;
          const recLow = Array.isArray(recCi) ? recCi[0] : (recCi && recCi.low !== undefined ? recCi.low : 0);
          const recHigh = Array.isArray(recCi) ? recCi[1] : (recCi && recCi.high !== undefined ? recCi.high : 0);
          const precLow = Array.isArray(precCi) ? precCi[0] : (precCi && precCi.low !== undefined ? precCi.low : 0);
          const precHigh = Array.isArray(precCi) ? precCi[1] : (precCi && precCi.high !== undefined ? precCi.high : 0);
          ciText += ` Bootstrap 95% CIs: Recall [${formatPct(recLow)}, ${formatPct(recHigh)}], Precision [${formatPct(precLow)}, ${formatPct(precHigh)}].`;
        }
        warnLow.textContent = ciText;
        warnLow.style.display = 'block';
      } else {
        warnLow.style.display = 'none';
      }
    }
  }

  // 7 Interactive Charts Rendering
  function renderAnalysisCharts(summary) {
    const eda = summary.eda || {};
    const scoring = summary.scoring || {};

    // 1. Transaction Type Mix
    destroyChart('analysisTypeMix');
    const ctxType = document.getElementById('chartTypeMix');
    if (ctxType && eda.type_distribution) {
      const labels = Object.keys(eda.type_distribution);
      const values = Object.values(eda.type_distribution);
      const bgColors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];

      charts.analysisTypeMix = new Chart(ctxType, {
        type: typeMixMode,
        data: {
          labels: labels,
          datasets: [{
            data: values,
            backgroundColor: bgColors,
            borderWidth: 1,
            borderColor: '#ffffff',
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'right' },
            tooltip: {
              callbacks: {
                label: (ctx) => ` ${ctx.label}: ${ctx.raw.toLocaleString()} txs (${((ctx.raw / (eda.rows_valid || 1)) * 100).toFixed(1)}%)`
              }
            }
          }
        }
      });
    }

    // Toggle Type Mix Chart
    const btnToggleTypeMix = document.getElementById('btn-toggle-type-mix');
    if (btnToggleTypeMix) {
      btnToggleTypeMix.onclick = () => {
        typeMixMode = typeMixMode === 'doughnut' ? 'bar' : 'doughnut';
        renderAnalysisCharts(currentAnalysisSummary);
      };
    }

    // 2. Flagged vs Cleared Share
    destroyChart('analysisFlagShare');
    const ctxFlagShare = document.getElementById('chartFlagShare');
    if (ctxFlagShare) {
      const flagged = scoring.flagged_count || 0;
      const scored = scoring.rows_scored || 1;
      const cleared = Math.max(0, scored - flagged);
      charts.analysisFlagShare = new Chart(ctxFlagShare, {
        type: 'doughnut',
        data: {
          labels: ['Cleared (LEGIT)', 'Flagged (FRAUD)'],
          datasets: [{
            data: [cleared, flagged],
            backgroundColor: ['#22c55e', '#ef4444'],
            borderWidth: 2,
            borderColor: '#ffffff',
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'bottom' },
            tooltip: {
              callbacks: {
                label: (ctx) => ` ${ctx.label}: ${ctx.raw.toLocaleString()} (${((ctx.raw / scored) * 100).toFixed(2)}%)`
              }
            }
          }
        }
      });
    }

    // 3. Flagged by Type Bar Chart
    destroyChart('analysisFlagByType');
    const ctxFlagByType = document.getElementById('chartFlagByType');
    if (ctxFlagByType && scoring.flagged_by_type) {
      const types = Object.keys(scoring.flagged_by_type);
      const counts = Object.values(scoring.flagged_by_type);
      charts.analysisFlagByType = new Chart(ctxFlagByType, {
        type: 'bar',
        data: {
          labels: types,
          datasets: [{
            label: 'Flagged Count',
            data: counts,
            backgroundColor: '#ef4444',
            borderRadius: 4,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { beginAtZero: true, grid: { color: '#f1f5f9' } },
            x: { grid: { display: false } }
          }
        }
      });
    }

    // 4. Amount Distribution Histogram (Log Bins, Flagged vs Cleared)
    destroyChart('analysisAmountDist');
    const ctxAmt = document.getElementById('chartAmountDist');
    const amtData = scoring.amount_distribution;
    if (ctxAmt && amtData) {
      charts.analysisAmountDist = new Chart(ctxAmt, {
        type: 'bar',
        data: {
          labels: amtData.bin_labels || [],
          datasets: [
            {
              label: 'Cleared (LEGIT)',
              data: amtData.cleared_counts || [],
              backgroundColor: '#93c5fd',
              borderRadius: 3,
            },
            {
              label: 'Flagged (FRAUD)',
              data: amtData.flagged_counts || [],
              backgroundColor: '#ef4444',
              borderRadius: 3,
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { position: 'top' } },
          scales: {
            x: { stacked: true, grid: { display: false } },
            y: { stacked: true, beginAtZero: true, grid: { color: '#f1f5f9' } }
          }
        }
      });
    }

    // 5. Volume Over Steps
    destroyChart('analysisTimeOverTime');
    const ctxTime = document.getElementById('chartTimeOverTime');
    const tsData = scoring.time_series_chart;
    if (ctxTime && tsData) {
      charts.analysisTimeOverTime = new Chart(ctxTime, {
        type: 'line',
        data: {
          labels: tsData.labels || [],
          datasets: [
            {
              label: 'Total Volume',
              data: tsData.total || [],
              borderColor: '#2563eb',
              backgroundColor: 'rgba(37, 99, 235, 0.08)',
              fill: true,
              tension: 0.2,
              borderWidth: 2,
              pointRadius: (tsData.labels || []).length > 50 ? 0 : 3,
            },
            {
              label: 'Flagged Volume',
              data: tsData.flagged || [],
              borderColor: '#ef4444',
              backgroundColor: 'rgba(239, 68, 68, 0.1)',
              fill: false,
              tension: 0.2,
              borderWidth: 2,
              pointRadius: (tsData.labels || []).length > 50 ? 0 : 3,
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { position: 'top' } },
          scales: {
            y: { beginAtZero: true, grid: { color: '#f1f5f9' } },
            x: { grid: { display: false } }
          }
        }
      });
    }

    // 6. Probability Score Histogram
    destroyChart('analysisScoreHist');
    const ctxScore = document.getElementById('chartScoreHist');
    const sh = scoring.prob_histogram;
    if (ctxScore && sh) {
      const binLabels = (sh.bin_edges || []).slice(0, -1).map((edge, i) => `${edge.toFixed(2)}-${sh.bin_edges[i+1].toFixed(2)}`);
      charts.analysisScoreHist = new Chart(ctxScore, {
        type: 'bar',
        data: {
          labels: binLabels,
          datasets: [{
            label: 'Transaction Count',
            data: sh.counts || [],
            backgroundColor: binLabels.map((_, i) => (i >= (binLabels.length * 0.1) ? '#ef4444' : '#3b82f6')),
            borderRadius: 3,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { beginAtZero: true, grid: { color: '#f1f5f9' } },
            x: { grid: { display: false } }
          }
        }
      });
    }

    // 7. Drained Account Breakdown
    destroyChart('analysisDrained');
    const ctxDrained = document.getElementById('chartDrained');
    if (ctxDrained) {
      const drained = scoring.drained_flagged_count || 0;
      const nonDrained = scoring.non_drained_flagged_count || 0;
      charts.analysisDrained = new Chart(ctxDrained, {
        type: 'doughnut',
        data: {
          labels: ['Account Completely Drained', 'Partial Balance Remaining'],
          datasets: [{
            data: [drained, nonDrained],
            backgroundColor: ['#dc2626', '#fbbf24'],
            borderWidth: 2,
            borderColor: '#ffffff',
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { position: 'bottom' } }
        }
      });
    }
  }

  // Flagged Transactions Table Controller
  async function loadFlaggedTransactions(page) {
    if (!currentAnalysisId) return;
    flaggedTablePage = page;

    const tbody = document.getElementById('flagged-table-body');
    const pageInfo = document.getElementById('flagged-page-info');
    const btnPrev = document.getElementById('btn-flagged-prev');
    const btnNext = document.getElementById('btn-flagged-next');

    if (tbody) {
      tbody.innerHTML = '<tr><td colspan="10" style="text-align: center; padding: 20px; color: var(--text-muted);">Loading transactions...</td></tr>';
    }

    try {
      let url = `/analysis/${currentAnalysisId}/rows?page=${page}&page_size=${flaggedTablePageSize}`;
      if (flaggedSearchQuery) url += `&search=${encodeURIComponent(flaggedSearchQuery)}`;
      if (flaggedFilterType) url += `&filter_type=${encodeURIComponent(flaggedFilterType)}`;
      if (flaggedFilterDecision) url += `&decision=${encodeURIComponent(flaggedFilterDecision)}`;

      const resp = await apiFetch(url);
      if (!resp.ok) throw new Error('Could not fetch table rows');
      const data = await resp.json();

      flaggedTableTotalPages = data.total_pages || 1;
      if (pageInfo) {
        pageInfo.textContent = `Page ${data.page} of ${flaggedTableTotalPages} (${data.total_rows.toLocaleString()} transactions)`;
      }

      if (btnPrev) btnPrev.disabled = data.page <= 1;
      if (btnNext) btnNext.disabled = data.page >= flaggedTableTotalPages;

      if (!tbody) return;
      tbody.innerHTML = '';

      if (!data.rows || data.rows.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" style="text-align: center; padding: 20px; color: var(--text-muted);">No records found matching filters.</td></tr>';
        return;
      }

      data.rows.forEach(r => {
        const tr = document.createElement('tr');
        tr.style.cursor = 'pointer';
        tr.style.transition = 'background 0.1s ease';
        tr.title = 'Click to open SHAP risk explanation';
        tr.onmouseover = () => { tr.style.backgroundColor = 'var(--bg-muted)'; };
        tr.onmouseout = () => { tr.style.backgroundColor = ''; };

        const isFraud = (r.decision || '').toUpperCase() === 'FRAUD';
        const prob = r.fraud_probability !== undefined ? (r.fraud_probability * 100).toFixed(2) + '%' : '0.0%';

        tr.innerHTML = `
          <td><strong>#${r.row_id || 0}</strong></td>
          <td>${r.step}</td>
          <td><span class="badge-pill" style="font-size: 11px;">${escapeHtml(r.type)}</span></td>
          <td>${formatRs(r.amount)}</td>
          <td><code>${escapeHtml(r.nameOrig)}</code></td>
          <td><small>${formatRs(r.oldbalanceOrg)} → ${formatRs(r.newbalanceOrig)}</small></td>
          <td><code>${escapeHtml(r.nameDest)}</code></td>
          <td><small>${formatRs(r.oldbalanceDest)} → ${formatRs(r.newbalanceDest)}</small></td>
          <td><strong style="color: ${isFraud ? 'var(--danger)' : 'var(--success)'};">${prob}</strong></td>
          <td><span class="badge-pill" style="background: ${isFraud ? 'var(--danger-light)' : 'var(--success-light)'}; color: ${isFraud ? 'var(--danger)' : 'var(--success)'}; font-weight: 700;">${escapeHtml(r.decision)}</span></td>
        `;

        tr.addEventListener('click', () => {
          openAnalysisRowShap(r);
        });

        tbody.appendChild(tr);
      });
    } catch (e) {
      if (tbody) {
        tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; color: var(--danger); padding: 20px;">Error: ${escapeHtml(e.message)}</td></tr>`;
      }
    }
  }

  // Row Click: On-Demand SHAP Explanation in Drawer
  async function openAnalysisRowShap(row) {
    const drawer = document.getElementById('transaction-drawer');
    const backdrop = document.getElementById('drawer-backdrop');
    const drawerTitle = document.getElementById('drawer-title');
    const drawerInfo = document.getElementById('drawer-info-grid');
    const drawerText = document.getElementById('drawer-explanation-text');

    if (!drawer || !backdrop) return;
    drawerTitle.textContent = `Analysis Tx #${row.row_id} — ${row.type}`;

    if (drawerInfo) {
      drawerInfo.innerHTML = `
        <div><strong>Type:</strong> ${escapeHtml(row.type)}</div>
        <div><strong>Amount:</strong> ${formatRs(row.amount)}</div>
        <div><strong>Step:</strong> ${row.step}</div>
        <div><strong>Decision:</strong> <span style="font-weight: 700; color: ${row.decision === 'FRAUD' ? 'var(--danger)' : 'var(--success)'};">${escapeHtml(row.decision)}</span></div>
        <div><strong>Origin Acct:</strong> <code>${escapeHtml(row.nameOrig)}</code></div>
        <div><strong>Dest Acct:</strong> <code>${escapeHtml(row.nameDest)}</code></div>
        <div><strong>Orig Balance:</strong> ${formatRs(row.oldbalanceOrg)} → ${formatRs(row.newbalanceOrig)}</div>
        <div><strong>Dest Balance:</strong> ${formatRs(row.oldbalanceDest)} → ${formatRs(row.newbalanceDest)}</div>
        <div><strong>Fraud Prob:</strong> ${(row.fraud_probability * 100).toFixed(2)}%</div>
        <div><strong>Threshold:</strong> ${activeModelThreshold.toFixed(4)}</div>
      `;
    }

    if (drawerText) {
      drawerText.textContent = 'Computing on-demand SHAP TreeExplainer values...';
    }

    drawer.classList.add('open');
    backdrop.classList.add('open');

    try {
      const rowIdx = (row.row_id || 1) - 1;
      const resp = await apiFetch(`/analysis/${currentAnalysisId}/shap/${rowIdx}`);
      if (!resp.ok) throw new Error('Could not compute SHAP for row');
      const shapData = await resp.json();

      if (drawerText) {
        drawerText.textContent = shapData.explanation || 'No textual explanation returned.';
      }

      // Render SHAP Horizontal Bar Chart
      destroyChart('drawerShap');
      const ctxShap = document.getElementById('drawerShapChart');
      if (ctxShap && shapData.shap_top_features) {
        const top5 = shapData.shap_top_features;
        const labels = top5.map(f => f.feature);
        const values = top5.map(f => f.shap_value);
        const bgColors = values.map(v => v >= 0 ? '#ef4444' : '#22c55e');

        charts.drawerShap = new Chart(ctxShap, {
          type: 'bar',
          data: {
            labels: labels,
            datasets: [{
              label: 'SHAP Contribution',
              data: values,
              backgroundColor: bgColors,
              borderRadius: 4,
            }]
          },
          options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { display: false },
              tooltip: {
                callbacks: {
                  label: (ctx) => ` Impact: ${ctx.raw >= 0 ? '+' : ''}${ctx.raw.toFixed(4)}`
                }
              }
            },
            scales: {
              x: { grid: { color: '#f1f5f9' } },
              y: { grid: { display: false } }
            }
          }
        });
      }
    } catch (err) {
      if (drawerText) {
        drawerText.textContent = 'Error computing SHAP values: ' + err.message;
      }
    }
  }

  // Table pagination and filters listeners
  const btnFlaggedPrev = document.getElementById('btn-flagged-prev');
  if (btnFlaggedPrev) {
    btnFlaggedPrev.addEventListener('click', () => {
      if (flaggedTablePage > 1) loadFlaggedTransactions(flaggedTablePage - 1);
    });
  }

  const btnFlaggedNext = document.getElementById('btn-flagged-next');
  if (btnFlaggedNext) {
    btnFlaggedNext.addEventListener('click', () => {
      if (flaggedTablePage < flaggedTableTotalPages) loadFlaggedTransactions(flaggedTablePage + 1);
    });
  }

  const flaggedPageSizeSel = document.getElementById('flagged-page-size');
  if (flaggedPageSizeSel) {
    flaggedPageSizeSel.addEventListener('change', () => {
      flaggedTablePageSize = parseInt(flaggedPageSizeSel.value, 10);
      loadFlaggedTransactions(1);
    });
  }

  const flaggedSearchInput = document.getElementById('flagged-search-input');
  if (flaggedSearchInput) {
    let searchDebounce = null;
    flaggedSearchInput.addEventListener('input', () => {
      clearTimeout(searchDebounce);
      searchDebounce = setTimeout(() => {
        flaggedSearchQuery = flaggedSearchInput.value.trim();
        loadFlaggedTransactions(1);
      }, 250);
    });
  }

  const flaggedFilterTypeEl = document.getElementById('flagged-filter-type');
  if (flaggedFilterTypeEl) {
    flaggedFilterTypeEl.addEventListener('change', () => {
      flaggedFilterType = flaggedFilterTypeEl.value;
      loadFlaggedTransactions(1);
    });
  }

  const flaggedFilterDecisionEl = document.getElementById('flagged-filter-decision');
  if (flaggedFilterDecisionEl) {
    flaggedFilterDecisionEl.addEventListener('change', () => {
      flaggedFilterDecision = flaggedFilterDecisionEl.value;
      loadFlaggedTransactions(1);
    });
  }

  const btnExportFlaggedCsv = document.getElementById('btn-export-flagged-csv');
  if (btnExportFlaggedCsv) {
    btnExportFlaggedCsv.addEventListener('click', () => {
      if (!currentAnalysisId) return;
      downloadFromUrl(`/analysis/${currentAnalysisId}/export/csv`, `flagged_analysis_${currentAnalysisId.slice(0, 8)}.csv`);
    });
  }

  // Downloads tab actions
  const btnDlResultsCsv = document.getElementById('btn-download-results-csv');
  if (btnDlResultsCsv) {
    btnDlResultsCsv.addEventListener('click', () => {
      if (!currentAnalysisId) return;
      downloadFromUrl(`/analysis/${currentAnalysisId}/export/csv`, `results_${currentAnalysisId.slice(0, 8)}.csv`);
    });
  }

  const btnDlSummaryXlsx = document.getElementById('btn-download-summary-xlsx');
  if (btnDlSummaryXlsx) {
    btnDlSummaryXlsx.addEventListener('click', () => {
      if (!currentAnalysisId) return;
      downloadFromUrl(`/analysis/${currentAnalysisId}/export/xlsx`, `summary_${currentAnalysisId.slice(0, 8)}.xlsx`);
    });
  }

  const btnDlRejectedCsv = document.getElementById('btn-download-rejected-csv');
  if (btnDlRejectedCsv) {
    btnDlRejectedCsv.addEventListener('click', () => {
      if (!currentAnalysisId) return;
      downloadFromUrl(`/analysis/${currentAnalysisId}/export/rejected`, `rejected_rows_${currentAnalysisId.slice(0, 8)}.csv`);
    });
  }

  const btnDlRejectedBanner = document.getElementById('btn-dl-rejected-banner');
  if (btnDlRejectedBanner) {
    btnDlRejectedBanner.addEventListener('click', () => {
      if (!currentAnalysisId) return;
      downloadFromUrl(`/analysis/${currentAnalysisId}/export/rejected`, `rejected_rows_${currentAnalysisId.slice(0, 8)}.csv`);
    });
  }

  // Header "New Upload" Button
  const btnNewAnalysis = document.getElementById('btn-new-analysis');
  if (btnNewAnalysis) {
    btnNewAnalysis.addEventListener('click', () => {
      setSelectedFile(null);
      currentAnalysisId = null;
      window.currentAnalysisId = null;
      currentAnalysisSummary = null;
      window.currentAnalysisSummary = null;
      goToAnalysisStep(1);
    });
  }

  // "My Analyses" Modal Controller
  const modalMyAnalyses = document.getElementById('modal-my-analyses');
  const btnOpenMyAnalyses = document.getElementById('btn-open-my-analyses');
  const btnCloseAnalysesModal = document.getElementById('btn-close-analyses-modal');
  const analysesTableBody = document.getElementById('analyses-table-body');

  function openAnalysesModal() {
    if (modalMyAnalyses) modalMyAnalyses.classList.add('active');
    loadMyAnalysesList();
  }

  function closeAnalysesModal() {
    if (modalMyAnalyses) modalMyAnalyses.classList.remove('active');
  }

  if (btnOpenMyAnalyses) btnOpenMyAnalyses.addEventListener('click', openAnalysesModal);
  if (btnCloseAnalysesModal) btnCloseAnalysesModal.addEventListener('click', closeAnalysesModal);

  async function loadMyAnalysesList() {
    if (!analysesTableBody) return;
    analysesTableBody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 20px;">Loading your analyses...</td></tr>';

    try {
      const resp = await apiFetch('/analysis/list');
      if (!resp.ok) throw new Error('Could not fetch analyses list');
      const data = await resp.json();
      const analyses = data.analyses || [];

      analysesTableBody.innerHTML = '';
      if (analyses.length === 0) {
        analysesTableBody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 20px;">No past analyses found. Click "New Upload" to get started!</td></tr>';
        return;
      }

      analyses.forEach(a => {
        const tr = document.createElement('tr');
        const isDone = a.status === 'done';

        tr.innerHTML = `
          <td><strong>${escapeHtml(a.display_name || a.filename)}</strong></td>
          <td><small>${escapeHtml(a.filename)}</small></td>
          <td>${(a.rows_total || 0).toLocaleString()}</td>
          <td><span class="badge-pill" style="font-size: 11px; font-weight: 700; ${isDone ? 'background: var(--success-light); color: var(--success);' : 'background: var(--bg-muted);'}">${escapeHtml(a.status)}</span></td>
          <td><small>${new Date(a.created_at).toLocaleDateString()}</small></td>
          <td>
            <div style="display: flex; gap: 6px;">
              ${isDone ? `<button class="btn btn-primary btn-sm btn-open-analysis" data-id="${a.id}" style="padding: 2px 8px; font-size: 11px;">Open</button>` : ''}
              <button class="btn btn-secondary btn-sm btn-rename-analysis" data-id="${a.id}" data-name="${escapeHtml(a.display_name || a.filename)}" style="padding: 2px 8px; font-size: 11px;">Rename</button>
              <button class="btn btn-danger btn-sm btn-delete-analysis" data-id="${a.id}" style="padding: 2px 8px; font-size: 11px;">Delete</button>
            </div>
          </td>
        `;

        // Open button
        const btnOpen = tr.querySelector('.btn-open-analysis');
        if (btnOpen) {
          btnOpen.addEventListener('click', async () => {
            closeAnalysesModal();
            await loadAnalysisResults(a.id);
            goToAnalysisStep(5);
          });
        }

        // Rename button
        const btnRename = tr.querySelector('.btn-rename-analysis');
        if (btnRename) {
          btnRename.addEventListener('click', async () => {
            const currentName = btnRename.getAttribute('data-name');
            const newName = prompt('Enter new display name for this analysis:', currentName);
            if (newName && newName.trim() && newName.trim() !== currentName) {
              const formData = new FormData();
              formData.append('new_name', newName.trim());
              await apiFetch(`/analysis/${a.id}/rename`, {
                method: 'PATCH',
                body: formData,
              });
              loadMyAnalysesList();
            }
          });
        }

        // Delete button
        const btnDelete = tr.querySelector('.btn-delete-analysis');
        if (btnDelete) {
          btnDelete.addEventListener('click', async () => {
            if (confirm(`Are you sure you want to permanently delete "${a.display_name || a.filename}"? This will delete all parquet results and uploaded files.`)) {
              await apiFetch(`/analysis/${a.id}`, { method: 'DELETE' });
              if (currentAnalysisId === a.id) {
                currentAnalysisId = null;
                currentAnalysisSummary = null;
                goToAnalysisStep(1);
              }
              loadMyAnalysesList();
            }
          });
        }

        analysesTableBody.appendChild(tr);
      });
    } catch (err) {
      analysesTableBody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--danger); padding: 20px;">Error loading analyses: ${escapeHtml(err.message)}</td></tr>`;
    }
  }

  // =========================================================================
  // 9. APP INITIALIZATION
  // =========================================================================
  async function initApp() {
    initWebSocket();
    await loadRecentPredictions();
    renderScorerShapChart();
  }

  window.FinSecureApp = {
    predictionsMap,
    addOrUpdateTransaction,
    loadRecentPredictions,
    getSocket: () => scanSocket,
    reconnectWebSocket: () => {
      if (scanSocket) {
        scanSocket.close();
      }
      initWebSocket();
    }
  };

  initApp();

})();
