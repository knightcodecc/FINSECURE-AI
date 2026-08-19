// Batch Analysis script for batch.html

// State
let mappedBatchData = null; // { summary: {}, rows: [] }
let currentFilter = 'ALL';
let currentPage = 1;
const pageSize = 50;

let probDistChart = null;
let typeBreakdownChart = null;

// Dark Theme Color Tokens
const COLOR_SAFE = '#00ff88';
const COLOR_FRAUD = '#ff4444';
const COLOR_WARNING = '#ffa500';
const COLOR_BLUE = '#3b82f6';
const COLOR_TEXT = '#f0f6fc';
const COLOR_MUTED = '#8b949e';
const COLOR_BORDER = '#30363d';

// DOM Elements
const csvFileInput = document.getElementById('csvFileInput');
const fileNameDisplay = document.getElementById('fileNameDisplay');
const analyzeCsvBtn = document.getElementById('analyzeCsvBtn');
const loadHoldoutBtn = document.getElementById('loadHoldoutBtn');
const batchLoading = document.getElementById('batchLoading');

const summaryPanel = document.getElementById('summaryPanel');
const confusionMatrixPanel = document.getElementById('confusionMatrixPanel');
const chartsPanel = document.getElementById('chartsPanel');
const tablePanel = document.getElementById('tablePanel');
const batchTableBody = document.getElementById('batchTableBody');

const prevPageBtn = document.getElementById('prevPageBtn');
const nextPageBtn = document.getElementById('nextPageBtn');
const pageIndicator = document.getElementById('pageIndicator');
const filterBtns = document.querySelectorAll('.filter-btn');
const exportFlaggedBtn = document.getElementById('exportFlaggedBtn');

// Connection Status UI
const connectionDot = document.getElementById('connectionDot');
const connectionText = document.getElementById('connectionText');

// File Selection Handler
if (csvFileInput) {
    csvFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            fileNameDisplay.textContent = file.name;
            analyzeCsvBtn.disabled = false;
        } else {
            fileNameDisplay.textContent = 'No file chosen';
            analyzeCsvBtn.disabled = true;
        }
    });
}

// Process CSV content via POST /predict/batch/csv
async function processCsvData(csvBody) {
    if (batchLoading) batchLoading.classList.add('active');
    if (analyzeCsvBtn) analyzeCsvBtn.disabled = true;
    if (loadHoldoutBtn) loadHoldoutBtn.disabled = true;

    if (summaryPanel) summaryPanel.style.display = 'none';
    if (confusionMatrixPanel) confusionMatrixPanel.style.display = 'none';
    if (chartsPanel) chartsPanel.style.display = 'none';
    if (tablePanel) tablePanel.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE}/predict/batch/csv`, {
            method: 'POST',
            headers: { 'Content-Type': 'text/csv' },
            body: csvBody
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Server error ${response.status}`);
        }

        const data = await response.json();
        mappedBatchData = data;

        renderSummary(data.summary);
        renderConfusionMatrix(data.summary);
        renderCharts(data.rows || []);
        
        currentPage = 1;
        renderTable();

        if (summaryPanel) summaryPanel.style.display = 'block';
        if (chartsPanel) chartsPanel.style.display = 'block';
        if (tablePanel) tablePanel.style.display = 'block';

    } catch (err) {
        console.error('Batch scoring failed:', err);
        alert(`Batch scoring error: ${err.message}`);
    } finally {
        if (batchLoading) batchLoading.classList.remove('active');
        if (analyzeCsvBtn && csvFileInput && csvFileInput.files[0]) analyzeCsvBtn.disabled = false;
        if (loadHoldoutBtn) loadHoldoutBtn.disabled = false;
    }
}

// Event: Analyze Uploaded CSV
if (analyzeCsvBtn) {
    analyzeCsvBtn.addEventListener('click', () => {
        const file = csvFileInput.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = async (e) => {
            const content = e.target.result;
            await processCsvData(content);
        };
        reader.readAsText(file);
    });
}

// Event: Load Pre-Split Holdout Dataset
if (loadHoldoutBtn) {
    loadHoldoutBtn.addEventListener('click', async () => {
        if (batchLoading) batchLoading.classList.add('active');
        if (analyzeCsvBtn) analyzeCsvBtn.disabled = true;
        if (loadHoldoutBtn) loadHoldoutBtn.disabled = true;

        try {
            const res = await fetch(`${API_BASE}/archive/holdout`);
            if (!res.ok) throw new Error('Failed to fetch holdout dataset sample');
            
            const jsonRows = await res.json();
            if (!jsonRows || jsonRows.length === 0) throw new Error('Holdout sample is empty');
            
            // Convert JSON array to CSV format
            const headers = Object.keys(jsonRows[0]).join(',');
            const rows = jsonRows.map(row => Object.values(row).join(',')).join('\n');
            const csvText = `${headers}\n${rows}`;

            await processCsvData(csvText);

        } catch (err) {
            console.error('Holdout dataset load failed:', err);
            alert(`Holdout dataset error: ${err.message}`);
            if (batchLoading) batchLoading.classList.remove('active');
            if (analyzeCsvBtn && csvFileInput && csvFileInput.files[0]) analyzeCsvBtn.disabled = false;
            if (loadHoldoutBtn) loadHoldoutBtn.disabled = false;
        }
    });
}

// Render Summary Panel
function renderSummary(summary) {
    if (!summary) return;

    document.getElementById('summaryTotalScored').textContent = formatNumber(summary.total_scored);
    document.getElementById('summaryFlagged').textContent = formatNumber(summary.flagged_count);
    
    // Show evaluation metrics only if ground truth was present
    const hasGt = summary.has_ground_truth && summary.actual_fraud_count !== null && summary.actual_fraud_count !== undefined;
    document.getElementById('summaryAccuracy').textContent = (hasGt && summary.accuracy !== null) ? `${(summary.accuracy * 100).toFixed(2)}%` : 'N/A';
    document.getElementById('summaryPrecision').textContent = (hasGt && summary.precision !== null) ? `${(summary.precision * 100).toFixed(2)}%` : 'N/A';
    document.getElementById('summaryRecall').textContent = (hasGt && summary.recall !== null) ? `${(summary.recall * 100).toFixed(2)}%` : 'N/A';
    document.getElementById('summaryF1').textContent = (hasGt && summary.f1_score !== null) ? `${(summary.f1_score * 100).toFixed(2)}%` : 'N/A';
    
    document.getElementById('summaryLatency').textContent = `${(summary.avg_processing_time_ms || 0).toFixed(2)} ms/tx`;

    const noticeBanner = document.getElementById('groundTruthNoticeBanner');
    if (noticeBanner) {
        noticeBanner.style.display = hasGt ? 'none' : 'block';
    }
}

// Render Confusion Matrix (Ground Truth Present)
function renderConfusionMatrix(summary) {
    if (!summary || summary.actual_fraud_count === null || summary.actual_fraud_count === undefined) {
        if (confusionMatrixPanel) confusionMatrixPanel.style.display = 'none';
        return;
    }

    if (confusionMatrixPanel) confusionMatrixPanel.style.display = 'block';
    
    document.getElementById('cmTN').textContent = formatNumber(summary.true_negatives ?? 0);
    document.getElementById('cmFP').textContent = formatNumber(summary.false_positives ?? 0);
    document.getElementById('cmFN').textContent = formatNumber(summary.false_negatives ?? 0);
    document.getElementById('cmTP').textContent = formatNumber(summary.true_positives ?? 0);
}

// Render Chart.js Analytics (Histogram & Type Breakdown)
function renderCharts(rows) {
    if (!rows || rows.length === 0) return;

    // 1. Fraud Probability Histogram Bins
    const bins = [0, 0, 0, 0, 0]; // 0-20%, 20-40%, 40-60%, 60-80%, 80-100%
    rows.forEach(r => {
        const p = (r.fraud_probability || 0) * 100;
        if (p < 20) bins[0]++;
        else if (p < 40) bins[1]++;
        else if (p < 60) bins[2]++;
        else if (p < 80) bins[3]++;
        else bins[4]++;
    });

    const pCtx = document.getElementById('probDistChart');
    if (pCtx) {
        if (probDistChart) probDistChart.destroy();
        probDistChart = new Chart(pCtx, {
            type: 'bar',
            data: {
                labels: ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'],
                datasets: [{
                    label: 'Transaction Count',
                    data: bins,
                    backgroundColor: [COLOR_SAFE, COLOR_BLUE, COLOR_WARNING, '#ec4899', COLOR_FRAUD],
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: COLOR_MUTED }, grid: { color: COLOR_BORDER } },
                    y: { ticks: { color: COLOR_MUTED }, grid: { color: COLOR_BORDER } }
                }
            }
        });
    }

    // 2. Flagged Transactions by Type
    const typeCounts = {};
    rows.filter(r => r.predicted_label).forEach(r => {
        const t = r.type || 'UNKNOWN';
        typeCounts[t] = (typeCounts[t] || 0) + 1;
    });

    const typeLabels = Object.keys(typeCounts).length > 0 ? Object.keys(typeCounts) : ['No Fraud Flagged'];
    const typeValues = Object.keys(typeCounts).length > 0 ? Object.values(typeCounts) : [0];

    const tCtx = document.getElementById('typeBreakdownChart');
    if (tCtx) {
        if (typeBreakdownChart) typeBreakdownChart.destroy();
        typeBreakdownChart = new Chart(tCtx, {
            type: 'bar',
            data: {
                labels: typeLabels,
                datasets: [{
                    label: 'Flagged Count',
                    data: typeValues,
                    backgroundColor: COLOR_FRAUD,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: COLOR_MUTED }, grid: { color: COLOR_BORDER } },
                    y: { ticks: { color: COLOR_MUTED }, grid: { color: COLOR_BORDER } }
                }
            }
        });
    }
}

// Filter Buttons Event Handlers
filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        filterBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentFilter = btn.getAttribute('data-filter');
        currentPage = 1;
        renderTable();
    });
});

// Render Results Table
function renderTable() {
    if (!mappedBatchData || !batchTableBody) return;

    let filteredRows = mappedBatchData.rows || [];
    if (currentFilter === 'FLAGGED') {
        filteredRows = filteredRows.filter(r => r.predicted_label);
    } else if (currentFilter !== 'ALL') {
        filteredRows = filteredRows.filter(r => r.result_type === currentFilter);
    }

    const totalPages = Math.ceil(filteredRows.length / pageSize) || 1;
    if (currentPage > totalPages) currentPage = totalPages;

    const startIdx = (currentPage - 1) * pageSize;
    const pageRows = filteredRows.slice(startIdx, startIdx + pageSize);

    if (pageRows.length === 0) {
        batchTableBody.innerHTML = `
            <tr>
                <td colspan="7" style="text-align: center; color: var(--text-secondary); padding: 2rem;">
                    No transactions match filter '${currentFilter}'
                </td>
            </tr>
        `;
    } else {
        batchTableBody.innerHTML = pageRows.map(r => {
            const probPct = ((r.fraud_probability || 0) * 100).toFixed(1);
            const isFraud = r.predicted_label;
            
            let matchBadge = '<span class="match-badge grey">--</span>';
            if (r.result_type === 'TP') matchBadge = '<span class="match-badge tp">‼️ True Positive</span>';
            else if (r.result_type === 'FP') matchBadge = '<span class="match-badge fp">⚠️ False Positive</span>';
            else if (r.result_type === 'TN') matchBadge = '<span class="match-badge tn">🛡️ True Negative</span>';
            else if (r.result_type === 'FN') matchBadge = '<span class="match-badge fn">🚨 False Negative</span>';

            const actualDisplay = r.actual_is_fraud !== null && r.actual_is_fraud !== undefined 
                ? (r.actual_is_fraud === 1 ? '‼️ 1 (Fraud)' : '🛡️ 0 (Safe)') 
                : 'N/A';

            const riskClass = getRiskLevelClass(r.risk_level);
            const riskIcon = getRiskIcon(isFraud, r.risk_level);

            return `
                <tr class="${isFraud ? 'fraud-row' : ''}">
                    <td class="id-cell">#${r.row_id}</td>
                    <td>${r.type}</td>
                    <td class="amount-cell">₹${formatNumber((r.amount || 0).toFixed(2))}</td>
                    <td>
                        <div class="confidence-container">
                            <div class="confidence-bar-bg">
                                <div class="confidence-bar-fill ${isFraud ? 'fraud' : 'safe'}" style="width: ${probPct}%"></div>
                            </div>
                            <span class="confidence-text">${probPct}%</span>
                        </div>
                    </td>
                    <td>
                        <span class="risk-badge ${riskClass}">
                            ${riskIcon} ${isFraud ? 'FRAUD' : 'SAFE'}
                        </span>
                    </td>
                    <td>${actualDisplay}</td>
                    <td>${matchBadge}</td>
                </tr>
            `;
        }).join('');
    }

    pageIndicator.textContent = `Page ${currentPage} of ${totalPages} (${filteredRows.length} rows)`;
    prevPageBtn.disabled = (currentPage <= 1);
    nextPageBtn.disabled = (currentPage >= totalPages);
}

// Pagination Controls
if (prevPageBtn) {
    prevPageBtn.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            renderTable();
        }
    });
}

if (nextPageBtn) {
    nextPageBtn.addEventListener('click', () => {
        currentPage++;
        renderTable();
    });
}

// Export Flagged Transactions CSV
if (exportFlaggedBtn) {
    exportFlaggedBtn.addEventListener('click', () => {
        if (!mappedBatchData || !mappedBatchData.rows) {
            alert('No transaction data to export.');
            return;
        }

        const flagged = mappedBatchData.rows.filter(r => r.predicted_label);
        if (flagged.length === 0) {
            alert('No flagged fraud transactions found in this batch.');
            return;
        }

        const headers = ['Row ID', 'Step', 'Type', 'Amount (INR)', 'Fraud Probability', 'Risk Level', 'Actual isFraud', 'Match Result'];
        const csvRows = [headers.join(',')];

        flagged.forEach(r => {
            const rowStr = [
                r.row_id,
                r.step,
                r.type,
                (r.amount || 0).toFixed(2),
                ((r.fraud_probability || 0) * 100).toFixed(2) + '%',
                r.risk_level || 'LOW',
                r.actual_is_fraud !== null ? r.actual_is_fraud : '',
                r.result_type || ''
            ].map(val => `"${val}"`).join(',');
            csvRows.push(rowStr);
        });

        const csvContent = csvRows.join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.setAttribute('href', url);
        link.setAttribute('download', `finsecure_flagged_transactions_${Date.now()}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
}

// Initialization & WebSocket Status
document.addEventListener('DOMContentLoaded', () => {
    new WebSocketWrapper(`${WS_BASE}/ws/stats`, {
        onStateChange: (state, detail) => {
            if (!connectionDot || !connectionText) return;
            if (state === 'connected') {
                connectionDot.className = 'status-dot';
                connectionText.textContent = 'Connected';
            } else if (state === 'reconnecting') {
                connectionDot.className = 'status-dot reconnecting';
                connectionText.textContent = 'Reconnecting...';
            } else {
                connectionDot.className = 'status-dot disconnected';
                connectionText.textContent = 'Disconnected';
            }
        }
    });
});
