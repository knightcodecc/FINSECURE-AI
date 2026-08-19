// Dataset & Training Dashboard script for dataset.html

let classBalanceChart = null;
let typeDistChart = null;
let featureImportanceChart = null;

// Dark Theme Color Tokens
const COLOR_SAFE = '#00ff88';
const COLOR_FRAUD = '#ff4444';
const COLOR_WARNING = '#ffa500';
const COLOR_TEXT = '#f0f6fc';
const COLOR_MUTED = '#8b949e';
const COLOR_BORDER = '#30363d';

async function fetchAndRenderDatasetMetrics() {
    try {
        const response = await fetch(`${API_BASE}/stats/dataset-metrics`);
        if (!response.ok) throw new Error(`HTTP Error ${response.status}`);
        
        const data = await response.json();
        renderDatasetOverview(data);
        renderModelPerformance(data);

    } catch (err) {
        console.error('Failed to fetch dataset metrics:', err);
    }
}

function renderDatasetOverview(data) {
    document.getElementById('dsTotalTx').textContent = formatNumber(data.total_transactions);
    document.getElementById('dsLegitCount').textContent = formatNumber(data.legitimate_count);
    document.getElementById('dsFraudCount').textContent = formatNumber(data.fraud_count);
    document.getElementById('dsFraudRatio').textContent = `${data.fraud_percentage.toFixed(4)}%`;

    if (data.amount_stats) {
        document.getElementById('amtMean').textContent = `₹${formatNumber(data.amount_stats.mean.toFixed(2))}`;
        document.getElementById('amtMedian').textContent = `₹${formatNumber(data.amount_stats.median.toFixed(2))}`;
        document.getElementById('amtMin').textContent = `₹${formatNumber(data.amount_stats.min.toFixed(2))}`;
        document.getElementById('amtMax').textContent = `₹${formatNumber(data.amount_stats.max.toFixed(2))}`;
    }

    // Render Class Balance Donut Chart
    const cbCtx = document.getElementById('classBalanceChart');
    if (cbCtx) {
        if (classBalanceChart) classBalanceChart.destroy();
        classBalanceChart = new Chart(cbCtx, {
            type: 'doughnut',
            data: {
                labels: ['Legitimate (99.87%)', 'Fraudulent (0.13%)'],
                datasets: [{
                    data: [data.legitimate_count, data.fraud_count],
                    backgroundColor: [COLOR_SAFE, COLOR_FRAUD],
                    borderColor: COLOR_BORDER,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: COLOR_TEXT, font: { size: 12 } }
                    }
                }
            }
        });
    }

    // Render Transaction Type Bar Chart
    const tdCtx = document.getElementById('typeDistChart');
    if (tdCtx && data.transaction_types) {
        if (typeDistChart) typeDistChart.destroy();
        const types = Object.keys(data.transaction_types);
        const counts = Object.values(data.transaction_types);

        typeDistChart = new Chart(tdCtx, {
            type: 'bar',
            data: {
                labels: types,
                datasets: [{
                    label: 'Transaction Count',
                    data: counts,
                    backgroundColor: ['#3b82f6', '#8b5cf6', '#ec4899', COLOR_WARNING, '#10b981'],
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { ticks: { color: COLOR_MUTED }, grid: { color: COLOR_BORDER } },
                    y: { ticks: { color: COLOR_MUTED }, grid: { color: COLOR_BORDER } }
                }
            }
        });
    }
}

function renderModelPerformance(data) {
    if (!data.model_metrics) return;
    const m = data.model_metrics;

    document.getElementById('mdlAccuracy').textContent = `${(m.accuracy * 100).toFixed(2)}%`;
    document.getElementById('mdlPrecision').textContent = `${(m.precision * 100).toFixed(2)}%`;
    document.getElementById('mdlRecall').textContent = `${(m.recall * 100).toFixed(2)}%`;
    document.getElementById('mdlF1').textContent = `${(m.f1_score * 100).toFixed(2)}%`;
    document.getElementById('mdlRocAuc').textContent = m.roc_auc.toFixed(4);

    if (m.confusion_matrix) {
        document.getElementById('cmTN').textContent = formatNumber(m.confusion_matrix.true_negatives);
        document.getElementById('cmFP').textContent = formatNumber(m.confusion_matrix.false_positives);
        document.getElementById('cmFN').textContent = formatNumber(m.confusion_matrix.false_negatives);
        document.getElementById('cmTP').textContent = formatNumber(m.confusion_matrix.true_positives);
    }

    // Render Feature Importance Horizontal Bar Chart
    const fiCtx = document.getElementById('featureImportanceChart');
    if (fiCtx && data.feature_importances) {
        if (featureImportanceChart) featureImportanceChart.destroy();
        const features = data.feature_importances.map(f => f.feature);
        const importances = data.feature_importances.map(f => f.importance);

        featureImportanceChart = new Chart(fiCtx, {
            type: 'bar',
            data: {
                labels: features,
                datasets: [{
                    label: 'Feature Importance',
                    data: importances,
                    backgroundColor: COLOR_SAFE,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { ticks: { color: COLOR_MUTED }, grid: { color: COLOR_BORDER } },
                    y: { ticks: { color: COLOR_MUTED }, grid: { color: COLOR_BORDER } }
                }
            }
        });
    }
}

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    fetchAndRenderDatasetMetrics();

    new WebSocketWrapper(`${WS_BASE}/ws/stats`, {
        onStateChange: (state, detail) => {
            const connectionDot = document.getElementById('connectionDot');
            const connectionText = document.getElementById('connectionText');
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
