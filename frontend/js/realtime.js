// Realtime Scanner script for realtime.html

const WS_URL = `${WS_BASE}/ws/scan`;

// State
let wsWrapper = null;
let isRunning = false;
let isPaused = false;
const MAX_TRANSACTIONS = 50;
let sessionStartTime = null;

let stats = {
    totalScanned: 0,
    totalFraud: 0,
    totalFraudAmount: 0,
    scanRate: 0,
    fraudRate: 0
};

let fraudHistory = new Array(60).fill(0);
let historyIndex = 0;

// DOM Elements
const feedList = document.getElementById('feedList');
const startBtn = document.getElementById('startBtn');
const pauseBtn = document.getElementById('pauseBtn');
const stopBtn = document.getElementById('stopBtn');
const speedSelect = document.getElementById('speedSelect');
const autoScrollCheckbox = document.getElementById('autoScroll');
const alertPopup = document.getElementById('alertPopup');
const alertAmount = document.getElementById('alertAmount');
const connectionDot = document.getElementById('connectionDot');
const connectionText = document.getElementById('connectionText');

// Connection UI update helper
function updateConnectionUI(state, detail = '') {
    if (!connectionDot || !connectionText) return;
    
    if (state === 'connected') {
        connectionDot.className = 'status-dot';
        connectionText.textContent = 'Connected';
    } else if (state === 'reconnecting') {
        connectionDot.className = 'status-dot reconnecting';
        connectionText.textContent = detail ? `Reconnecting... (${detail})` : 'Reconnecting...';
    } else if (state === 'connecting') {
        connectionDot.className = 'status-dot reconnecting';
        connectionText.textContent = 'Connecting...';
    } else {
        connectionDot.className = 'status-dot disconnected';
        connectionText.textContent = 'Disconnected';
    }
}

// Update statistics display
function updateStats() {
    const totalScannedEl = document.getElementById('totalScanned');
    const totalFraudEl = document.getElementById('totalFraud');
    const fraudRateEl = document.getElementById('fraudRate');
    const rateFillEl = document.getElementById('rateFill');
    const fraudAmountEl = document.getElementById('fraudAmount');
    const tpsValueEl = document.getElementById('tpsValue');

    if (totalScannedEl) totalScannedEl.textContent = formatNumber(stats.totalScanned);
    if (totalFraudEl) totalFraudEl.textContent = formatNumber(stats.totalFraud);
    if (fraudRateEl) fraudRateEl.textContent = (stats.fraudRate * 100).toFixed(2) + '%';
    if (rateFillEl) rateFillEl.style.width = (stats.fraudRate * 100) + '%';
    if (fraudAmountEl) fraudAmountEl.textContent = '₹' + formatNumber(Math.floor(stats.totalFraudAmount));
    if (tpsValueEl) tpsValueEl.textContent = stats.scanRate.toFixed(1);
}

// Update chart
function updateChart() {
    const container = document.getElementById('chartContainer');
    if (!container) return;

    const maxValue = Math.max(...fraudHistory, 1);
    container.innerHTML = fraudHistory.map(value => {
        const height = (value / maxValue) * 100;
        return `<div class="chart-bar" style="height: ${height}%"></div>`;
    }).join('');
}

// Show alert
function showAlert(amount) {
    if (!alertPopup || !alertAmount) return;
    alertAmount.textContent = '₹' + formatNumber(amount.toFixed(2));
    alertPopup.classList.add('show');

    setTimeout(() => {
        alertPopup.classList.remove('show');
    }, 3000);
}

// Add transaction to feed
function addTransaction(tx) {
    if (!feedList) return;

    const noData = feedList.querySelector('.no-data');
    if (noData) feedList.innerHTML = '';

    const row = document.createElement('div');
    row.className = 'tx-row' + (tx.is_fraud ? ' fraud' : '');

    row.innerHTML = `
        <span class="tx-id">${formatTxId(tx.transaction_id)}</span>
        <span class="tx-type">${tx.type}</span>
        <span class="tx-amount">₹${formatNumber(tx.amount.toFixed(2))}</span>
        <span class="tx-badge ${getRiskLevelClass(tx.risk_level)}">
            ${getRiskIcon(tx.is_fraud, tx.risk_level)} ${tx.risk_level}
        </span>
        <span class="tx-prob">${(tx.fraud_probability * 100).toFixed(1)}%</span>
        <span class="tx-time">${formatTime(tx.timestamp)}</span>
    `;

    feedList.insertBefore(row, feedList.firstChild);

    if (feedList.children.length > MAX_TRANSACTIONS) {
        feedList.removeChild(feedList.lastChild);
    }

    if (autoScrollCheckbox && autoScrollCheckbox.checked) {
        feedList.scrollTop = 0;
    }

    if (tx.risk_level === 'CRITICAL' || tx.is_fraud) {
        showAlert(tx.amount);
    }
}

// Connect using WebSocketWrapper
function connectScanner() {
    if (wsWrapper) {
        wsWrapper.close();
    }

    wsWrapper = new WebSocketWrapper(WS_URL, {
        autoConnect: true,
        onStateChange: (state, detail) => {
            updateConnectionUI(state, detail);
            if (state === 'connected') {
                isRunning = true;
                if (!sessionStartTime) sessionStartTime = Date.now();
                if (startBtn) startBtn.disabled = true;
                if (pauseBtn) pauseBtn.disabled = false;
                if (stopBtn) stopBtn.disabled = false;
            } else if (state === 'disconnected') {
                isRunning = false;
            }
        },
        onMessage: (tx) => {
            if (isPaused) return;

            addTransaction(tx);

            stats.totalScanned++;
            if (tx.is_fraud) {
                stats.totalFraud++;
                stats.totalFraudAmount += tx.amount;
                fraudHistory[historyIndex] = (fraudHistory[historyIndex] || 0) + 1;
            }

            stats.fraudRate = stats.totalFraud / stats.totalScanned;
            updateStats();
        }
    });
}

function stopScanner() {
    if (wsWrapper) {
        wsWrapper.close();
        wsWrapper = null;
    }
    isRunning = false;
    isPaused = false;
    if (startBtn) startBtn.disabled = false;
    if (pauseBtn) pauseBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = true;
    updateConnectionUI('disconnected');
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    if (startBtn) {
        startBtn.addEventListener('click', () => {
            if (!isRunning) {
                connectScanner();
            } else if (isPaused) {
                isPaused = false;
                if (pauseBtn) pauseBtn.textContent = 'Pause';
            }
        });
    }

    if (pauseBtn) {
        pauseBtn.addEventListener('click', () => {
            if (isPaused) {
                isPaused = false;
                pauseBtn.textContent = 'Pause';
            } else {
                isPaused = true;
                pauseBtn.textContent = 'Resume';
            }
        });
    }

    if (stopBtn) {
        stopBtn.addEventListener('click', () => {
            stopScanner();
            if (feedList) feedList.innerHTML = '<div class="no-data">Click "Start" to begin scanning...</div>';
            stats = {
                totalScanned: 0,
                totalFraud: 0,
                totalFraudAmount: 0,
                scanRate: 0,
                fraudRate: 0
            };
            sessionStartTime = null;
            fraudHistory = new Array(60).fill(0);
            updateStats();
            updateChart();
        });
    }

    if (speedSelect) {
        speedSelect.addEventListener('change', (e) => {
            console.log('Speed changed to:', e.target.value, 'ms');
        });
    }

    setInterval(() => {
        if (isRunning && !isPaused) {
            historyIndex = (historyIndex + 1) % 60;
            if (sessionStartTime) {
                const elapsedSeconds = (Date.now() - sessionStartTime) / 1000;
                stats.scanRate = elapsedSeconds > 0 ? (stats.totalScanned / elapsedSeconds) : 0;
            }
            updateStats();
            updateChart();
        }
    }, 1000);

    updateChart();
    updateConnectionUI('disconnected');
});
