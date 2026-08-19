// Dashboard script for index.html

// State
let transactionHistory = [];
const MAX_HISTORY = 20;

// DOM Elements
const form = document.getElementById('transactionForm');
const resultCard = document.getElementById('resultCard');
const loadingIndicator = document.getElementById('loadingIndicator');
const historyBody = document.getElementById('historyBody');
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

// WebSocket Connection Wrapper for Dashboard Stats/Status
let dashboardSocket = null;

function initDashboardConnection() {
    dashboardSocket = new WebSocketWrapper(`${WS_BASE}/ws/stats`, {
        autoConnect: true,
        onStateChange: (state, detail) => {
            updateConnectionUI(state, detail);
        },
        onMessage: (data) => {
            // Stats update payload if needed
        }
    });
}

// Render result card
function renderResult(prediction) {
    const probability = prediction.fraud_probability * 100;
    const riskLevel = prediction.risk_level;
    const isFraud = prediction.is_fraud;

    const riskFactorsHTML = (prediction.top_risk_factors || []).map(factor => `
        <div class="risk-factor">
            <span class="risk-factor-name">${factor.feature}</span>
            <div class="risk-factor-bar-container">
                <div class="risk-factor-bar" style="width: ${Math.min(100, Math.max(0, factor.importance * 100))}%"></div>
            </div>
            <span class="risk-factor-value">${typeof factor.value === 'number' ? factor.value.toFixed(2) : factor.value}</span>
        </div>
    `).join('');

    resultCard.innerHTML = `
        <div class="risk-gauge-container">
            <div class="risk-gauge">
                <div class="gauge-cover">
                    <div class="gauge-value" style="color: ${isFraud ? 'var(--accent-fraud)' : 'var(--accent-safe)'}">
                        ${probability.toFixed(1)}%
                    </div>
                    <div class="gauge-label">Fraud Probability</div>
                </div>
            </div>
        </div>

        <div class="risk-level-container">
            <span class="risk-badge ${getRiskLevelClass(riskLevel)}">
                ${getRiskIcon(isFraud, riskLevel)} ${riskLevel}
            </span>
        </div>

        ${prediction.explanation ? `
            <div class="explanation-box">
                <div class="explanation-title">🤖 AI Analysis</div>
                <div class="explanation-text">${prediction.explanation}</div>
            </div>
        ` : ''}

        <div class="risk-factors">
            <div class="explanation-title">Top Risk Factors</div>
            ${riskFactorsHTML}
        </div>

        <div class="processing-time">
            Processing Time: <span>${prediction.processing_time_ms}ms</span>
        </div>
    `;
}

// Render history table
function renderHistory() {
    if (!historyBody) return;

    if (transactionHistory.length === 0) {
        historyBody.innerHTML = `
            <tr>
                <td colspan="6" style="text-align: center; color: var(--text-secondary); padding: 2rem;">
                    No transactions analyzed yet
                </td>
            </tr>
        `;
        return;
    }

    historyBody.innerHTML = transactionHistory.map(tx => `
        <tr class="${tx.is_fraud ? 'fraud' : 'legitimate'}">
            <td class="id-cell">${formatTxId(tx.transaction_id)}...</td>
            <td>${tx.type}</td>
            <td class="amount-cell">₹${formatNumber(tx.amount.toFixed(2))}</td>
            <td>${(tx.fraud_probability * 100).toFixed(1)}%</td>
            <td>
                <span class="risk-badge ${getRiskLevelClass(tx.risk_level)}">
                    ${getRiskIcon(tx.is_fraud, tx.risk_level)} ${tx.risk_level}
                </span>
            </td>
            <td class="id-cell">${formatTime(tx.timestamp)}</td>
        </tr>
    `).join('');
}

// Form submission handler
if (form) {
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        form.style.display = 'none';
        if (loadingIndicator) loadingIndicator.classList.add('active');

        const transaction = {
            step: parseInt(document.getElementById('step').value) || 1,
            type: document.getElementById('type').value,
            amount: parseFloat(document.getElementById('amount').value),
            oldbalanceOrg: parseFloat(document.getElementById('oldbalanceOrg').value) || 0,
            newbalanceOrig: parseFloat(document.getElementById('newbalanceOrig').value) || 0,
            oldbalanceDest: parseFloat(document.getElementById('oldbalanceDest').value) || 0,
            newbalanceDest: parseFloat(document.getElementById('newbalanceDest').value) || 0
        };

        try {
            const response = await fetch(`${API_BASE}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(transaction)
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            const prediction = await response.json();

            transactionHistory.unshift({
                transaction_id: prediction.transaction_id,
                type: transaction.type,
                amount: transaction.amount,
                fraud_probability: prediction.fraud_probability,
                risk_level: prediction.risk_level,
                is_fraud: prediction.is_fraud,
                timestamp: prediction.timestamp
            });

            if (transactionHistory.length > MAX_HISTORY) {
                transactionHistory = transactionHistory.slice(0, MAX_HISTORY);
            }

            renderResult(prediction);
            renderHistory();

        } catch (error) {
            console.error('Error:', error);
            alert('Failed to analyze transaction. Please check if the API is running.');
        } finally {
            if (loadingIndicator) loadingIndicator.classList.remove('active');
            form.style.display = 'block';
        }
    });
}

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    initDashboardConnection();
    renderHistory();
});
