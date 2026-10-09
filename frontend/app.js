let shapChartInstance = null;
let psiChartInstance = null;

const API_BASE = "http://localhost:8000";

// Initialize Charts
function initShapChart(labels = [], data = [], colors = []) {
  const ctx = document.getElementById('shapChart').getContext('2d');
  if (shapChartInstance) shapChartInstance.destroy();

  shapChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'SHAP Contribution',
        data: data,
        backgroundColor: colors,
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } },
        y: { grid: { display: false }, ticks: { color: '#f3f4f6' } }
      }
    }
  });
}

function initPsiChart(labels = [], data = []) {
  const ctx = document.getElementById('psiChart').getContext('2d');
  if (psiChartInstance) psiChartInstance.destroy();

  psiChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Feature PSI',
        data: data,
        backgroundColor: data.map(v => v > 0.25 ? '#ef4444' : (v > 0.1 ? '#f59e0b' : '#3b82f6')),
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { color: '#9ca3af', font: { size: 10 } } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } }
      }
    }
  });
}

// Submit Transaction Form
document.getElementById('tx-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const payload = {
    step: 1,
    type: document.getElementById('type').value,
    amount: parseFloat(document.getElementById('amount').value),
    nameOrig: document.getElementById('nameOrig').value,
    oldbalanceOrg: parseFloat(document.getElementById('oldbalanceOrg').value),
    newbalanceOrig: parseFloat(document.getElementById('newbalanceOrig').value),
    nameDest: document.getElementById('nameDest').value,
    oldbalanceDest: parseFloat(document.getElementById('oldbalanceDest').value),
    newbalanceDest: parseFloat(document.getElementById('newbalanceDest').value)
  };

  try {
    const resp = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const res = await resp.json();

    // Update Pill Container
    const pillContainer = document.getElementById('result-pill-container');
    if (res.is_fraud) {
      pillContainer.innerHTML = `<span class="result-pill pill-fraud">FLAGGED FRAUD (${(res.prediction_proba * 100).toFixed(1)}%)</span>`;
    } else {
      pillContainer.innerHTML = `<span class="result-pill pill-clean">CLEARED LEGITIMATE (${(res.prediction_proba * 100).toFixed(1)}%)</span>`;
    }

    // Update Text Explanation
    document.getElementById('explanation-text').innerText = res.explanation;

    // Render SHAP Bar Chart
    const labels = res.shap_top_features.map(f => f.human_name);
    const values = res.shap_top_features.map(f => f.shap_value);
    const colors = values.map(v => v > 0 ? '#ef4444' : '#10b981'); // Red = fraud, Green = legit

    initShapChart(labels, values, colors);

  } catch (err) {
    alert("Error connecting to server API: " + err);
  }
});

// Simulate Drift Button
document.getElementById('btn-simulate-drift').addEventListener('click', async () => {
  try {
    const resp = await fetch(`${API_BASE}/drift/simulate`, { method: 'POST' });
    const res = await resp.json();

    document.getElementById('drift-overall-status').innerText = res.overall_status;
    document.getElementById('drift-score-psi').innerText = res.score_psi.toFixed(4);
    document.getElementById('drifted-count').innerText = res.drifted_feature_count;

    if (res.retrain_recommended) {
      document.getElementById('retrain-banner').style.display = 'block';
    }

    const labels = Object.keys(res.feature_psi).slice(0, 10);
    const values = Object.values(res.feature_psi).slice(0, 10);
    initPsiChart(labels, values);

  } catch (err) {
    alert("Error triggering drift simulation: " + err);
  }
});

// RAG Chatbot Send
document.getElementById('btn-chat-send').addEventListener('click', async () => {
  const inputEl = document.getElementById('chat-input');
  const query = inputEl.value.trim();
  if (!query) return;

  const chatBox = document.getElementById('chat-box');
  chatBox.innerHTML += `<div class="chat-msg"><span class="chat-user">You:</span> ${query}</div>`;
  inputEl.value = '';

  try {
    const resp = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query })
    });
    const res = await resp.json();
    chatBox.innerHTML += `<div class="chat-msg"><span class="chat-bot">Bot:</span> ${res.answer}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;
  } catch (err) {
    chatBox.innerHTML += `<div class="chat-msg"><span class="chat-bot">Bot:</span> Error querying chatbot API.</div>`;
  }
});

// On Load Initializers
window.addEventListener('DOMContentLoaded', () => {
  initShapChart(["Transaction Amount", "Sender Account Drained", "Sender Connection to Flagged"], [1.2, 0.8, 0.5], ['#ef4444', '#ef4444', '#ef4444']);
  initPsiChart(["amount", "orig_in_degree", "log_amount", "hour_of_day"], [0.02, 0.05, 0.01, 0.03]);
});
