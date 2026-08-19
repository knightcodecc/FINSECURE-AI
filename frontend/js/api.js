// Centralized API Configuration
const API_BASE = 'http://localhost:8000';
const WS_BASE = 'ws://localhost:8000';

// Formatting Helpers
function formatNumber(num) {
    if (num === null || num === undefined || isNaN(num)) return "0";
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function formatTime(timestamp) {
    if (!timestamp) return new Date().toLocaleTimeString();
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
}

function formatTxId(id) {
    if (!id) return '--------';
    return id.substring(0, 8);
}

function getRiskLevelClass(level) {
    if (!level) return 'low';
    return level.toLowerCase();
}

function getRiskIcon(isFraud, riskLevel) {
    if (isFraud || riskLevel === 'CRITICAL') return '‼️';
    if (riskLevel === 'HIGH') return '🚨';
    if (riskLevel === 'MEDIUM') return '⚠️';
    return '🛡️';
}

function getRiskLabel(isFraud, riskLevel) {
    const icon = getRiskIcon(isFraud, riskLevel);
    if (isFraud) {
        return `${icon} ${riskLevel || 'FRAUD'}`;
    }
    return `${icon} ${riskLevel || 'LEGITIMATE'}`;
}

// Reusable WebSocket Wrapper with Exponential-Backoff Reconnect
class WebSocketWrapper {
    constructor(url, options = {}) {
        this.url = url;
        this.onMessage = options.onMessage || (() => {});
        this.onStateChange = options.onStateChange || (() => {});
        this.autoConnect = options.autoConnect !== false;

        this.ws = null;
        this.state = 'disconnected'; // 'connected' | 'reconnecting' | 'disconnected' | 'connecting'
        this.reconnectAttempts = 0;
        this.baseDelay = 1000; // 1s
        this.maxDelay = 30000; // 30s
        this.reconnectTimer = null;
        this.isManuallyClosed = false;

        if (this.autoConnect) {
            this.connect();
        }
    }

    setState(newState, detail = '') {
        this.state = newState;
        this.onStateChange(this.state, detail);
    }

    connect() {
        this.isManuallyClosed = false;
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }

        if (this.reconnectAttempts === 0) {
            this.setState('connecting');
        } else {
            this.setState('reconnecting', `Attempt ${this.reconnectAttempts}`);
        }

        try {
            this.ws = new WebSocket(this.url);
        } catch (err) {
            this.handleFailure();
            return;
        }

        this.ws.onopen = () => {
            this.reconnectAttempts = 0;
            this.setState('connected');
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.onMessage(data, event);
            } catch (e) {
                this.onMessage(event.data, event);
            }
        };

        this.ws.onerror = () => {
            // Handled by onclose
        };

        this.ws.onclose = () => {
            if (this.isManuallyClosed) {
                this.setState('disconnected');
            } else {
                this.handleFailure();
            }
        };
    }

    handleFailure() {
        if (this.isManuallyClosed) return;
        this.reconnectAttempts++;
        const delay = Math.min(this.baseDelay * Math.pow(2, this.reconnectAttempts - 1), this.maxDelay);
        const nextSec = Math.round(delay / 1000);
        this.setState('reconnecting', `Retrying in ${nextSec}s...`);

        this.reconnectTimer = setTimeout(() => {
            this.connect();
        }, delay);
    }

    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(typeof data === 'string' ? data : JSON.stringify(data));
            return true;
        }
        return false;
    }

    close() {
        this.isManuallyClosed = true;
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.setState('disconnected');
    }
}
