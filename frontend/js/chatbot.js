// Process Q&A Chatbot widget handler for FinSecure AI Dashboard

document.addEventListener('DOMContentLoaded', () => {
    // Inject floating Chatbot button and slide-out drawer into DOM
    const chatbotHtml = `
    <!-- Floating Chatbot Toggle Button -->
    <button id="chatbotToggleBtn" class="chatbot-toggle-btn" title="Open FinSecure Process Q&A Assistant">
        💬 <span class="chatbot-btn-text">Fraud Operations AI</span>
    </button>

    <!-- Slide-out Chatbot Drawer -->
    <div id="chatbotDrawer" class="chatbot-drawer">
        <div class="chatbot-header">
            <div class="chatbot-header-title">
                <span>🤖</span>
                <div>
                    <strong>Process Q&A Assistant</strong>
                    <div style="font-size: 0.75rem; color: #8b949e; font-weight: normal;">RAG Operations & Fraud Insights</div>
                </div>
            </div>
            <button id="chatbotCloseBtn" class="chatbot-close-btn">&times;</button>
        </div>

        <div id="chatbotMessages" class="chatbot-messages">
            <div class="chat-msg bot-msg">
                👋 Hello! I am your <strong>FinSecure Fraud Operations Assistant</strong>. Ask me about historical fraud stats, specific transaction IDs, top risk factors, or recent alerts!
            </div>
        </div>

        <form id="chatbotForm" class="chatbot-input-form">
            <input type="text" id="chatbotInput" class="chatbot-input" placeholder="Ask a question (e.g., 'What is the overall fraud rate?')..." required>
            <button type="submit" class="chatbot-send-btn">Send</button>
        </form>
    </div>
    `;

    document.body.insertAdjacentHTML('beforeend', chatbotHtml);

    // DOM References
    const toggleBtn = document.getElementById('chatbotToggleBtn');
    const closeBtn = document.getElementById('chatbotCloseBtn');
    const drawer = document.getElementById('chatbotDrawer');
    const form = document.getElementById('chatbotForm');
    const input = document.getElementById('chatbotInput');
    const messagesContainer = document.getElementById('chatbotMessages');

    if (!toggleBtn || !drawer) return;

    toggleBtn.addEventListener('click', () => {
        drawer.classList.toggle('active');
        if (drawer.classList.contains('active')) {
            input.focus();
        }
    });

    closeBtn.addEventListener('click', () => {
        drawer.classList.remove('active');
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const userMsg = input.value.trim();
        if (!userMsg) return;

        // Render User Message
        appendMessage('user', userMsg);
        input.value = '';

        // Render Loading Indicator
        const loadingId = appendLoading();

        try {
            const apiBase = (typeof API_BASE !== 'undefined') ? API_BASE : 'http://localhost:8000';
            const res = await fetch(`${apiBase}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: userMsg })
            });

            removeLoading(loadingId);

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || `Server status ${res.status}`);
            }

            const data = await res.json();
            appendMessage('bot', data.answer);
        } catch (err) {
            removeLoading(loadingId);
            appendMessage('bot', `⚠️ Sorry, I encountered an error: ${err.message}`);
        }
    });

    function appendMessage(sender, text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-msg ${sender === 'user' ? 'user-msg' : 'bot-msg'}`;
        msgDiv.innerHTML = text.replace(/\n/g, '<br>');
        messagesContainer.appendChild(msgDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function appendLoading() {
        const id = 'loading-' + Date.now();
        const loadDiv = document.createElement('div');
        loadDiv.id = id;
        loadDiv.className = 'chat-msg bot-msg chat-loading';
        loadDiv.innerHTML = '⚡ <em>Querying DB and synthesizing answer...</em>';
        messagesContainer.appendChild(loadDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        return id;
    }

    function removeLoading(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }
});
