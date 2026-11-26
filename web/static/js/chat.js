import { streamQuery } from './api.js';
import { renderSafeMarkdown } from './markdown.js';

export const STARTERS = [
    { text: 'कम्पनी दर्ता कसरी गर्ने?', sub: 'Step-by-step registration procedure' },
    { text: 'दफा ४३ मा के व्यवस्था छ?', sub: 'Statutory provision lookup' },
    { text: 'What documents are needed to register a private company?', sub: 'Checklist & offices' },
    { text: 'How to register a non-profit company in Nepal?', sub: 'Section 166 requirements' }
];

export function createChatController({
    chatScroll,
    chatColumn,
    queryInput,
    sendBtn,
    welcomeTemplate,
    getSessionId,
    ensureSessionId,
    setSessionId,
    getMessages,
    onFinalLoaded
}) {
    let activeAbortController = null;
    let isStreaming = false;

    function autoResizeInput() {
        queryInput.style.height = 'auto';
        queryInput.style.height = Math.min(queryInput.scrollHeight, 180) + 'px';
    }

    function setStreamingState(streaming) {
        isStreaming = streaming;
        queryInput.disabled = streaming;
        if (streaming) {
            sendBtn.textContent = '■';
            sendBtn.classList.add('stop');
            sendBtn.title = 'Stop generating';
            sendBtn.setAttribute('aria-label', 'Stop generating');
        } else {
            sendBtn.textContent = '↑';
            sendBtn.classList.remove('stop');
            sendBtn.title = 'Send';
            sendBtn.setAttribute('aria-label', 'Send message');
        }
    }

    function appendUserRow(text) {
        const welcome = chatColumn.querySelector('.welcome-state');
        if (welcome) welcome.remove();
        const row = document.createElement('div');
        row.className = 'msg-row user';
        const bubble = document.createElement('div');
        bubble.className = 'user-bubble';
        bubble.textContent = text;
        row.appendChild(bubble);
        chatColumn.appendChild(row);
        chatScroll.scrollTop = chatScroll.scrollHeight;
    }

    function addThinkingStep(row, message, active = true) {
        const list = row.querySelector('.thinking-steps');
        const prev = list.querySelector('.thinking-step.active');
        if (prev) {
            prev.classList.remove('active');
            const icon = prev.querySelector('.step-icon');
            if (icon) icon.outerHTML = '<span class="step-icon">✓</span>';
        }
        const li = document.createElement('li');
        li.className = 'thinking-step' + (active ? ' active' : '');
        li.innerHTML = active
            ? '<span class="step-icon step-spinner"></span><span></span>'
            : '<span class="step-icon">✓</span><span></span>';
        li.querySelector('span:last-child').textContent = message;
        list.appendChild(li);
        chatScroll.scrollTop = chatScroll.scrollHeight;
    }

    function createAssistantRow(initialSteps = [], streaming = true) {
        const welcome = chatColumn.querySelector('.welcome-state');
        if (welcome) welcome.remove();
        const row = document.createElement('div');
        row.className = 'msg-row assistant';
        row.innerHTML = `
            <div class="assistant-avatar">⚖️</div>
            <div class="assistant-body">
                <details class="thinking-box" ${streaming ? 'open' : ''}>
                    <summary>
                        <span class="step-spinner"></span>
                        <span class="thinking-title">सोच्दैछु… (Thinking)</span>
                    </summary>
                    <ul class="thinking-steps"></ul>
                </details>
                <div class="answer-markdown"></div>
                <div class="msg-actions" hidden></div>
                <div class="sources-container"></div>
            </div>
        `;
        initialSteps.forEach(s => addThinkingStep(row, s, false));
        chatColumn.appendChild(row);
        chatScroll.scrollTop = chatScroll.scrollHeight;
        return row;
    }

    function finalizeThinking(row) {
        const details = row.querySelector('.thinking-box');
        if (!details) return;
        const list = details.querySelector('.thinking-steps');
        const prev = list.querySelector('.thinking-step.active');
        if (prev) {
            prev.classList.remove('active');
            const icon = prev.querySelector('.step-icon');
            if (icon) icon.outerHTML = '<span class="step-icon">✓</span>';
        }
        if (list.children.length === 0) {
            details.remove();
            return;
        }
        const summary = details.querySelector('summary');
        summary.textContent = `▸ प्रक्रिया (${list.children.length} steps)`;
        details.open = false;
    }

    function renderSources(row, citations) {
        const container = row.querySelector('.sources-container');
        container.innerHTML = '';
        if (!citations || !citations.length) return;

        const details = document.createElement('details');
        details.className = 'sources-panel';
        const summary = document.createElement('summary');
        summary.textContent = `Sources (${citations.length})`;
        details.appendChild(summary);

        const pills = document.createElement('div');
        pills.className = 'source-pills';

        const excerptBox = document.createElement('div');
        excerptBox.className = 'source-excerpt';
        excerptBox.hidden = true;

        citations.forEach((cite, idx) => {
            const pill = document.createElement('button');
            pill.type = 'button';
            pill.className = 'source-pill';
            pill.textContent = `[${idx + 1}] ${cite.label || cite.act_source}`;
            pill.setAttribute('aria-expanded', 'false');
            pill.addEventListener('click', () => {
                const wasOpen = pill.getAttribute('aria-expanded') === 'true';
                pills.querySelectorAll('.source-pill').forEach(p => p.setAttribute('aria-expanded', 'false'));
                if (wasOpen) {
                    excerptBox.hidden = true;
                    return;
                }
                pill.setAttribute('aria-expanded', 'true');
                excerptBox.hidden = false;
                excerptBox.innerHTML = '';

                const meta = document.createElement('div');
                meta.className = 'source-excerpt-meta';
                const left = document.createElement('strong');
                left.textContent = cite.label || '';
                const right = document.createElement('span');
                right.textContent = `Status: ${cite.status || 'active'}`;
                meta.appendChild(left);
                meta.appendChild(right);

                const body = document.createElement('div');
                body.textContent = cite.excerpt || 'No excerpt available.';
                excerptBox.appendChild(meta);
                excerptBox.appendChild(body);
            });
            pills.appendChild(pill);
        });

        details.appendChild(pills);
        details.appendChild(excerptBox);
        container.appendChild(details);
    }

    function finalizeAssistantRow(row, data, isLatest = true) {
        finalizeThinking(row);
        const mdEl = row.querySelector('.answer-markdown');
        mdEl.innerHTML = renderSafeMarkdown(data.answer || '');

        const actions = row.querySelector('.msg-actions');
        actions.innerHTML = '';
        actions.hidden = false;

        const copyBtn = document.createElement('button');
        copyBtn.type = 'button';
        copyBtn.className = 'action-pill';
        copyBtn.textContent = '📋 Copy';
        copyBtn.addEventListener('click', async () => {
            try {
                await navigator.clipboard.writeText(data.answer || '');
                copyBtn.textContent = '✓ Copied';
                setTimeout(() => (copyBtn.textContent = '📋 Copy'), 1500);
            } catch (_) {}
        });
        actions.appendChild(copyBtn);

        if (isLatest) {
            const regenBtn = document.createElement('button');
            regenBtn.type = 'button';
            regenBtn.className = 'action-pill';
            regenBtn.textContent = '↻ Regenerate';
            regenBtn.addEventListener('click', () => regenerateLast());
            actions.appendChild(regenBtn);
        }

        const citations = Array.isArray(data.citations)
            ? data.citations
            : (data.chunks || []).map(c => ({
                label: `${c.act_source || ''}${c.hierarchy?.sec ? ', दफा ' + c.hierarchy.sec : ''}`,
                excerpt: (c.text || '').slice(0, 280),
                status: c.status || 'active'
            }));
        renderSources(row, citations);
        chatScroll.scrollTop = chatScroll.scrollHeight;
    }

    function renderThread() {
        chatColumn.innerHTML = '';
        const messages = getMessages();
        if (!messages.length) {
            const frag = welcomeTemplate.content.cloneNode(true);
            const grid = frag.querySelector('.starter-grid');
            STARTERS.forEach(s => {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'starter-chip';
                btn.innerHTML = '<strong></strong><span></span>';
                btn.querySelector('strong').textContent = s.text;
                btn.querySelector('span').textContent = s.sub;
                btn.addEventListener('click', () => {
                    queryInput.value = s.text;
                    autoResizeInput();
                    sendQuery();
                });
                grid.appendChild(btn);
            });
            chatColumn.appendChild(frag);
            return;
        }

        messages.forEach((msg, idx) => {
            if (msg.role === 'user') {
                appendUserRow(msg.content);
            } else {
                const isLast = idx === messages.length - 1;
                const row = createAssistantRow(msg.steps || [], false);
                finalizeAssistantRow(row, {
                    answer: msg.content,
                    citations: msg.citations || [],
                    chunks: []
                }, isLast);
            }
        });
        chatScroll.scrollTop = chatScroll.scrollHeight;
    }

    async function runStreamRequest(query, regenerate = false, targetRow = null) {
        const sessionId = ensureSessionId();
        activeAbortController = new AbortController();
        setStreamingState(true);

        const botRow = targetRow || createAssistantRow([], true);
        let streamedText = '';

        try {
            await streamQuery({
                query,
                sessionId,
                regenerate,
                signal: activeAbortController.signal,
                onStep: (payload) => addThinkingStep(botRow, payload.message, true),
                onToken: (payload) => {
                    streamedText += payload.text || '';
                    botRow.querySelector('.answer-markdown').innerHTML = renderSafeMarkdown(streamedText);
                    chatScroll.scrollTop = chatScroll.scrollHeight;
                },
                onFinal: async (payload) => {
                    if (payload.session_id) setSessionId(payload.session_id);
                    finalizeAssistantRow(botRow, payload, true);
                    await onFinalLoaded(getSessionId());
                }
            });
        } catch (err) {
            finalizeThinking(botRow);
            if (err.name === 'AbortError') {
                if (!streamedText) {
                    botRow.querySelector('.answer-markdown').textContent = 'Generation stopped.';
                }
            } else {
                botRow.querySelector('.answer-markdown').textContent = `त्रुटि (Error): ${err.message}`;
            }
        } finally {
            activeAbortController = null;
            setStreamingState(false);
            queryInput.focus();
        }
    }

    function sendQuery() {
        if (isStreaming && activeAbortController) {
            activeAbortController.abort();
            return;
        }
        const query = queryInput.value.trim();
        if (!query) return;

        queryInput.value = '';
        autoResizeInput();
        chatColumn.querySelectorAll('.action-pill').forEach(b => {
            if (b.textContent.includes('Regenerate')) b.remove();
        });
        appendUserRow(query);
        runStreamRequest(query, false, null);
    }

    function regenerateLast() {
        if (isStreaming) return;
        const lastUser = [...getMessages()].reverse().find(m => m.role === 'user');
        if (!lastUser) return;
        const rows = chatColumn.querySelectorAll('.msg-row.assistant');
        const lastRow = rows[rows.length - 1];
        if (lastRow) lastRow.remove();
        runStreamRequest(lastUser.content, true, null);
    }

    function abortIfStreaming() {
        if (isStreaming && activeAbortController) activeAbortController.abort();
    }

    return {
        autoResizeInput,
        renderThread,
        sendQuery,
        abortIfStreaming,
        getIsStreaming: () => isStreaming
    };
}
