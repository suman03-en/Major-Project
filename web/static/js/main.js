import {
    deleteConversation,
    fetchConversation,
    fetchConversations,
    renameConversation
} from './api.js';
import { createChatController } from './chat.js';
import {
    initSidebarToggle,
    initTheme,
    renderConversationList
} from './sidebar.js';

const ACTIVE_KEY = 'bizzguide_active_session_id';

const sidebar = document.getElementById('sidebar');
const mobileBackdrop = document.getElementById('mobileBackdrop');
const convList = document.getElementById('convList');
const chatScroll = document.getElementById('chatScroll');
const chatColumn = document.getElementById('chatColumn');
const topbarTitle = document.getElementById('topbarTitle');
const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const themeBtn = document.getElementById('themeBtn');
const welcomeTemplate = document.getElementById('welcomeTemplate');

let activeSessionId = localStorage.getItem(ACTIVE_KEY) || '';
let activeMessages = [];

function setSessionId(id) {
    activeSessionId = id;
    localStorage.setItem(ACTIVE_KEY, activeSessionId);
}

function ensureSessionId() {
    if (!activeSessionId) {
        const next = (crypto.randomUUID && crypto.randomUUID()) || String(Date.now());
        setSessionId(next);
    }
    return activeSessionId;
}

initTheme(themeBtn);
const { closeMobileSidebar } = initSidebarToggle({
    sidebar,
    mobileBackdrop,
    openBtn: document.getElementById('openSidebarBtn'),
    closeBtn: document.getElementById('closeSidebarBtn')
});

const chat = createChatController({
    chatScroll,
    chatColumn,
    queryInput,
    sendBtn,
    welcomeTemplate,
    getSessionId: () => activeSessionId,
    ensureSessionId,
    setSessionId,
    getMessages: () => activeMessages,
    onFinalLoaded: (sid) => loadConversation(sid, true)
});

async function refreshConversations() {
    try {
        const items = await fetchConversations();
        renderConversationList({
            convList,
            items,
            activeSessionId,
            onSelect: (sid) => loadConversation(sid),
            onRename: async (item) => {
                const nextTitle = prompt('Rename conversation:', item.title || '');
                if (!nextTitle || !nextTitle.trim()) return;
                const ok = await renameConversation(item.session_id, nextTitle.trim());
                if (!ok) {
                    alert('Could not rename conversation.');
                    return;
                }
                if (item.session_id === activeSessionId) {
                    topbarTitle.textContent = nextTitle.trim();
                }
                refreshConversations();
            },
            onDelete: async (item) => {
                const ok = await deleteConversation(item.session_id);
                if (!ok) {
                    alert('Could not delete conversation.');
                    return;
                }
                if (item.session_id === activeSessionId) startNewChat();
                else refreshConversations();
            }
        });
    } catch (err) {
        console.error('Error refreshing conversations:', err);
    }
}

async function loadConversation(sessionId, force = false) {
    if (chat.getIsStreaming() && !force) return;
    try {
        const data = await fetchConversation(sessionId);
        if (!data) {
            startNewChat();
            return;
        }
        setSessionId(data.session_id);
        topbarTitle.textContent = data.title || 'New Chat';
        activeMessages = data.messages || [];
        chat.renderThread();
        refreshConversations();
        closeMobileSidebar();
    } catch (err) {
        console.error('Error loading conversation:', err);
    }
}

function startNewChat() {
    chat.abortIfStreaming();
    const next = (crypto.randomUUID && crypto.randomUUID()) || String(Date.now());
    setSessionId(next);
    activeMessages = [];
    topbarTitle.textContent = 'New Chat';
    chat.renderThread();
    refreshConversations();
    queryInput.focus();
}

document.getElementById('newChatBtn').addEventListener('click', startNewChat);
document.getElementById('quickNewBtn').addEventListener('click', startNewChat);
queryInput.addEventListener('input', chat.autoResizeInput);
sendBtn.addEventListener('click', chat.sendQuery);
queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chat.sendQuery();
    }
});

(async () => {
    await refreshConversations();
    if (activeSessionId) {
        await loadConversation(activeSessionId);
    } else {
        chat.renderThread();
    }
})();
