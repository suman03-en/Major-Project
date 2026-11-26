export async function fetchConversations() {
    const res = await fetch('/conversations');
    if (!res.ok) {
        throw new Error(`Failed to list conversations: ${res.status}`);
    }
    const data = await res.json();
    return data.conversations || [];
}

export async function fetchConversation(sessionId) {
    const res = await fetch(`/conversations/${encodeURIComponent(sessionId)}`);
    if (!res.ok) return null;
    return res.json();
}

export async function renameConversation(sessionId, title) {
    const res = await fetch(`/conversations/${encodeURIComponent(sessionId)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title })
    });
    return res.ok;
}

export async function deleteConversation(sessionId) {
    const res = await fetch(`/conversations/${encodeURIComponent(sessionId)}`, {
        method: 'DELETE'
    });
    return res.ok;
}

export async function streamQuery({
    query,
    sessionId,
    regenerate = false,
    signal,
    onStep,
    onToken,
    onFinal
}) {
    const res = await fetch('/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal,
        body: JSON.stringify({
            query,
            session_id: sessionId,
            top_k: 5,
            regenerate
        })
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';

        for (const rawEvent of events) {
            let eventName = 'message';
            let dataStr = '';
            for (const line of rawEvent.split('\n')) {
                if (line.startsWith('event: ')) eventName = line.slice(7).trim();
                else if (line.startsWith('data: ')) dataStr += line.slice(6);
            }
            if (!dataStr) continue;
            const payload = JSON.parse(dataStr);

            if (eventName === 'step' && onStep) {
                onStep(payload);
            } else if (eventName === 'token' && onToken) {
                onToken(payload);
            } else if (eventName === 'final' && onFinal) {
                await onFinal(payload);
            } else if (eventName === 'error') {
                throw new Error(payload.detail || 'Unknown server error');
            }
        }
    }
}
