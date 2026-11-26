export function renderSafeMarkdown(md) {
    if (typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
        return DOMPurify.sanitize(marked.parse(md || ''));
    }
    const div = document.createElement('div');
    div.textContent = md || '';
    return div.innerHTML.replace(/\n/g, '<br>');
}
