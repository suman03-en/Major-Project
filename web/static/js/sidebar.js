const THEME_KEY = 'bizzguide_theme';

export function initTheme(themeBtn) {
    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem(THEME_KEY, theme);
        themeBtn.textContent = theme === 'dark' ? '🌙 Dark' : '☀️ Light';
    }

    const savedTheme = localStorage.getItem(THEME_KEY)
        || (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
    applyTheme(savedTheme);

    themeBtn.addEventListener('click', () => {
        const cur = document.documentElement.getAttribute('data-theme');
        applyTheme(cur === 'dark' ? 'light' : 'dark');
    });
}

export function initSidebarToggle({ sidebar, mobileBackdrop, openBtn, closeBtn }) {
    function toggleSidebar() {
        if (window.innerWidth <= 768) {
            const open = sidebar.classList.toggle('mobile-open');
            mobileBackdrop.classList.toggle('visible', open);
        } else {
            sidebar.classList.toggle('collapsed');
        }
    }

    function closeMobileSidebar() {
        if (window.innerWidth <= 768) {
            sidebar.classList.remove('mobile-open');
            mobileBackdrop.classList.remove('visible');
        }
    }

    openBtn.addEventListener('click', toggleSidebar);
    closeBtn.addEventListener('click', toggleSidebar);
    mobileBackdrop.addEventListener('click', () => {
        sidebar.classList.remove('mobile-open');
        mobileBackdrop.classList.remove('visible');
    });

    return { closeMobileSidebar };
}

export function renderConversationList({
    convList,
    items,
    activeSessionId,
    onSelect,
    onRename,
    onDelete
}) {
    convList.innerHTML = '';
    if (!items.length) {
        const empty = document.createElement('div');
        empty.className = 'conv-group-label';
        empty.textContent = 'No conversations yet';
        convList.appendChild(empty);
        return;
    }

    const todayStr = new Date().toDateString();
    const todayItems = items.filter(i => new Date(i.updated_at).toDateString() === todayStr);
    const olderItems = items.filter(i => new Date(i.updated_at).toDateString() !== todayStr);

    function appendGroup(label, list) {
        if (!list.length) return;
        const lbl = document.createElement('div');
        lbl.className = 'conv-group-label';
        lbl.textContent = label;
        convList.appendChild(lbl);

        list.forEach(item => {
            const row = document.createElement('div');
            row.className = 'conv-item' + (item.session_id === activeSessionId ? ' active' : '');

            const titleSpan = document.createElement('span');
            titleSpan.className = 'conv-title';
            titleSpan.textContent = item.title || 'New Chat';
            titleSpan.title = item.title || 'New Chat';
            row.appendChild(titleSpan);

            const actions = document.createElement('div');
            actions.className = 'conv-actions';

            const renameBtn = document.createElement('button');
            renameBtn.type = 'button';
            renameBtn.className = 'icon-btn';
            renameBtn.textContent = '✎';
            renameBtn.title = 'Rename';
            renameBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                onRename(item);
            });

            const delBtn = document.createElement('button');
            delBtn.type = 'button';
            delBtn.className = 'icon-btn danger';
            delBtn.textContent = '🗑';
            delBtn.title = 'Delete';
            delBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                onDelete(item);
            });

            actions.appendChild(renameBtn);
            actions.appendChild(delBtn);
            row.appendChild(actions);

            row.addEventListener('click', () => onSelect(item.session_id));
            convList.appendChild(row);
        });
    }

    appendGroup('Today', todayItems);
    appendGroup('Previous', olderItems);
}
