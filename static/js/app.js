/* FPViber - frontend logic */

const API = {
    status: () => fetch("/api/status").then(r => r.json()),
    listSessions: () => fetch("/api/sessions").then(r => r.json()),
    createSession: () =>
        fetch("/api/sessions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
        }).then(r => r.json()),
    getSession: id => fetch(`/api/sessions/${id}`).then(r => r.json()),
    renameSession: (id, title) =>
        fetch(`/api/sessions/${id}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title }),
        }).then(r => r.json()),
    deleteSession: id =>
        fetch(`/api/sessions/${id}`, { method: "DELETE" }).then(r => r.json()),
    sendMessage: (id, content) =>
        fetch(`/api/sessions/${id}/messages`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content }),
        }).then(async r => {
            const data = await r.json();
            if (!r.ok) throw new Error(data.detail || data.error || "Request failed");
            return data;
        }),
};

const state = {
    sessions: [],
    activeSessionId: null,
    messages: [],
    engineReady: false,
    isSending: false,
};

const els = {
    app: document.querySelector(".app"),
    sidebar: document.getElementById("sidebar"),
    sessionList: document.getElementById("sessionList"),
    newChatBtn: document.getElementById("newChatBtn"),
    newChatBtnLarge: document.getElementById("newChatBtnLarge"),
    toggleSidebar: document.getElementById("toggleSidebar"),
    renameBtn: document.getElementById("renameBtn"),
    deleteBtn: document.getElementById("deleteBtn"),
    messages: document.getElementById("messages"),
    emptyState: document.getElementById("emptyState"),
    conversationTitle: document.getElementById("conversationTitle"),
    conversationMeta: document.getElementById("conversationMeta"),
    form: document.getElementById("chatForm"),
    input: document.getElementById("chatInput"),
    sendBtn: document.getElementById("sendBtn"),
    statusDot: document.querySelector(".status__dot"),
    statusText: document.querySelector(".status__text"),
    suggestions: document.getElementById("suggestions"),
};

// ==========================================================
// Utilities
// ==========================================================
function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text ?? "";
    return div.innerHTML;
}

function svgIcon(id, className = "") {
    return `<svg class="${className}" aria-hidden="true"><use href="#${id}"></use></svg>`;
}

function formatDate(iso) {
    if (!iso) return "";
    try {
        const d = new Date(iso);
        const now = new Date();
        const isToday = d.toDateString() === now.toDateString();
        if (isToday) {
            return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        }
        return d.toLocaleDateString([], { month: "short", day: "numeric" });
    } catch {
        return "";
    }
}

function autoresize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 180) + "px";
}

let toastEl = null;
let toastTimer = null;
function toast(message, { error = false } = {}) {
    if (!toastEl) {
        toastEl = document.createElement("div");
        toastEl.className = "toast";
        document.body.appendChild(toastEl);
    }
    toastEl.textContent = message;
    toastEl.classList.toggle("toast--error", error);
    toastEl.classList.add("is-visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toastEl.classList.remove("is-visible"), 3500);
}

// ==========================================================
// Engine status polling
// ==========================================================
async function pollEngineStatus() {
    try {
        const s = await API.status();
        if (s.error) {
            els.statusDot.dataset.state = "error";
            els.statusText.textContent = `Error: ${s.error.message}`;
            return;
        }
        if (s.ready) {
            state.engineReady = true;
            els.statusDot.dataset.state = "ready";
            const docCount = s.documents || 0;
            els.statusText.textContent =
                `Online - ${s.chunks} chunks / ${docCount} docs`;
            els.input.disabled = false;
            els.sendBtn.disabled = false;
            els.input.placeholder = "Ask anything about FPV...";
            return;
        }
        els.statusDot.dataset.state = "loading";
        let label = humanizeStatus(s.status);
        if (s.status === "embedding_documents" && s.progress?.total) {
            label = `Embedding ${s.progress.current}/${s.progress.total}...`;
        }
        els.statusText.textContent = label;
        setTimeout(pollEngineStatus, 1500);
    } catch (e) {
        els.statusDot.dataset.state = "error";
        els.statusText.textContent = "Server unreachable";
        setTimeout(pollEngineStatus, 3000);
    }
}

function humanizeStatus(key) {
    const map = {
        not_initialised: "Spinning up motors...",
        loading_documents: "Loading documents...",
        chunking_documents: "Slicing knowledge base...",
        embedding_documents: "Computing embeddings...",
        building_index: "Building FAISS index...",
        ready: "Online",
    };
    return map[key] || "Spinning up motors...";
}

// ==========================================================
// Sessions
// ==========================================================
async function loadSessions() {
    const data = await API.listSessions();
    state.sessions = data.sessions || [];
    renderSessions();
}

function renderSessions() {
    els.sessionList.innerHTML = "";
    if (state.sessions.length === 0) {
        const empty = document.createElement("div");
        empty.className = "session-list__empty";
        empty.textContent = "No conversations yet";
        els.sessionList.appendChild(empty);
        return;
    }

    for (const s of state.sessions) {
        const item = document.createElement("div");
        item.className = "session-item";
        if (s.id === state.activeSessionId) item.classList.add("is-active");
        item.dataset.id = s.id;
        item.innerHTML = `
            <span class="session-item__title">${escapeHtml(s.title)}</span>
            <span class="session-item__date">${formatDate(s.updated_at)}</span>
        `;
        item.addEventListener("click", () => selectSession(s.id));
        els.sessionList.appendChild(item);
    }
}

async function selectSession(id) {
    state.activeSessionId = id;
    const data = await API.getSession(id);
    state.messages = data.messages || [];
    els.conversationTitle.textContent = data.title || "Conversation";
    els.conversationMeta.textContent =
        state.messages.length > 0
            ? `${state.messages.length} message${state.messages.length === 1 ? "" : "s"}`
            : "No messages yet";
    els.renameBtn.disabled = false;
    els.deleteBtn.disabled = false;
    renderMessages();
    renderSessions();
}

async function newSession({ select = true } = {}) {
    const session = await API.createSession();
    state.sessions.unshift(session);
    renderSessions();
    if (select) {
        await selectSession(session.id);
        els.input.focus();
    }
    return session;
}

async function renameActiveSession() {
    if (!state.activeSessionId) return;
    const current = state.sessions.find(s => s.id === state.activeSessionId);
    const title = prompt("Rename conversation", current?.title || "");
    if (!title || title.trim() === "") return;
    const updated = await API.renameSession(state.activeSessionId, title.trim());
    const idx = state.sessions.findIndex(s => s.id === updated.id);
    if (idx !== -1) state.sessions[idx] = updated;
    els.conversationTitle.textContent = updated.title;
    renderSessions();
}

async function deleteActiveSession() {
    if (!state.activeSessionId) return;
    if (!confirm("Delete this conversation? This cannot be undone.")) return;
    const id = state.activeSessionId;
    await API.deleteSession(id);
    state.sessions = state.sessions.filter(s => s.id !== id);
    state.activeSessionId = null;
    state.messages = [];
    els.conversationTitle.textContent = "New conversation";
    els.conversationMeta.textContent = "";
    els.renameBtn.disabled = true;
    els.deleteBtn.disabled = true;
    renderSessions();
    renderMessages();
    toast("Conversation deleted");
}

// ==========================================================
// Messages
// ==========================================================
function renderMessages() {
    els.messages.innerHTML = "";

    if (state.messages.length === 0) {
        els.messages.appendChild(els.emptyState);
        return;
    }

    const inner = document.createElement("div");
    inner.className = "messages__inner";
    for (const m of state.messages) {
        inner.appendChild(renderMessage(m));
    }
    els.messages.appendChild(inner);
    requestAnimationFrame(() => {
        els.messages.scrollTop = els.messages.scrollHeight;
    });
}

function renderMessage(msg) {
    const wrap = document.createElement("div");
    wrap.className = `message message--${msg.role}`;

    const avatar = document.createElement("div");
    avatar.className = "message__avatar";
    if (msg.role === "user") {
        avatar.textContent = "YOU";
    } else {
        avatar.innerHTML = svgIcon("i-drone");
    }

    const body = document.createElement("div");
    body.className = "message__body";

    const role = document.createElement("div");
    role.className = "message__role";
    role.textContent = msg.role === "user" ? "You" : "FPViber";

    const content = document.createElement("div");
    content.className = "message__content";
    content.textContent = msg.content;

    body.appendChild(role);
    body.appendChild(content);

    if (msg.role === "assistant") {
        const sources = collectSources(msg);
        if (sources.length > 0) {
            body.appendChild(renderSources(sources));
        }
        if (Array.isArray(msg.context) && msg.context.length > 0) {
            body.appendChild(renderContext(msg.context));
        }
    }

    wrap.appendChild(avatar);
    wrap.appendChild(body);
    return wrap;
}

function collectSources(msg) {
    // Prefer the metadata.sources field; fall back to deduped context sources.
    if (msg.metadata && Array.isArray(msg.metadata.sources) && msg.metadata.sources.length > 0) {
        return msg.metadata.sources;
    }
    if (Array.isArray(msg.context)) {
        const seen = new Set();
        const out = [];
        for (const c of msg.context) {
            if (c.source && !seen.has(c.source)) {
                seen.add(c.source);
                out.push(c.source);
            }
        }
        return out;
    }
    return [];
}

function renderSources(sources) {
    const wrap = document.createElement("div");
    wrap.className = "sources";
    wrap.innerHTML = `<span class="sources__label">Sources</span>` +
        sources
            .map(
                s =>
                    `<span class="source-chip">${svgIcon("i-propeller")}${escapeHtml(s)}</span>`
            )
            .join("");
    return wrap;
}

function renderContext(chunks) {
    const details = document.createElement("details");
    details.className = "context";

    const summary = document.createElement("summary");
    summary.className = "context__summary";
    summary.textContent = `Retrieved chunks (${chunks.length})`;
    details.appendChild(summary);

    const body = document.createElement("div");
    body.className = "context__body";
    for (const c of chunks) {
        const item = document.createElement("div");
        item.className = "context__chunk";
        const meta = document.createElement("div");
        meta.className = "context__chunk-meta";
        meta.innerHTML = `
            <span>${escapeHtml(c.source || "unknown")}</span>
            <span>score ${typeof c.score === "number" ? c.score.toFixed(3) : "-"}</span>
        `;
        const text = document.createElement("div");
        text.textContent = c.text;
        item.appendChild(meta);
        item.appendChild(text);
        body.appendChild(item);
    }
    details.appendChild(body);
    return details;
}

function appendMessageEphemeral(msg) {
    let inner = els.messages.querySelector(".messages__inner");
    if (!inner) {
        els.messages.innerHTML = "";
        inner = document.createElement("div");
        inner.className = "messages__inner";
        els.messages.appendChild(inner);
    }
    inner.appendChild(renderMessage(msg));
    els.messages.scrollTop = els.messages.scrollHeight;
}

function appendDroneLoader() {
    let inner = els.messages.querySelector(".messages__inner");
    if (!inner) {
        els.messages.innerHTML = "";
        inner = document.createElement("div");
        inner.className = "messages__inner";
        els.messages.appendChild(inner);
    }
    const wrap = document.createElement("div");
    wrap.className = "message message--assistant";
    wrap.id = "droneLoader";
    wrap.innerHTML = `
        <div class="message__avatar">${svgIcon("i-drone")}</div>
        <div class="message__body">
            <div class="message__role">FPViber</div>
            <div class="message__content">
                <div class="drone-loader">
                    <div class="drone-loader__sky">
                        <svg class="drone-loader__drone" viewBox="0 0 64 64" aria-hidden="true">
                            <g fill="none" stroke="currentColor" stroke-width="2.5"
                               stroke-linecap="round" stroke-linejoin="round">
                                <line x1="14" y1="14" x2="50" y2="50"/>
                                <line x1="50" y1="14" x2="14" y2="50"/>
                                <rect x="24" y="24" width="16" height="16" rx="2"/>
                                <circle class="prop" cx="14" cy="14" r="8"/>
                                <circle class="prop" cx="50" cy="14" r="8"/>
                                <circle class="prop" cx="14" cy="50" r="8"/>
                                <circle class="prop" cx="50" cy="50" r="8"/>
                                <circle cx="32" cy="32" r="2" fill="currentColor"/>
                            </g>
                        </svg>
                    </div>
                    <span class="drone-loader__text">FPViber is flying</span>
                </div>
            </div>
        </div>
    `;
    inner.appendChild(wrap);
    els.messages.scrollTop = els.messages.scrollHeight;
}

function removeDroneLoader() {
    const t = document.getElementById("droneLoader");
    if (t) t.remove();
}

// ==========================================================
// Sending
// ==========================================================
async function sendMessage(content) {
    if (state.isSending) return;
    if (!state.engineReady) {
        toast("FPViber is still warming up", { error: true });
        return;
    }

    const text = content.trim();
    if (!text) return;

    if (!state.activeSessionId) {
        await newSession({ select: true });
    }

    state.isSending = true;
    els.sendBtn.disabled = true;
    els.input.disabled = true;

    const tempUser = {
        id: `tmp-${Date.now()}`,
        role: "user",
        content: text,
        created_at: new Date().toISOString(),
    };
    state.messages.push(tempUser);
    appendMessageEphemeral(tempUser);
    appendDroneLoader();

    els.input.value = "";
    autoresize(els.input);

    try {
        const result = await API.sendMessage(state.activeSessionId, text);
        removeDroneLoader();

        const userIdx = state.messages.findIndex(m => m.id === tempUser.id);
        if (userIdx !== -1) state.messages[userIdx] = result.user_message;
        state.messages.push(result.assistant_message);
        appendMessageEphemeral(result.assistant_message);

        await loadSessions();
        const updated = state.sessions.find(s => s.id === state.activeSessionId);
        if (updated) {
            els.conversationTitle.textContent = updated.title;
        }
        els.conversationMeta.textContent = `${state.messages.length} messages`;
    } catch (err) {
        removeDroneLoader();
        toast(err.message || "Failed to send message", { error: true });
    } finally {
        state.isSending = false;
        els.sendBtn.disabled = !state.engineReady;
        els.input.disabled = !state.engineReady;
        els.input.focus();
    }
}

// ==========================================================
// Wiring
// ==========================================================
els.form.addEventListener("submit", e => {
    e.preventDefault();
    sendMessage(els.input.value);
});

els.input.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage(els.input.value);
    }
});

els.input.addEventListener("input", () => autoresize(els.input));

els.newChatBtn.addEventListener("click", () => newSession());
els.newChatBtnLarge.addEventListener("click", () => newSession());
els.renameBtn.addEventListener("click", renameActiveSession);
els.deleteBtn.addEventListener("click", deleteActiveSession);

els.toggleSidebar.addEventListener("click", () => {
    els.app.classList.toggle("sidebar-collapsed");
    els.app.classList.toggle("sidebar-open");
});

els.suggestions?.addEventListener("click", e => {
    const btn = e.target.closest(".suggestion");
    if (!btn) return;
    const q = btn.dataset.q;
    els.input.value = q;
    autoresize(els.input);
    sendMessage(q);
});

// ==========================================================
// Boot
// ==========================================================
(async function boot() {
    pollEngineStatus();
    await loadSessions();
})();
