(function () {
    "use strict";

    const app = document.getElementById("report-app");
    if (!app) return;

    const EVIDENCE_LIMIT_KEY = "regcheck.report.evidenceLimit";
    const DEFAULT_EVIDENCE_LIMIT = 2;
    const EVIDENCE_LIMIT_OPTIONS = [2, 4, 6];

    function loadEvidenceLimit() {
        const parsed = parseInt(window.localStorage ? window.localStorage.getItem(EVIDENCE_LIMIT_KEY) : "", 10);
        return EVIDENCE_LIMIT_OPTIONS.includes(parsed) ? parsed : DEFAULT_EVIDENCE_LIMIT;
    }

    const state = {
        taskId: app.dataset.taskId || "",
        demoSrc: app.dataset.demoSrc || "",
        items: [],
        manifest: null,
        manifestUnavailable: false,
        evidenceStatus: null,
        evidenceError: null,
        view: "decision",         // "decision" | "documents"
        activeIndex: 0,
        activeQuoteId: null,      // active highlighted quote in the documents view
        evidenceLimit: loadEvidenceLimit(),
        renderDataCache: new Map(),
        lastWorkflowStatus: null,
        logSeen: "",
        pollHandle: null,
        decisionResizersDone: false,
        // Per-source-panel viewer state in the Evidence view (persists across
        // re-renders): mode "page" (PDF images) or "text", zoom factor, search query.
        docPanelUI: {
            reg: { mode: null, zoom: 1, query: "" },
            ppr: { mode: null, zoom: 1, query: "" },
        },
    };

    const els = {
        viewDecision: document.getElementById("view-decision"),
        viewDocuments: document.getElementById("view-documents"),
        list: document.getElementById("dimension-list"),
        count: document.getElementById("dimension-count"),
        detail: document.getElementById("dimension-detail"),
        empty: document.getElementById("report-empty-state"),
        quotesPane: document.getElementById("quotes-pane"),
        viewToggle: document.getElementById("report-view-toggle"),
        statusPill: document.getElementById("report-status-pill"),
        statusText: document.getElementById("report-status-text"),
        copyLink: document.getElementById("copy-report-link-btn"),
        copyLinkLabel: document.getElementById("copy-report-link-label"),
        csv: document.getElementById("download-report-csv-btn"),
        toast: document.getElementById("report-toast"),
        log: document.getElementById("report-log"),
        logList: document.getElementById("report-log-list"),
        logProgress: document.getElementById("report-log-progress"),
    };

    const prefersReducedMotion = window.matchMedia
        ? window.matchMedia("(prefers-reduced-motion: reduce)").matches
        : false;

    /* ── helpers ─────────────────────────────────────────────────────────── */

    function escapeHtml(value) {
        return (value === null || value === undefined ? "" : String(value))
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function judgementInfo(value) {
        const normalized = (value || "").toString().trim().toLowerCase();
        if (normalized === "yes") return { label: "Deviation", tone: "flag" };
        if (normalized === "no") return { label: "No deviation", tone: "ok" };
        return { label: "Missing", tone: "warn" };
    }

    function parseQuotes(quotes) {
        if (!quotes) return [];
        const parts = [];
        const regex = /\[((?:PREREG|PAPER)_(\d+)),\s*relevance_score=([0-9.]+)\]\s*([\s\S]*?)(?=(\[(?:PREREG|PAPER)_[0-9]{4,}[^\]]*\])|$)/g;
        let match;
        while ((match = regex.exec(quotes)) !== null) {
            const chunk = (match[0] || "").trim();
            if (!chunk) continue;
            const id = match[1] || "";
            const num = parseInt(match[2], 10);
            const score = parseFloat(match[3]);
            const text = (match[4] || "").trim();
            parts.push({
                id,
                num: Number.isFinite(num) ? num : 0,
                score: Number.isFinite(score) ? score : 0,
                text,
                raw: chunk,
            });
        }
        if (parts.length > 0) return parts;
        return quotes
            .split(/\n\s*\n/)
            .map((text, index) => ({
                id: `QUOTE_${index + 1}`,
                num: index + 1,
                score: 0,
                text: text.trim(),
                raw: text.trim(),
            }))
            .filter((item) => item.text);
    }

    function sortedQuotes(quotes, limit = "all") {
        const sorted = [...quotes].sort((a, b) => {
            const scoreDiff = (b.score || 0) - (a.score || 0);
            if (scoreDiff !== 0) return scoreDiff;
            return (a.num || 0) - (b.num || 0);
        });
        if (limit === "all") return sorted;
        return sorted.slice(0, limit);
    }

    function evidenceLimitLabel(limit) {
        return limit === "all" ? "All chunks" : `Top ${limit}`;
    }

    function persistEvidenceLimit(limit) {
        state.evidenceLimit = limit;
        if (window.localStorage) {
            window.localStorage.setItem(EVIDENCE_LIMIT_KEY, String(limit));
        }
    }

    function currentItem() {
        if (!state.items.length) return null;
        const index = Math.min(Math.max(state.activeIndex, 0), state.items.length - 1);
        state.activeIndex = index;
        return state.items[index];
    }

    function quotesForRole(item, role) {
        const raw = role === "reg" ? item.registration_content_quotes : item.paper_content_quotes;
        return sortedQuotes(parseQuotes(raw || ""), state.evidenceLimit)
            .map((q) => Object.assign({}, q, { doc: role }));
    }

    function manifestSources() {
        return (state.manifest && state.manifest.sources) || {};
    }

    function chunkForQuote(quote) {
        if (!state.manifest || !state.manifest.chunks) return null;
        return state.manifest.chunks[quote.id] || null;
    }

    function firstLocation(chunk, kind) {
        if (!chunk || !Array.isArray(chunk.locations)) return null;
        return chunk.locations.find((location) => location.kind === kind) || null;
    }

    function isRegistrationSource(sourceId) {
        if (!sourceId || !state.manifest || !state.manifest.chunks) return false;
        return Object.values(state.manifest.chunks)
            .some((c) => c.source_id === sourceId && /^PREREG/i.test(c.id || ""));
    }

    function roleSourceId(item, role) {
        const quotes = parseQuotes(role === "reg" ? item.registration_content_quotes : item.paper_content_quotes);
        for (const q of quotes) {
            const chunk = chunkForQuote(q);
            if (chunk && chunk.source_id) return chunk.source_id;
        }
        const ids = Object.keys(manifestSources());
        for (const id of ids) {
            if (role === "reg" ? isRegistrationSource(id) : !isRegistrationSource(id)) return id;
        }
        return null;
    }

    function quoteHasSource(quote) {
        const chunk = chunkForQuote(quote);
        return !state.manifestUnavailable && !!(chunk && chunk.source_id);
    }

    /* ── status pill / log / toast ───────────────────────────────────────── */

    function setStatus(stateValue, detailText, counts) {
        if (!els.statusPill) return;
        // Once complete, drop the status pill entirely — the report itself is the
        // signal. Keep it for processing/failed states.
        els.statusPill.classList.toggle("d-none", stateValue === "SUCCESS");
        const running = stateValue !== "SUCCESS" && stateValue !== "FAILURE";
        els.statusPill.classList.toggle("is-running", running);
        els.statusPill.classList.toggle("is-done", stateValue === "SUCCESS");
        els.statusPill.classList.toggle("is-error", stateValue === "FAILURE");
        const hasCounts = counts && Number.isFinite(counts.processed) && Number.isFinite(counts.total) && counts.total > 0;
        if (els.statusText) {
            els.statusText.textContent =
                stateValue === "SUCCESS" ? "Complete" :
                stateValue === "FAILURE" ? "Processing failed" :
                hasCounts ? `Processing · ${counts.processed}/${counts.total}` : "Processing";
        }
        if (detailText && detailText !== state.lastWorkflowStatus) {
            state.lastWorkflowStatus = detailText;
            els.statusPill.title = detailText;
        }
    }

    function appendLog(text, counts) {
        if (els.logProgress && counts && Number.isFinite(counts.processed) && Number.isFinite(counts.total) && counts.total > 0) {
            els.logProgress.textContent = `${counts.processed}/${counts.total} dimensions`;
        }
        if (!text || !els.logList || text === state.logSeen) return;
        state.logSeen = text;
        const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
        els.logList.innerHTML = `<li><time>${escapeHtml(time)}</time><span>${escapeHtml(text)}</span></li>`;
        if (els.log) els.log.classList.remove("d-none");
    }

    function showToast(message) {
        if (!els.toast) return;
        els.toast.textContent = message;
        els.toast.classList.add("is-visible");
        window.clearTimeout(showToast._handle);
        showToast._handle = window.setTimeout(() => {
            els.toast.classList.remove("is-visible");
        }, 1800);
    }

    /* ── resizable columns ───────────────────────────────────────────────── */

    function installResizers(container, panels, storageKey, mins, defaults) {
        if (!container || panels.length < 2) return;
        let stored = null;
        try { stored = JSON.parse(window.localStorage.getItem(storageKey) || "null"); } catch (_e) { stored = null; }
        panels.forEach((panel, i) => {
            const grow = (Array.isArray(stored) && stored[i]) || defaults[i] || 1;
            panel.dataset.grow = defaults[i] || 1;
            panel.style.flex = `${grow} 1 0`;
            panel.style.minWidth = (mins[i] || 200) + "px";
        });
        const persist = () => {
            const grows = panels.map((p) => parseFloat(p.style.flexGrow) || 1);
            try { window.localStorage.setItem(storageKey, JSON.stringify(grows)); } catch (_e) { /* ignore */ }
        };
        for (let i = 0; i < panels.length - 1; i += 1) {
            const splitter = document.createElement("div");
            splitter.className = "report-splitter";
            splitter.setAttribute("role", "separator");
            splitter.setAttribute("aria-orientation", "vertical");
            splitter.setAttribute("tabindex", "0");
            splitter.setAttribute("aria-label", "Drag to resize panels");
            panels[i].after(splitter);
            attachSplitter(splitter, panels[i], panels[i + 1], persist);
        }
    }

    function attachSplitter(splitter, left, right, persist) {
        const reset = () => {
            left.style.flex = `${left.dataset.grow || 1} 1 0`;
            right.style.flex = `${right.dataset.grow || 1} 1 0`;
            persist();
        };
        splitter.addEventListener("dblclick", reset);
        splitter.addEventListener("pointerdown", (event) => {
            if (getComputedStyle(splitter).display === "none") return;
            event.preventDefault();
            splitter.classList.add("is-active");
            document.body.classList.add("is-col-resizing");
            const startX = event.clientX;
            const leftW = left.getBoundingClientRect().width;
            const rightW = right.getBoundingClientRect().width;
            const total = leftW + rightW;
            const totalGrow = (parseFloat(left.style.flexGrow) || 1) + (parseFloat(right.style.flexGrow) || 1);
            const leftMin = parseFloat(left.style.minWidth) || 120;
            const rightMin = parseFloat(right.style.minWidth) || 120;
            const move = (ev) => {
                let nl = leftW + (ev.clientX - startX);
                nl = Math.max(leftMin, Math.min(nl, total - rightMin));
                const nr = total - nl;
                left.style.flex = `${(nl / total) * totalGrow} 1 0`;
                right.style.flex = `${(nr / total) * totalGrow} 1 0`;
            };
            const up = () => {
                document.removeEventListener("pointermove", move);
                document.removeEventListener("pointerup", up);
                splitter.classList.remove("is-active");
                document.body.classList.remove("is-col-resizing");
                persist();
            };
            document.addEventListener("pointermove", move);
            document.addEventListener("pointerup", up);
        });
    }

    /* ── top-level render ────────────────────────────────────────────────── */

    function render() {
        const hasItems = state.items.length > 0;
        const inDocuments = state.view === "documents" && hasItems;
        if (els.viewDecision) els.viewDecision.classList.toggle("d-none", inDocuments);
        if (els.viewDocuments) els.viewDocuments.classList.toggle("d-none", !inDocuments);

        // Overview ⇄ Comparison toggle: only meaningful once the report has loaded.
        if (els.viewToggle) {
            els.viewToggle.classList.toggle("d-none", !hasItems);
            els.viewToggle.querySelectorAll("[data-view]").forEach((btn) => {
                const target = btn.dataset.view === "documents" ? "documents" : "decision";
                btn.classList.toggle("is-active", target === state.view);
            });
        }

        renderDimensionList();
        renderDimensionDetail();
        renderSummariesPane();
        if (inDocuments) renderDocumentsView();

        if (!state.decisionResizersDone && els.viewDecision) {
            const panels = [
                els.viewDecision.querySelector(".report-dimension-rail"),
                els.viewDecision.querySelector(".report-detail"),
                els.viewDecision.querySelector(".report-quotes"),
            ].filter(Boolean);
            if (panels.length === 3) {
                installResizers(els.viewDecision, panels, "regcheck.report.decisionCols",
                    [150, 280, 320], [1, 2, 2.5]);
                state.decisionResizersDone = true;
            }
        }
    }

    /* ── decision view: dimensions rail ──────────────────────────────────── */

    function renderDimensionList() {
        if (!els.list) return;
        if (els.count) els.count.textContent = String(state.items.length);
        els.list.innerHTML = "";
        state.items.forEach((item, index) => {
            const info = judgementInfo(item.deviation_judgement);
            const button = document.createElement("button");
            button.type = "button";
            button.className = `dimension-card ${index === state.activeIndex ? "is-active" : ""}`;
            button.innerHTML = `
                <span class="dimension-card__title">${escapeHtml(item.dimension || `Dimension ${index + 1}`)}</span>
                <span class="dimension-card__meta">
                    <span class="judgement-chip judgement-chip--${info.tone}">${info.label}</span>
                </span>
            `;
            button.addEventListener("click", () => {
                state.activeIndex = index;
                state.view = "decision";
                state.activeQuoteId = null;
                render();
                const detail = document.querySelector(".report-detail");
                if (detail) detail.scrollTop = 0;
                if (els.quotesPane) els.quotesPane.scrollTop = 0;
            });
            els.list.appendChild(button);
        });
    }

    /* ── decision view: middle detail (decision + rationale + summaries) ──── */

    function renderDimensionDetail() {
        if (!els.detail || !els.empty) return;
        if (!state.items.length) {
            els.detail.classList.add("d-none");
            els.empty.classList.remove("d-none");
            return;
        }
        els.empty.classList.add("d-none");
        els.detail.classList.remove("d-none");
        const item = currentItem() || {};
        const info = judgementInfo(item.deviation_judgement);
        els.detail.innerHTML = `
            <div class="detail-head">
                <h2>${escapeHtml(item.dimension || "Dimension")}</h2>
            </div>
            <div class="verdict-banner verdict-banner--${info.tone}">
                <div class="verdict-banner__head">
                    <span class="verdict-banner__label">${info.label}</span>
                </div>
                <p class="verdict-banner__body">${linkifyQuoteRefs(item.deviation_information || "No deviation information found.")}</p>
            </div>
        `;
        // Quote references (PREREG_0001 / PAPER_0001) jump to the Evidence view.
        bindQuoteRefs(els.detail);
    }

    // Turn PREREG_0001 / PAPER_0001 tokens in the deviation text into buttons
    // that open the Evidence view with that quote highlighted.
    function linkifyQuoteRefs(text) {
        return escapeHtml(text || "").replace(
            /\b((?:PREREG|PAPER)_\d+)\b/g,
            (id) => `<button type="button" class="quote-ref" data-quote-ref="${id}">${id}</button>`
        );
    }

    function openQuoteRef(id) {
        if (!id) return;
        const item = currentItem();
        if (!item) return;
        const role = /^PREREG/i.test(id) ? "reg" : "ppr";
        const all = parseQuotes(role === "reg" ? item.registration_content_quotes : item.paper_content_quotes);
        if (!all.some((q) => q.id === id)) return;   // unknown reference — ignore
        // Make sure the referenced quote is within the limited set shown in Evidence.
        if (!sortedQuotes(all, state.evidenceLimit).some((q) => q.id === id)) {
            persistEvidenceLimit("all");
        }
        state.view = "documents";
        state.activeQuoteId = id;
        render();   // documents view scrolls the active quote into view on render
    }

    function bindQuoteRefs(scope) {
        if (!scope) return;
        scope.querySelectorAll(".quote-ref").forEach((btn) => {
            btn.addEventListener("click", () => openQuoteRef(btn.dataset.quoteRef));
        });
    }

    /* ── decision view: right pane (quotes list) ─────────────────────────── */

    function limitControlHtml() {
        return `
            <div class="evidence-limit-control" role="group" aria-label="Evidence chunks shown">
                ${EVIDENCE_LIMIT_OPTIONS.map((option) => `
                    <button type="button" class="evidence-limit-btn ${option === state.evidenceLimit ? "is-active" : ""}" data-evidence-limit="${option}">
                        ${escapeHtml(option === "all" ? "All" : String(option))}
                    </button>
                `).join("")}
            </div>`;
    }

    function bindLimitControl(scope, rerender) {
        scope.querySelectorAll("[data-evidence-limit]").forEach((button) => {
            button.addEventListener("click", () => {
                const raw = button.dataset.evidenceLimit;
                const next = raw === "all" ? "all" : parseInt(raw, 10);
                if (!EVIDENCE_LIMIT_OPTIONS.includes(next)) return;
                persistEvidenceLimit(next);
                rerender();
            });
        });
    }

    // Overview right pane: registration + paper summaries (evidence quotes now
    // live only on the Evidence view).
    function renderSummariesPane() {
        if (!els.quotesPane) return;
        if (!state.items.length) {
            els.quotesPane.innerHTML = "";
            return;
        }
        const item = currentItem() || {};
        els.quotesPane.innerHTML = `
            <div class="summary-grid summary-grid--pane">
                <div class="summary-box">
                    <p class="section-title evidence-column__title--reg">Registration summary</p>
                    <p>${linkifyQuoteRefs(item.registration_content_summary || "No summary available.")}</p>
                </div>
                <div class="summary-box">
                    <p class="section-title evidence-column__title--paper">Paper summary</p>
                    <p>${linkifyQuoteRefs(item.paper_content_summary || "No summary available.")}</p>
                </div>
            </div>
        `;
        bindQuoteRefs(els.quotesPane);
    }

    function renderQuoteList(container, quotes, role, opts) {
        if (!container) return;
        opts = opts || {};
        container.innerHTML = "";
        if (!quotes.length) {
            container.innerHTML = `<div class="evidence-empty">No evidence found</div>`;
            return;
        }
        quotes.forEach((quote) => {
            const chunk = chunkForQuote(quote);
            const hasScore = Number.isFinite(quote.score) && quote.score > 0;
            const active = opts.activeId && quote.id === opts.activeId;
            const button = document.createElement("button");
            button.type = "button";
            button.className = `evidence-card evidence-card--${role} ${active ? "is-active" : ""}`;
            button.dataset.quote = quote.id;
            button.innerHTML = `
                <span class="evidence-card__head">
                    <span class="evidence-card__id">${escapeHtml(quote.id || "Evidence")}</span>
                    <span class="evidence-card__score" title="Relevance score">${hasScore ? quote.score.toFixed(2) : "&mdash;"}</span>
                </span>
                <span class="evidence-card__text">${escapeHtml((chunk && chunk.text) || quote.text || quote.raw || "")}</span>
                ${opts.compact ? "" : `<span class="evidence-card__cue">Open in documents &rsaquo;</span>`}
            `;
            button.addEventListener("click", () => (opts.onPick ? opts.onPick(quote) : openDocuments(quote)));
            container.appendChild(button);
        });
    }

    /* ── documents view (step 2): registration | quotes | paper ──────────── */

    function openDocuments(quote) {
        state.view = "documents";
        state.activeQuoteId = quote ? quote.id : null;
        render();
    }

    function backToDecision() {
        state.view = "decision";
        render();
    }

    function combinedQuotes(item) {
        return quotesForRole(item, "reg").concat(quotesForRole(item, "ppr"));
    }

    function renderDocumentsView() {
        if (!els.viewDocuments) return;
        const item = currentItem();
        if (!item) return;
        const info = judgementInfo(item.deviation_judgement);
        const regQuotes = quotesForRole(item, "reg");
        const pprQuotes = quotesForRole(item, "ppr");
        const all = regQuotes.concat(pprQuotes);
        if (!all.some((q) => q.id === state.activeQuoteId)) {
            state.activeQuoteId = all.length ? all[0].id : null;
        }

        els.viewDocuments.innerHTML = `
            <div class="docs-bar">
                <button type="button" class="docs-back" data-back>
                    <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M15 18l-6-6 6-6"/></svg>
                    Back to overview
                </button>
                <span class="docs-bar__dim">${escapeHtml(item.dimension || "Dimension")}</span>
                <span class="judgement-chip judgement-chip--${info.tone}">${info.label}</span>
            </div>
            <div class="docs-body">
                <section class="docs-panel docs-panel--doc">
                    <div class="docs-panel__head">
                        <span class="docs-tag docs-tag--reg">Registration</span>
                        <div class="docs-tools" id="docs-reg-tools"></div>
                    </div>
                    <div class="docs-panel__scroll" id="docs-reg-scroll"><div class="source-placeholder">Loading…</div></div>
                </section>
                <section class="docs-panel docs-panel--quotes">
                    <div class="docs-panel__head">
                        <span class="section-title">Quotes</span>
                        ${limitControlHtml()}
                    </div>
                    <div class="docs-panel__scroll" id="docs-quotes"></div>
                </section>
                <section class="docs-panel docs-panel--doc">
                    <div class="docs-panel__head">
                        <span class="docs-tag docs-tag--ppr">Paper</span>
                        <div class="docs-tools" id="docs-ppr-tools"></div>
                    </div>
                    <div class="docs-panel__scroll" id="docs-ppr-scroll"><div class="source-placeholder">Loading…</div></div>
                </section>
            </div>
        `;

        const back = els.viewDocuments.querySelector("[data-back]");
        if (back) back.addEventListener("click", backToDecision);
        bindLimitControl(els.viewDocuments, renderDocumentsView);

        const body = els.viewDocuments.querySelector(".docs-body");
        const panels = [...els.viewDocuments.querySelectorAll(".docs-panel")];
        installResizers(body, panels, "regcheck.report.docsCols", [220, 200, 220], [2, 1.3, 2]);

        renderMiddleQuotes(regQuotes, pprQuotes);
        Promise.all([
            renderDocPanel(item, "reg", regQuotes),
            renderDocPanel(item, "ppr", pprQuotes),
        ]).then(() => {
            refreshActiveHighlight();
            markUnlocatedQuotes();
            scrollActiveIntoView();
        });
    }

    function renderMiddleQuotes(regQuotes, pprQuotes) {
        const container = els.viewDocuments.querySelector("#docs-quotes");
        if (!container) return;
        container.innerHTML = `
            <div class="evidence-column">
                <p class="evidence-column__title evidence-column__title--reg">Registration</p>
                <div class="evidence-list" id="docs-reg-quotes"></div>
            </div>
            <div class="evidence-column">
                <p class="evidence-column__title evidence-column__title--paper">Paper</p>
                <div class="evidence-list" id="docs-ppr-quotes"></div>
            </div>
        `;
        const pick = (quote) => setActiveQuote(quote.id);
        renderQuoteList(document.getElementById("docs-reg-quotes"), regQuotes, "reg", { compact: true, onPick: pick, activeId: state.activeQuoteId });
        renderQuoteList(document.getElementById("docs-ppr-quotes"), pprQuotes, "ppr", { compact: true, onPick: pick, activeId: state.activeQuoteId });
    }

    function bindQuoteClicks(scroll) {
        scroll.querySelectorAll("[data-quote]").forEach((node) => {
            node.addEventListener("click", () => setActiveQuote(node.dataset.quote));
        });
    }

    function applyDocZoom(scroll, zoom) {
        if (scroll) scroll.style.setProperty("--doc-zoom", String(zoom || 1));
    }

    async function renderDocPanel(item, role, quotes) {
        const scroll = els.viewDocuments.querySelector(`#docs-${role}-scroll`);
        const tools = els.viewDocuments.querySelector(`#docs-${role}-tools`);
        if (!scroll) return;
        const ui = state.docPanelUI[role] || (state.docPanelUI[role] = { mode: null, zoom: 1, query: "" });
        const sourceId = roleSourceId(item, role);
        const source = sourceId ? manifestSources()[sourceId] : null;
        if (tools) tools.innerHTML = "";
        if (state.manifestUnavailable || !source) {
            scroll.innerHTML = `<div class="source-placeholder">Source document unavailable.</div>`;
            return;
        }

        const renderData = await getRenderData(source.id);
        const text = (renderData && renderData.text) || "";
        const canPdf = source.kind === "pdf" && source.render_mode === "pdf" && !!source.page_url_template;
        const mode = canPdf ? (ui.mode || "page") : "text";
        const pageCount = source.page_count || (source.pages || []).length || 1;

        const renderText = () => {
            scroll.innerHTML = `<div class="source-text source-text--doc ${role === "reg" ? "is-reg" : ""}">${buildTextDoc(text, quotes)}</div>`;
            scroll.dataset.mode = "text";
        };

        if (mode === "page") {
            scroll.innerHTML = buildPdfDoc(source, quotes, role, renderData);
            scroll.dataset.mode = "page";
            // (ii) Fallback: if the server can't render page images, swap to text.
            let fellBack = false;
            scroll.querySelectorAll(".pdf-stage img").forEach((img) => {
                img.addEventListener("error", () => {
                    if (fellBack || !text) return;
                    fellBack = true;
                    ui.mode = "text";
                    refreshDocPanel(role);
                });
            });
        } else {
            renderText();
        }

        applyDocZoom(scroll, ui.zoom);
        bindQuoteClicks(scroll);

        // (iii) Per-panel viewer tools: zoom, page nav (page mode), search, Page/Text toggle.
        if (tools) {
            tools.innerHTML = buildDocToolbar(role, canPdf, scroll.dataset.mode, pageCount);
            wireDocToolbar(role, scroll, pageCount);
        }
        if (scroll.dataset.mode === "page") setupPageNav(role, scroll, pageCount);
        if (ui.query) applyDocSearch(role);
    }

    async function refreshDocPanel(role) {
        const item = currentItem();
        if (!item) return;
        await renderDocPanel(item, role, quotesForRole(item, role));
        refreshActiveHighlight();
        markUnlocatedQuotes();
        scrollActiveIntoView();
    }

    function buildDocToolbar(role, canPdf, mode, pageCount) {
        const zoom = `<div class="docs-tool-group">
                <button type="button" class="docs-tool-btn" data-doc-zoom="out" title="Zoom out" aria-label="Zoom out">&minus;</button>
                <button type="button" class="docs-tool-btn" data-doc-zoom="in" title="Zoom in" aria-label="Zoom in">+</button>
            </div>`;
        const page = mode === "page" ? `<div class="docs-tool-group">
                <button type="button" class="docs-tool-btn" data-doc-page="prev" title="Previous page" aria-label="Previous page">&lsaquo;</button>
                <span class="docs-page__ind">1 / ${pageCount}</span>
                <button type="button" class="docs-tool-btn" data-doc-page="next" title="Next page" aria-label="Next page">&rsaquo;</button>
            </div>` : "";
        const search = `<div class="docs-tool-group docs-search">
                <input type="search" class="docs-search__input" placeholder="Search" aria-label="Search document">
                <span class="docs-search__count"></span>
                <button type="button" class="docs-tool-btn" data-doc-search="prev" title="Previous match" aria-label="Previous match">&lsaquo;</button>
                <button type="button" class="docs-tool-btn" data-doc-search="next" title="Next match" aria-label="Next match">&rsaquo;</button>
            </div>`;
        const toggle = canPdf ? `<div class="docs-tool-group docs-mode">
                <button type="button" class="docs-tool-btn ${mode === "page" ? "is-active" : ""}" data-doc-mode="page">Page</button>
                <button type="button" class="docs-tool-btn ${mode === "text" ? "is-active" : ""}" data-doc-mode="text">Text</button>
            </div>` : "";
        return zoom + page + search + toggle;
    }

    function wireDocToolbar(role, scroll, pageCount) {
        const tools = els.viewDocuments.querySelector(`#docs-${role}-tools`);
        if (!tools) return;
        const ui = state.docPanelUI[role];
        tools.querySelectorAll("[data-doc-zoom]").forEach((b) => {
            b.addEventListener("click", () => {
                const dir = b.dataset.docZoom === "in" ? 1 : -1;
                ui.zoom = Math.min(2.5, Math.max(0.6, Math.round((ui.zoom + dir * 0.15) * 100) / 100));
                applyDocZoom(scroll, ui.zoom);
            });
        });
        tools.querySelectorAll("[data-doc-page]").forEach((b) => {
            b.addEventListener("click", () => pageStep(role, b.dataset.docPage === "next" ? 1 : -1));
        });
        tools.querySelectorAll("[data-doc-mode]").forEach((b) => {
            b.addEventListener("click", () => { ui.mode = b.dataset.docMode; refreshDocPanel(role); });
        });
        const input = tools.querySelector(".docs-search__input");
        if (input) {
            input.value = ui.query || "";
            input.addEventListener("input", async () => {
                ui.query = input.value;
                if (scroll.dataset.mode !== "text") {
                    ui.mode = "text";   // search needs a text layer to highlight
                    await refreshDocPanel(role);
                    const next = els.viewDocuments.querySelector(`#docs-${role}-tools .docs-search__input`);
                    if (next) { next.focus(); next.setSelectionRange(next.value.length, next.value.length); }
                } else {
                    applyDocSearch(role);
                }
            });
        }
        tools.querySelectorAll("[data-doc-search]").forEach((b) => {
            b.addEventListener("click", () => docSearchStep(role, b.dataset.docSearch === "next" ? 1 : -1));
        });
    }

    function setupPageNav(role, scroll, pageCount) {
        const ind = els.viewDocuments.querySelector(`#docs-${role}-tools .docs-page__ind`);
        const stages = [...scroll.querySelectorAll(".pdf-stage")];
        if (!stages.length) return;
        const update = () => {
            const cr = scroll.getBoundingClientRect();
            let cur = 1;
            stages.forEach((st) => {
                if (st.getBoundingClientRect().top - cr.top <= cr.height * 0.4) {
                    cur = parseInt(st.dataset.page, 10) || cur;
                }
            });
            state.docPanelUI[role]._page = cur;
            if (ind) ind.textContent = `${cur} / ${pageCount}`;
        };
        let raf = null;
        scroll.addEventListener("scroll", () => {
            if (raf) return;
            raf = window.requestAnimationFrame(() => { raf = null; update(); });
        });
        update();
    }

    function pageStep(role, dir) {
        const scroll = els.viewDocuments.querySelector(`#docs-${role}-scroll`);
        if (!scroll) return;
        const stages = [...scroll.querySelectorAll(".pdf-stage")];
        const cur = state.docPanelUI[role]._page || 1;
        const target = Math.min(Math.max(cur + dir, 1), stages.length);
        const st = stages.find((s) => parseInt(s.dataset.page, 10) === target);
        if (st) st.scrollIntoView({ behavior: prefersReducedMotion ? "auto" : "smooth", block: "start" });
    }

    /* ── document search (Custom Highlight API, text mode) ───────────────────── */
    function findTextRanges(root, query) {
        const ranges = [];
        const needle = (query || "").toLowerCase();
        if (!needle) return ranges;
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
        let node;
        while ((node = walker.nextNode())) {
            const hay = node.nodeValue.toLowerCase();
            let from = 0;
            let idx = hay.indexOf(needle, from);
            while (idx !== -1) {
                const range = document.createRange();
                range.setStart(node, idx);
                range.setEnd(node, idx + needle.length);
                ranges.push(range);
                from = idx + needle.length;
                idx = hay.indexOf(needle, from);
            }
        }
        return ranges;
    }

    function scrollRangeIntoView(scroll, range) {
        const rect = range.getBoundingClientRect();
        const cr = scroll.getBoundingClientRect();
        if (!rect.height) return;
        const top = scroll.scrollTop + (rect.top - cr.top) - (scroll.clientHeight / 2) + (rect.height / 2);
        scroll.scrollTo({ top, behavior: prefersReducedMotion ? "auto" : "smooth" });
    }

    function highlightSupported() {
        return !!(window.CSS && CSS.highlights && typeof Highlight !== "undefined");
    }

    function applyDocSearch(role) {
        const ui = state.docPanelUI[role];
        const scroll = els.viewDocuments.querySelector(`#docs-${role}-scroll`);
        const countEl = els.viewDocuments.querySelector(`#docs-${role}-tools .docs-search__count`);
        if (!scroll) return;
        const sourceText = scroll.querySelector(".source-text");
        const ranges = sourceText ? findTextRanges(sourceText, (ui.query || "").trim()) : [];
        ui._matches = ranges;
        ui._matchIndex = 0;
        if (highlightSupported()) {
            CSS.highlights.delete(`rc-search-${role}`);
            CSS.highlights.delete(`rc-search-active-${role}`);
            if (ranges.length) CSS.highlights.set(`rc-search-${role}`, new Highlight(...ranges));
        }
        if (countEl) countEl.textContent = ranges.length ? `1/${ranges.length}` : ((ui.query || "").trim() ? "0" : "");
        if (ranges.length) {
            setSearchActive(role, 0);
            scrollRangeIntoView(scroll, ranges[0]);
        }
    }

    function setSearchActive(role, index) {
        const ui = state.docPanelUI[role];
        if (!ui._matches || !ui._matches.length) return;
        ui._matchIndex = (index + ui._matches.length) % ui._matches.length;
        const countEl = els.viewDocuments.querySelector(`#docs-${role}-tools .docs-search__count`);
        if (countEl) countEl.textContent = `${ui._matchIndex + 1}/${ui._matches.length}`;
        if (highlightSupported()) {
            CSS.highlights.set(`rc-search-active-${role}`, new Highlight(ui._matches[ui._matchIndex]));
        }
    }

    function docSearchStep(role, dir) {
        const ui = state.docPanelUI[role];
        if (!ui._matches || !ui._matches.length) return;
        setSearchActive(role, ui._matchIndex + dir);
        const scroll = els.viewDocuments.querySelector(`#docs-${role}-scroll`);
        scrollRangeIntoView(scroll, ui._matches[ui._matchIndex]);
    }

    function setActiveQuote(id) {
        state.activeQuoteId = id;
        refreshActiveHighlight();
        // Reflect selection in the middle quote cards.
        els.viewDocuments.querySelectorAll("#docs-quotes .evidence-card").forEach((card) => {
            card.classList.toggle("is-active", card.dataset.quote === id);
        });
        scrollActiveIntoView();
    }

    function refreshActiveHighlight() {
        els.viewDocuments.querySelectorAll(".doc-mark, .pdf-highlight, .pdf-page-flag").forEach((node) => {
            node.classList.toggle("is-active", node.dataset.quote === state.activeQuoteId);
        });
    }

    // Flag middle-pane quote cards whose quote could not be anchored anywhere
    // in either document, so a dead click is explained rather than silent.
    function markUnlocatedQuotes() {
        els.viewDocuments.querySelectorAll("#docs-quotes .evidence-card").forEach((card) => {
            const located = !!document.getElementById(`docmark-${card.dataset.quote}`);
            card.classList.toggle("is-unlocated", !located);
            if (!located) card.title = "This quote could not be located in the rendered document.";
            else card.removeAttribute("title");
        });
    }

    function scrollActiveIntoView() {
        if (!state.activeQuoteId) return;
        const el = document.getElementById(`docmark-${state.activeQuoteId}`);
        if (!el) return;
        const container = el.closest(".docs-panel__scroll");
        if (!container) return;
        window.requestAnimationFrame(() => {
            const cr = container.getBoundingClientRect();
            const er = el.getBoundingClientRect();
            const top = container.scrollTop + (er.top - cr.top) - (container.clientHeight / 2) + (er.height / 2);
            container.scrollTo({ top, behavior: prefersReducedMotion ? "auto" : "smooth" });
        });
    }

    function highlightId(quote) {
        return `docmark-${quote.id}`;
    }

    // Resolve where a quote sits in the document text. Stored chunk offsets
    // are used only after validation; otherwise fall back to the locator's
    // tiered fuzzy matching (exact → normalized → dehyphenated → seeded).
    function resolveTextSpan(docText, quote) {
        if (!docText) return null;
        const locator = window.RegCheckLocator || null;
        const chunk = chunkForQuote(quote);
        const expected = (chunk && chunk.text) || quote.text || "";
        if (chunk) {
            const loc = firstLocation(chunk, "text") || firstLocation(chunk, "json");
            if (loc && Number.isFinite(loc.start) && Number.isFinite(loc.end) && loc.end > loc.start) {
                if (!locator || locator.spanMatches(docText, loc.start, loc.end, expected)) {
                    return { start: loc.start, end: loc.end, approximate: false };
                }
            }
        }
        if (!expected) return null;
        if (locator) {
            const hit = locator.locateQuote(docText, expected);
            if (hit) return hit;
            if (quote.text && quote.text !== expected) {
                return locator.locateQuote(docText, quote.text);
            }
            return null;
        }
        const index = docText.indexOf(expected);
        return index >= 0 ? { start: index, end: index + expected.length, approximate: false } : null;
    }

    function buildTextDoc(text, quotes) {
        if (!text) return `<div class="source-placeholder">Document text unavailable.</div>`;
        const ranges = [];
        quotes.forEach((quote) => {
            const span = resolveTextSpan(text, quote);
            if (span) ranges.push({ start: span.start, end: span.end, approximate: span.approximate, quote });
        });
        ranges.sort((a, b) => a.start - b.start);
        const clean = [];
        let lastEnd = -1;
        ranges.forEach((r) => { if (r.start >= lastEnd) { clean.push(r); lastEnd = r.end; } });

        let out = "";
        let cursor = 0;
        clean.forEach((r) => {
            out += escapeHtml(text.slice(cursor, r.start));
            const active = r.quote.id === state.activeQuoteId ? " is-active" : "";
            const approx = r.approximate ? ` title="Approximate location"` : "";
            out += `<mark class="doc-mark${active}" data-quote="${escapeHtml(r.quote.id)}" id="${highlightId(r.quote)}"${approx}>`
                + escapeHtml(text.slice(r.start, r.end))
                + `</mark>`;
            cursor = r.end;
        });
        out += escapeHtml(text.slice(cursor));
        return out;
    }

    function buildPdfDoc(source, quotes, role, renderData) {
        const pageCount = source.page_count || (source.pages || []).length || 1;
        const template = source.page_url_template || "";
        const hlBase = "pdf-highlight" + (role === "reg" ? " pdf-highlight--reg" : "");
        const locator = window.RegCheckLocator || null;
        const renderText = (renderData && renderData.text) || "";
        const pages = (renderData && renderData.pages) || source.pages || [];

        // Quotes without rect locations fall back to a clickable page-level
        // flag: find the quote in the render text, map the offset to a page.
        const flagsByPage = new Map();
        quotes.forEach((quote) => {
            const chunk = chunkForQuote(quote);
            const hasRects = chunk && Array.isArray(chunk.locations)
                && chunk.locations.some((l) => l.kind === "pdf" && (l.rects || []).length);
            if (hasRects) return;
            let offset = null;
            const span = renderText ? resolveTextSpan(renderText, quote) : null;
            if (span) offset = span.start;
            else if (chunk) {
                const loc = firstLocation(chunk, "text") || firstLocation(chunk, "json");
                if (loc && Number.isFinite(loc.start)) offset = loc.start;
            }
            if (offset === null || !locator) return;
            const pageHit = locator.pageForOffset(pages, offset, renderText.length || undefined);
            if (!pageHit) return;
            if (!flagsByPage.has(pageHit.page)) flagsByPage.set(pageHit.page, []);
            flagsByPage.get(pageHit.page).push({ quote, approximate: pageHit.approximate });
        });

        let html = "";
        for (let page = 1; page <= pageCount; page += 1) {
            const src = template.replace("{page_number}", String(page));
            const pageInfo = (source.pages || []).find((p) => p.page_number === page) || {};
            const highlights = quotes.flatMap((quote) => {
                const chunk = chunkForQuote(quote);
                const locs = chunk && Array.isArray(chunk.locations)
                    ? chunk.locations.filter((l) => l.kind === "pdf" && l.page === page)
                    : [];
                const active = quote.id === state.activeQuoteId ? " is-active" : "";
                return locs.flatMap((loc) => {
                    const width = loc.page_width || pageInfo.width || 1;
                    const height = loc.page_height || pageInfo.height || 1;
                    return (loc.rects || []).map((rect, ri) => {
                        const left = (rect.x0 / width) * 100;
                        const top = (rect.y0 / height) * 100;
                        const w = ((rect.x1 - rect.x0) / width) * 100;
                        const h = ((rect.y1 - rect.y0) / height) * 100;
                        const anchor = ri === 0 ? `id="${highlightId(quote)}"` : "";
                        return `<span class="${hlBase}${active}" ${anchor} data-quote="${escapeHtml(quote.id)}" style="left:${left}%;top:${top}%;width:${w}%;height:${h}%;"></span>`;
                    });
                });
            }).join("");
            const flags = (flagsByPage.get(page) || []).map((entry, index) => {
                const active = entry.quote.id === state.activeQuoteId ? " is-active" : "";
                const title = entry.approximate
                    ? "Quote is on (approximately) this page — exact position unavailable"
                    : "Quote is on this page — exact position unavailable";
                return `<button type="button" class="pdf-page-flag pdf-page-flag--${role}${active}"
                    id="${highlightId(entry.quote)}" data-quote="${escapeHtml(entry.quote.id)}"
                    style="top: calc(0.6rem + ${index * 1.9}rem)" title="${title}">${escapeHtml(entry.quote.id)}</button>`;
            }).join("");
            html += `<div class="pdf-stage" data-page="${page}"><img src="${escapeHtml(src)}" alt="Page ${page}" loading="lazy">${highlights}${flags}</div>`;
        }
        return html;
    }

    function stepQuote(dir) {
        const item = currentItem();
        if (!item) return;
        const all = combinedQuotes(item);
        const idx = all.findIndex((q) => q.id === state.activeQuoteId);
        const next = idx + dir;
        if (next < 0 || next >= all.length) return;
        setActiveQuote(all[next].id);
    }

    /* ── render-data fetch ───────────────────────────────────────────────── */

    async function getRenderData(sourceId) {
        if (state.renderDataCache.has(sourceId)) return state.renderDataCache.get(sourceId);
        try {
            const response = await fetch(`/report/${state.taskId}/sources/${sourceId}/render-data`);
            if (!response.ok) return null;
            const data = await response.json();
            state.renderDataCache.set(sourceId, data);
            return data;
        } catch (_error) {
            return null;
        }
    }

    /* ── CSV / share ─────────────────────────────────────────────────────── */

    function csvEscape(value) {
        const text = (value === null || value === undefined ? "" : value).toString();
        return `"${text.replace(/"/g, "\"\"")}"`;
    }

    function quotesPlain(quotes) {
        return sortedQuotes(parseQuotes(quotes || ""), "all")
            .map((quote) => {
                const score = quote.score ? `, relevance_score=${quote.score.toFixed(3)}` : "";
                return `[${quote.id}${score}] ${(quote.text || quote.raw || "").trim()}`;
            })
            .join("\n\n");
    }

    function downloadCsv() {
        const rows = [[
            "Dimension",
            "Preregistration Direct Quotes",
            "Preregistration Summary",
            "Paper Direct Quotes",
            "Paper Summary",
            "Deviation Information",
            "Deviation Judgement",
        ]];
        state.items.forEach((item) => {
            rows.push([
                item.dimension || "",
                quotesPlain(item.registration_content_quotes),
                item.registration_content_summary || "",
                quotesPlain(item.paper_content_quotes),
                item.paper_content_summary || "",
                item.deviation_information || "",
                item.deviation_judgement || "",
            ]);
        });
        const csvText = rows.map((row) => row.map(csvEscape).join(",")).join("\n");
        const blob = new Blob(["﻿", csvText], { type: "text/csv;charset=utf-8;" });
        const link = document.createElement("a");
        link.download = "regcheck-report.csv";
        link.href = window.URL.createObjectURL(blob);
        link.style.display = "none";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    async function copyReportLink() {
        let copied = false;
        try {
            await navigator.clipboard.writeText(window.location.href);
            copied = true;
        } catch (_error) {
            try {
                const temp = document.createElement("textarea");
                temp.value = window.location.href;
                temp.style.position = "fixed";
                temp.style.opacity = "0";
                document.body.appendChild(temp);
                temp.select();
                copied = document.execCommand("copy");
                document.body.removeChild(temp);
            } catch (_fallbackError) {
                copied = false;
            }
        }
        if (els.copyLinkLabel) els.copyLinkLabel.textContent = copied ? "Copied" : "Copy failed";
        if (els.copyLink) els.copyLink.classList.toggle("is-copied", copied);
        showToast(copied ? "Report link copied to clipboard" : "Could not copy link");
        window.setTimeout(() => {
            if (els.copyLinkLabel) els.copyLinkLabel.textContent = "Share";
            if (els.copyLink) els.copyLink.classList.remove("is-copied");
        }, 1800);
    }

    /* ── loading: poll / demo ────────────────────────────────────────────── */

    async function loadManifest() {
        try {
            const response = await fetch(`/report/${state.taskId}/manifest`);
            if (!response.ok) {
                state.manifest = null;
                state.manifestUnavailable = true;
                return;
            }
            state.manifest = await response.json();
            state.manifestUnavailable = false;
        } catch (_error) {
            state.manifest = state.manifest || null;
        }
    }

    async function pollTaskStatus() {
        try {
            const response = await fetch(`/task_status/${state.taskId}`);
            const data = await response.json();
            state.evidenceStatus = data.evidence_status || null;
            state.evidenceError = data.evidence_error || null;
            const counts = { processed: data.processed_dimensions, total: data.total_dimensions };
            setStatus(data.state, data.status || data.state || "", counts);
            appendLog(data.status || "", counts);
            if (data.result && Array.isArray(data.result.items)) {
                state.items = data.result.items;
                if (state.activeIndex >= state.items.length) state.activeIndex = 0;
            }
            if (data.evidence_available === false) {
                state.manifest = null;
                state.manifestUnavailable = data.state === "SUCCESS";
            } else {
                await loadManifest();
            }
            render();
            if (data.state === "SUCCESS" && els.log) els.log.classList.add("d-none");
            if (data.state !== "SUCCESS" && data.state !== "FAILURE") {
                state.pollHandle = window.setTimeout(pollTaskStatus, 3000);
            }
        } catch (_error) {
            setStatus("FAILURE", "Failed to fetch report status");
        }
    }

    async function loadDemo() {
        try {
            const response = await fetch(state.demoSrc);
            if (!response.ok) throw new Error("demo fixture unavailable");
            const data = await response.json();
            state.items = Array.isArray(data.items) ? data.items : [];
            state.manifest = data.manifest || null;
            state.manifestUnavailable = !state.manifest;
            if (data.render_data) {
                Object.keys(data.render_data).forEach((id) => {
                    state.renderDataCache.set(id, data.render_data[id]);
                });
            }
            setStatus("SUCCESS", "Sample report");
            // The demo intentionally keeps a "Sample report" badge (setStatus hides
            // the pill on SUCCESS for real reports).
            if (els.statusPill) els.statusPill.classList.remove("d-none");
            if (els.statusText) els.statusText.textContent = "Sample report";
            render();
        } catch (_error) {
            setStatus("FAILURE", "Could not load the sample report");
        }
    }

    /* ── keyboard ────────────────────────────────────────────────────────── */

    function onKey(event) {
        if (event.target && (event.target.tagName === "INPUT" || event.target.tagName === "TEXTAREA")) return;
        if (state.view !== "documents") return;
        if (event.key === "Escape") { event.preventDefault(); backToDecision(); }
        else if (event.key === "ArrowLeft") { event.preventDefault(); stepQuote(-1); }
        else if (event.key === "ArrowRight") { event.preventDefault(); stepQuote(1); }
    }

    /* ── wire up ─────────────────────────────────────────────────────────── */

    // The owner page may intercept the Share button (e.g. to open the restricted
    // share dialog instead of copying the link). If the hook returns true it has
    // handled the click; otherwise we fall back to copying the link.
    window.regcheckCopyReportLink = copyReportLink;
    if (els.copyLink) {
        els.copyLink.addEventListener("click", () => {
            const hook = window.regcheckShareClick;
            if (typeof hook === "function" && hook() === true) return;
            copyReportLink();
        });
    }
    if (els.csv) els.csv.addEventListener("click", downloadCsv);
    if (els.viewToggle) {
        els.viewToggle.querySelectorAll("[data-view]").forEach((btn) => {
            btn.addEventListener("click", () => {
                const target = btn.dataset.view === "documents" ? "documents" : "decision";
                if (target === state.view || !state.items.length) return;
                state.view = target;   // Comparison view picks a default quote in renderDocumentsView
                render();
            });
        });
    }
    window.addEventListener("keydown", onKey);
    window.addEventListener("beforeunload", () => {
        if (state.pollHandle) window.clearTimeout(state.pollHandle);
    });

    if (state.demoSrc) loadDemo();
    else pollTaskStatus();
})();
