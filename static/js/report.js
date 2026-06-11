(function () {
    "use strict";

    const app = document.getElementById("report-app");
    if (!app) return;

    const TOPK_KEY = "regcheck.report.topK";
    const DIM_KEY = "regcheck.report.dimIndex";
    const DEFAULT_TOPK = 2;

    function loadTopK() {
        const parsed = parseInt(window.localStorage ? window.localStorage.getItem(TOPK_KEY) : "", 10);
        return parsed >= 1 && parsed <= 3 ? parsed : DEFAULT_TOPK;
    }
    function loadDimIndex() {
        const parsed = parseInt(window.localStorage ? window.localStorage.getItem(DIM_KEY) : "", 10);
        return Number.isFinite(parsed) && parsed >= 0 ? parsed : 0;
    }

    const state = {
        taskId: app.dataset.taskId || "",
        demoSrc: app.dataset.demoSrc || "",
        items: [],
        manifest: null,
        manifestUnavailable: false,
        evidenceStatus: null,
        evidenceError: null,
        view: "board",            // board | poster | trace
        dimIndex: loadDimIndex(),
        activeQuote: null,        // chunk id of the active quote in trace
        topK: loadTopK(),
        expanded: new Set(),      // expanded quote ids in the poster
        lastScrollSrc: "enter",   // enter | step | doc
        renderDataCache: new Map(),
        lastWorkflowStatus: null,
        logSeen: "",
        pollHandle: null,
        ready: false,
    };

    const els = {
        views: document.getElementById("report-views"),
        statusPill: document.getElementById("report-status-pill"),
        statusText: document.getElementById("report-status-text"),
        statusSummary: document.getElementById("report-status-summary"),
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

    const STATUS = {
        dev: { label: "Deviation", tone: "flag" },
        ok: { label: "No deviation", tone: "ok" },
        mis: { label: "Missing", tone: "warn" },
    };

    function statusOf(value) {
        const normalized = (value || "").toString().trim().toLowerCase();
        if (normalized === "yes") return "dev";
        if (normalized === "no") return "ok";
        return "mis";
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
            });
        }
        if (parts.length > 0) return parts;
        return quotes
            .split(/\n\s*\n/)
            .map((text, index) => ({ id: `QUOTE_${index + 1}`, num: index + 1, score: 0, text: text.trim() }))
            .filter((item) => item.text);
    }

    function sortByScore(a, b) {
        const diff = (b.score || 0) - (a.score || 0);
        if (diff !== 0) return diff;
        return (a.num || 0) - (b.num || 0);
    }

    function manifestChunk(id) {
        if (!state.manifest || !state.manifest.chunks) return null;
        return state.manifest.chunks[id] || null;
    }

    function chunkText(quote) {
        const chunk = manifestChunk(quote.id);
        return (chunk && chunk.text) || quote.text || "";
    }

    function locationOf(chunk, kinds) {
        if (!chunk || !Array.isArray(chunk.locations)) return null;
        return chunk.locations.find((location) => kinds.includes(location.kind)) || null;
    }

    // A source is the registration if its chunks carry PREREG ids.
    function isRegistrationSource(sourceId) {
        if (!sourceId || !state.manifest || !state.manifest.chunks) return false;
        return Object.values(state.manifest.chunks)
            .some((c) => c.source_id === sourceId && /^PREREG/i.test(c.id || ""));
    }

    function sourcesByRole() {
        const sources = (state.manifest && state.manifest.sources) || {};
        let reg = null, ppr = null;
        Object.values(sources).forEach((source) => {
            if (isRegistrationSource(source.id)) reg = reg || source.id;
            else ppr = ppr || source.id;
        });
        return { reg, ppr };
    }

    function roleSourceId(role, dim) {
        const list = role === "reg" ? dim.reg : dim.ppr;
        for (const q of list) {
            const chunk = manifestChunk(q.id);
            if (chunk && chunk.source_id) return chunk.source_id;
        }
        return sourcesByRole()[role];
    }

    // Build the normalized dimension model the UI consumes.
    function dimensions() {
        return state.items.map((item, index) => {
            const reg = parseQuotes(item.registration_content_quotes || "")
                .sort(sortByScore)
                .map((q, i) => Object.assign({}, q, { doc: "reg", label: `R${i + 1}` }));
            const ppr = parseQuotes(item.paper_content_quotes || "")
                .sort(sortByScore)
                .map((q, i) => Object.assign({}, q, { doc: "ppr", label: `P${i + 1}` }));
            return {
                index,
                id: item.dimension || `dimension-${index + 1}`,
                name: item.dimension || `Dimension ${index + 1}`,
                status: statusOf(item.deviation_judgement),
                rationale: item.deviation_information || "",
                summaryReg: item.registration_content_summary || "",
                summaryPpr: item.paper_content_summary || "",
                reg,
                ppr,
            };
        });
    }

    function countStatuses(dims) {
        const counts = { dev: 0, ok: 0, mis: 0 };
        dims.forEach((d) => { counts[d.status] += 1; });
        return counts;
    }

    function visibleQuotes(dim) {
        return dim.reg.slice(0, state.topK).concat(dim.ppr.slice(0, state.topK));
    }

    function quoteHasSource(quote) {
        const chunk = manifestChunk(quote.id);
        return !state.manifestUnavailable && !!(chunk && chunk.source_id);
    }

    /* ── status pill / summary / log ─────────────────────────────────────── */

    function setStatus(stateValue, detailText, counts) {
        if (!els.statusPill) return;
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

    function renderStatusSummary() {
        if (!els.statusSummary) return;
        if (!state.ready || !state.items.length) {
            els.statusSummary.innerHTML = "";
            return;
        }
        const counts = countStatuses(dimensions());
        const segments = [
            { tone: "flag", value: counts.dev, label: "deviation" },
            { tone: "ok", value: counts.ok, label: "none" },
            { tone: "warn", value: counts.mis, label: "missing" },
        ].filter((segment) => segment.value > 0);
        els.statusSummary.innerHTML = segments.map((segment) => `
            <span class="sf-summary__item">
                <span class="verdict-dot verdict-dot--${segment.tone}"></span>
                <strong>${segment.value}</strong> ${escapeHtml(segment.label)}
            </span>
        `).join("");
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

    /* ── navigation ──────────────────────────────────────────────────────── */

    function setView(view) {
        state.view = view;
        render();
    }
    function openDimension(index) {
        state.dimIndex = index;
        if (window.localStorage) window.localStorage.setItem(DIM_KEY, String(index));
        state.expanded = new Set();
        setView("poster");
    }
    function setDimIndex(index) {
        state.dimIndex = index;
        if (window.localStorage) window.localStorage.setItem(DIM_KEY, String(index));
        state.expanded = new Set();
        render();
    }
    function setTopK(k) {
        state.topK = k;
        if (window.localStorage) window.localStorage.setItem(TOPK_KEY, String(k));
        render();
    }
    function openTrace(quoteId) {
        state.activeQuote = quoteId;
        state.lastScrollSrc = "enter";
        setView("trace");
    }

    function currentDim(dims) {
        const list = dims || dimensions();
        if (!list.length) return null;
        const index = Math.min(Math.max(state.dimIndex, 0), list.length - 1);
        state.dimIndex = index;
        return list[index];
    }

    /* ── render shell ────────────────────────────────────────────────────── */

    function render() {
        renderStatusSummary();
        if (!els.views) return;
        const dims = dimensions();
        state.ready = dims.length > 0;

        if (!state.ready) {
            els.views.innerHTML = `
                <div class="sf-loading">
                    <span class="sf-loading__spinner" aria-hidden="true"></span>
                    <p>${escapeHtml(state.demoSrc ? "Loading sample report…" : "Preparing your report…")}</p>
                </div>`;
            return;
        }

        let html = "";
        if (state.view === "board") html = renderBoard(dims);
        else if (state.view === "poster") html = renderPoster(dims);
        else if (state.view === "trace") html = renderTrace(dims);
        els.views.innerHTML = html;
        if (!prefersReducedMotion) {
            els.views.firstElementChild && els.views.firstElementChild.classList.add("sf-enter");
        }

        if (state.view === "board") bindBoard();
        else if (state.view === "poster") bindPoster(dims);
        else if (state.view === "trace") bindTrace(dims);
    }

    /* ── level 0: scorecard ──────────────────────────────────────────────── */

    function renderBoard(dims) {
        const counts = countStatuses(dims);
        const total = dims.length;
        const headline = `${counts.dev} of ${total} dimension${total === 1 ? "" : "s"} ${counts.dev === 1 ? "deviates" : "deviate"} from the plan`;
        const sublineParts = [];
        if (counts.mis > 0) {
            sublineParts.push(`${counts.mis} dimension${counts.mis === 1 ? " was" : "s were"} not registered`);
        }
        sublineParts.push("click a tile to read the decision");
        const subline = sublineParts.join(" · ");

        const tiles = dims.map((d) => {
            const info = STATUS[d.status];
            const preview = d.rationale || "No rationale provided.";
            return `
                <button type="button" class="sf-tile sf-tile--${info.tone}" data-dim="${d.index}">
                    <span class="sf-tile__top">
                        <span class="verdict-dot verdict-dot--${info.tone}"></span>
                        <span class="sf-verdict sf-verdict--${info.tone}">${info.label}</span>
                        <span class="sf-tile__go" aria-hidden="true">&#8599;</span>
                    </span>
                    <span class="sf-tile__name">${escapeHtml(d.name)}</span>
                    <span class="sf-tile__preview">${escapeHtml(preview)}</span>
                </button>`;
        }).join("");

        return `
            <div class="sf-board">
                <h1 class="sf-board__headline">${escapeHtml(headline)}</h1>
                <p class="sf-board__subline">${escapeHtml(subline)}</p>
                <div class="sf-board__grid">
                    ${tiles}
                    <button type="button" class="sf-tile sf-tile--export" data-export>
                        <span class="sf-tile__export-label">Export CSV &#8595;</span>
                    </button>
                </div>
            </div>`;
    }

    function bindBoard() {
        els.views.querySelectorAll("[data-dim]").forEach((tile) => {
            tile.addEventListener("click", () => openDimension(parseInt(tile.dataset.dim, 10)));
        });
        const exportTile = els.views.querySelector("[data-export]");
        if (exportTile) exportTile.addEventListener("click", downloadCsv);
    }

    /* ── level 1: judgment poster ────────────────────────────────────────── */

    function quoteRow(quote, opts) {
        const open = state.expanded.has(quote.id);
        const score = quote.score ? quote.score.toFixed(2) : "—";
        const traceable = quoteHasSource(quote);
        const traceLink = open && traceable
            ? `<button type="button" class="sf-quote__trace sf-quote__trace--${quote.doc}" data-trace="${escapeHtml(quote.id)}">trace in document &rarr;</button>`
            : "";
        const body = open
            ? `<div class="sf-quote__open">
                    <p class="sf-quote__full">${escapeHtml(chunkText(quote))}</p>
                    ${traceLink}
               </div>`
            : `<p class="sf-quote__preview">${escapeHtml(chunkText(quote))}</p>`;
        return `
            <div class="sf-quote sf-quote--${quote.doc} ${open ? "is-open" : ""}" data-quote="${escapeHtml(quote.id)}">
                <div class="sf-quote__head">
                    <span class="sf-quote__label sf-quote__label--${quote.doc}">${escapeHtml(quote.label)}</span>
                    <span class="sf-quote__sim">${escapeHtml(score)}</span>
                    <span class="sf-quote__chevron" aria-hidden="true">${open ? "▾" : "▸"}</span>
                </div>
                ${body}
            </div>`;
    }

    function evidenceColumn(role, quotes) {
        const labelText = role === "reg" ? "Registration" : "Paper";
        const rows = quotes.length
            ? quotes.map((q) => quoteRow(q, { role })).join("")
            : `<p class="sf-evidence__empty">No retrieved quotes.</p>`;
        return `
            <div class="sf-evidence__col">
                <span class="sf-evidence__col-title sf-evidence__col-title--${role}">${labelText}</span>
                ${rows}
            </div>`;
    }

    function renderPoster(dims) {
        const dim = currentDim(dims);
        const info = STATUS[dim.status];
        const total = dims.length;
        const vis = visibleQuotes(dim);
        const firstTraceable = vis.find(quoteHasSource);
        const openDocs = firstTraceable
            ? `<button type="button" class="sf-poster__opendocs" data-trace="${escapeHtml(firstTraceable.id)}">open documents &rarr;</button>`
            : "";
        const topkButtons = [1, 2, 3].map((k) => `
            <button type="button" class="sf-topk__btn ${state.topK === k ? "is-active" : ""}" data-topk="${k}">${k}</button>
        `).join("");

        const prevDisabled = dim.index === 0 ? "disabled" : "";
        const nextDisabled = dim.index === total - 1 ? "disabled" : "";

        const dots = dims.map((d) => {
            const tone = STATUS[d.status].tone;
            return `<button type="button" class="sf-dot verdict-dot--${tone} ${d.index === dim.index ? "is-active" : ""}" data-dim="${d.index}" title="${escapeHtml(d.name)}" aria-label="${escapeHtml(d.name)}"></button>`;
        }).join("");

        return `
            <div class="sf-poster">
                <button type="button" class="sf-poster__arrow" data-step="-1" ${prevDisabled} aria-label="Previous dimension">‹</button>
                <div class="sf-poster__scroll">
                    <div class="sf-poster__col">
                        <div class="sf-poster__crumb">
                            <button type="button" class="sf-pill" data-board>‹ All dimensions</button>
                            <span class="sf-poster__count">Dimension ${dim.index + 1} of ${total}</span>
                        </div>
                        <h1 class="sf-poster__title">${escapeHtml(dim.name)}</h1>
                        <div class="sf-poster__verdict">
                            <span class="verdict-dot verdict-dot--${info.tone}"></span>
                            <span class="sf-verdict sf-verdict--${info.tone}">${info.label}</span>
                        </div>
                        <div class="sf-rationale sf-rationale--${info.tone}">
                            <p>${escapeHtml(dim.rationale || "No rationale provided.")}</p>
                        </div>
                        <div class="sf-poster__divider"></div>
                        <div class="sf-evidence__bar">
                            <span class="sf-evidence__label">Evidence</span>
                            <div class="sf-topk" role="group" aria-label="Quotes per document">${topkButtons}</div>
                            <span class="sf-evidence__caption">quotes per document, by similarity</span>
                            ${openDocs}
                        </div>
                        <div class="sf-evidence__grid">
                            ${evidenceColumn("reg", dim.reg.slice(0, state.topK))}
                            ${evidenceColumn("ppr", dim.ppr.slice(0, state.topK))}
                        </div>
                    </div>
                </div>
                <button type="button" class="sf-poster__arrow" data-step="1" ${nextDisabled} aria-label="Next dimension">›</button>
                <div class="sf-poster__dots">
                    ${dots}
                    <span class="sf-poster__hint">← → to move · esc for scorecard</span>
                </div>
            </div>`;
    }

    function bindPoster(dims) {
        const dim = currentDim(dims);
        const total = dims.length;
        els.views.querySelectorAll("[data-step]").forEach((btn) => {
            btn.addEventListener("click", () => {
                const dir = parseInt(btn.dataset.step, 10);
                const next = dim.index + dir;
                if (next >= 0 && next < total) setDimIndex(next);
            });
        });
        const board = els.views.querySelector("[data-board]");
        if (board) board.addEventListener("click", () => setView("board"));
        els.views.querySelectorAll("[data-topk]").forEach((btn) => {
            btn.addEventListener("click", () => setTopK(parseInt(btn.dataset.topk, 10)));
        });
        els.views.querySelectorAll(".sf-dot[data-dim]").forEach((dot) => {
            dot.addEventListener("click", () => setDimIndex(parseInt(dot.dataset.dim, 10)));
        });
        els.views.querySelectorAll(".sf-quote[data-quote]").forEach((row) => {
            row.addEventListener("click", (event) => {
                if (event.target.closest("[data-trace]")) return;
                const id = row.dataset.quote;
                if (state.expanded.has(id)) state.expanded.delete(id);
                else state.expanded.add(id);
                render();
            });
        });
        els.views.querySelectorAll("[data-trace]").forEach((link) => {
            link.addEventListener("click", (event) => {
                event.stopPropagation();
                openTrace(link.dataset.trace);
            });
        });
    }

    /* ── level 2: document trace ─────────────────────────────────────────── */

    function renderTrace(dims) {
        const dim = currentDim(dims);
        const info = STATUS[dim.status];
        const total = dims.length;
        const prevDisabled = dim.index === 0 ? "disabled" : "";
        const nextDisabled = dim.index === total - 1 ? "disabled" : "";

        return `
            <div class="sf-trace">
                <div class="sf-trace__head">
                    <button type="button" class="sf-pill" data-back>‹ Back</button>
                    <div class="sf-trace__nav">
                        <button type="button" class="sf-trace__navbtn" data-dimstep="-1" ${prevDisabled} aria-label="Previous dimension">‹</button>
                        <span class="verdict-dot verdict-dot--${info.tone}"></span>
                        <span class="sf-trace__dimname">${escapeHtml(dim.name)}</span>
                        <span class="sf-trace__count">${dim.index + 1}/${total}</span>
                        <button type="button" class="sf-trace__navbtn" data-dimstep="1" ${nextDisabled} aria-label="Next dimension">›</button>
                    </div>
                    <span class="sf-verdict sf-verdict--${info.tone}">${info.label}</span>
                    <span class="sf-trace__hint">esc to go back · ← → steps quotes</span>
                </div>
                <div class="sf-trace__body">
                    <div class="sf-doc" data-doc="reg">
                        <div class="sf-doc__scroll" data-scroll="reg">
                            <div class="sf-doc__placeholder">Loading…</div>
                        </div>
                    </div>
                    <div class="sf-trace__split"></div>
                    <div class="sf-doc" data-doc="ppr">
                        <div class="sf-doc__scroll" data-scroll="ppr">
                            <div class="sf-doc__placeholder">Loading…</div>
                        </div>
                    </div>
                    <div class="sf-stepper" data-stepper></div>
                </div>
            </div>`;
    }

    function bindTrace(dims) {
        const dim = currentDim(dims);
        const total = dims.length;

        const back = els.views.querySelector("[data-back]");
        if (back) back.addEventListener("click", () => setView("poster"));
        els.views.querySelectorAll("[data-dimstep]").forEach((btn) => {
            btn.addEventListener("click", () => {
                const dir = parseInt(btn.dataset.dimstep, 10);
                const next = dim.index + dir;
                if (next < 0 || next >= total) return;
                const nextDim = dims[next];
                const vis = visibleQuotes(nextDim);
                state.dimIndex = next;
                if (window.localStorage) window.localStorage.setItem(DIM_KEY, String(next));
                state.expanded = new Set();
                state.activeQuote = vis.length ? vis[0].id : null;
                state.lastScrollSrc = "enter";
                render();
            });
        });

        // Default active quote if none / out of range.
        const vis = visibleQuotes(dim);
        if (!vis.some((q) => q.id === state.activeQuote)) {
            state.activeQuote = vis.length ? vis[0].id : null;
        }

        fillTraceColumns(dim).then(() => {
            renderStepper(dim);
            bindHighlightClicks();
            scrollTraceColumns(dim);
        });
    }

    async function fillTraceColumns(dim) {
        const regSource = roleSourceId("reg", dim);
        const pprSource = roleSourceId("ppr", dim);
        await Promise.all([
            fillColumn("reg", regSource, dim.reg.slice(0, state.topK)),
            fillColumn("ppr", pprSource, dim.ppr.slice(0, state.topK)),
        ]);
    }

    async function fillColumn(role, sourceId, quotes) {
        const scroll = els.views.querySelector(`[data-scroll="${role}"]`);
        if (!scroll) return;
        const sources = (state.manifest && state.manifest.sources) || {};
        const source = sourceId ? sources[sourceId] : null;
        if (state.manifestUnavailable || !source) {
            scroll.innerHTML = `<div class="sf-doc__placeholder">Source document unavailable.</div>`;
            return;
        }
        const tag = role === "reg" ? "Preregistration" : "Research paper";
        const title = source.label || source.id;
        let bodyHtml;
        if (source.kind === "pdf" && source.render_mode === "pdf") {
            bodyHtml = buildPdfBody(source, quotes);
        } else {
            const renderData = await getRenderData(source.id);
            const text = (renderData && renderData.text) || "";
            bodyHtml = `<div class="sf-doc__text">${buildTextBody(text, quotes)}</div>`;
        }
        scroll.innerHTML = `
            <div class="sf-doc__header">
                <span class="sf-doc__tag sf-doc__tag--${role}">${escapeHtml(tag)}</span>
                <span class="sf-doc__title">${escapeHtml(title)}</span>
            </div>
            ${bodyHtml}`;
    }

    function buildTextBody(text, quotes) {
        if (!text) return `<div class="sf-doc__placeholder">Document text unavailable.</div>`;
        const ranges = [];
        quotes.forEach((quote) => {
            const chunk = manifestChunk(quote.id);
            const loc = chunk ? locationOf(chunk, ["text", "json"]) : null;
            const body = chunkText(quote);
            let start = loc && Number.isFinite(loc.start) ? loc.start : text.indexOf(body);
            let end = loc && Number.isFinite(loc.end) ? loc.end : (start >= 0 ? start + body.length : -1);
            if (start >= 0 && end > start) ranges.push({ start, end, quote });
        });
        ranges.sort((a, b) => a.start - b.start);
        const clean = [];
        let lastEnd = -1;
        ranges.forEach((r) => { if (r.start >= lastEnd) { clean.push(r); lastEnd = r.end; } });

        let out = "";
        let cursor = 0;
        clean.forEach((r) => {
            out += escapeHtml(text.slice(cursor, r.start));
            const active = r.quote.id === state.activeQuote ? "is-active" : "";
            out += `<span class="sf-hl sf-hl--${r.quote.doc} ${active}" data-quote="${escapeHtml(r.quote.id)}" id="${highlightId(r.quote)}">`
                + escapeHtml(text.slice(r.start, r.end))
                + `<span class="sf-hl__tag">${escapeHtml(r.quote.label)}</span></span>`;
            cursor = r.end;
        });
        out += escapeHtml(text.slice(cursor));
        return out;
    }

    function buildPdfBody(source, quotes) {
        const pageCount = source.page_count || (source.pages || []).length || 1;
        const template = source.page_url_template || "";
        let html = "";
        for (let page = 1; page <= pageCount; page += 1) {
            const src = template.replace("{page_number}", String(page));
            const pageInfo = (source.pages || []).find((p) => p.page_number === page) || {};
            const highlights = quotes.flatMap((quote) => {
                const chunk = manifestChunk(quote.id);
                const locs = chunk && Array.isArray(chunk.locations)
                    ? chunk.locations.filter((l) => l.kind === "pdf" && l.page === page)
                    : [];
                const active = quote.id === state.activeQuote ? "is-active" : "";
                const idAttr = `id="${highlightId(quote)}"`;
                return locs.flatMap((loc) => {
                    const width = loc.page_width || pageInfo.width || 1;
                    const height = loc.page_height || pageInfo.height || 1;
                    return (loc.rects || []).map((rect, ri) => {
                        const left = (rect.x0 / width) * 100;
                        const top = (rect.y0 / height) * 100;
                        const w = ((rect.x1 - rect.x0) / width) * 100;
                        const h = ((rect.y1 - rect.y0) / height) * 100;
                        // Anchor the scroll target / tag to the first rect only.
                        const anchor = ri === 0 ? idAttr : "";
                        const tag = ri === 0 ? `<span class="sf-pdf-hl__tag">${escapeHtml(quote.label)}</span>` : "";
                        return `<span class="sf-pdf-hl sf-pdf-hl--${quote.doc} ${active}" ${anchor} data-quote="${escapeHtml(quote.id)}" style="left:${left}%;top:${top}%;width:${w}%;height:${h}%;">${tag}</span>`;
                    });
                });
            }).join("");
            html += `<div class="sf-pdf-stage" data-page="${page}"><img src="${escapeHtml(src)}" alt="Page ${page}" loading="lazy">${highlights}</div>`;
        }
        return `<div class="sf-doc__pdf">${html}</div>`;
    }

    function highlightId(quote) {
        return `hl-${quote.doc}-${quote.id}`;
    }

    function bindHighlightClicks() {
        els.views.querySelectorAll("[data-quote]").forEach((node) => {
            if (!node.classList.contains("sf-hl") && !node.classList.contains("sf-pdf-hl")) return;
            node.addEventListener("click", () => {
                state.lastScrollSrc = "doc";
                state.activeQuote = node.dataset.quote;
                refreshTraceActive();
            });
        });
    }

    // Update active styling + stepper without rebuilding documents.
    function refreshTraceActive() {
        const dim = currentDim();
        if (!dim) return;
        els.views.querySelectorAll(".sf-hl, .sf-pdf-hl").forEach((node) => {
            node.classList.toggle("is-active", node.dataset.quote === state.activeQuote);
        });
        renderStepper(dim);
        if (state.lastScrollSrc !== "doc") {
            scrollToActive(dim);
        }
    }

    function renderStepper(dim) {
        const stepper = els.views.querySelector("[data-stepper]");
        if (!stepper) return;
        const vis = visibleQuotes(dim);
        if (!vis.length) {
            stepper.innerHTML = `<span class="sf-stepper__empty">No retrieved quotes for this dimension.</span>`;
            return;
        }
        const idx = vis.findIndex((q) => q.id === state.activeQuote);
        const quote = vis[idx] || vis[0];
        const prevDisabled = idx <= 0 ? "disabled" : "";
        const nextDisabled = idx >= vis.length - 1 ? "disabled" : "";
        stepper.innerHTML = `
            <button type="button" class="sf-stepper__btn" data-qstep="-1" ${prevDisabled} aria-label="Previous quote">‹</button>
            <span class="sf-stepper__label sf-quote__label--${quote.doc}">${escapeHtml(quote.label)}</span>
            <span class="sf-stepper__sim">${escapeHtml(quote.score ? quote.score.toFixed(2) : "—")}</span>
            <span class="sf-stepper__count">${idx + 1} of ${vis.length}</span>
            <button type="button" class="sf-stepper__btn" data-qstep="1" ${nextDisabled} aria-label="Next quote">›</button>`;
        stepper.querySelectorAll("[data-qstep]").forEach((btn) => {
            btn.addEventListener("click", () => stepQuote(parseInt(btn.dataset.qstep, 10)));
        });
    }

    function stepQuote(dir) {
        const dim = currentDim();
        if (!dim) return;
        const vis = visibleQuotes(dim);
        const idx = vis.findIndex((q) => q.id === state.activeQuote);
        const next = idx + dir;
        if (next < 0 || next >= vis.length) return;
        state.lastScrollSrc = "step";
        state.activeQuote = vis[next].id;
        refreshTraceActive();
    }

    function scrollWithin(container, el, offset) {
        if (!container || !el) return;
        const cr = container.getBoundingClientRect();
        const er = el.getBoundingClientRect();
        container.scrollTo({
            top: container.scrollTop + (er.top - cr.top) - (offset || 90),
            behavior: prefersReducedMotion ? "auto" : "smooth",
        });
    }

    function scrollToActive(dim) {
        const vis = visibleQuotes(dim);
        const quote = vis.find((q) => q.id === state.activeQuote);
        if (!quote) return;
        const scroll = els.views.querySelector(`[data-scroll="${quote.doc}"]`);
        const el = document.getElementById(highlightId(quote));
        scrollWithin(scroll, el, 90);
    }

    function scrollTraceColumns(dim) {
        const vis = visibleQuotes(dim);
        const active = vis.find((q) => q.id === state.activeQuote);
        if (!active) return;
        scrollToActive(dim);
        // On entry, also bring the other document's first visible quote into view.
        if (state.lastScrollSrc === "enter") {
            const otherRole = active.doc === "reg" ? "ppr" : "reg";
            const otherFirst = vis.find((q) => q.doc === otherRole);
            if (otherFirst) {
                const scroll = els.views.querySelector(`[data-scroll="${otherRole}"]`);
                const el = document.getElementById(highlightId(otherFirst));
                scrollWithin(scroll, el, 90);
            }
        }
    }

    /* ── render-data / manifest fetch ────────────────────────────────────── */

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

    /* ── CSV / share ─────────────────────────────────────────────────────── */

    function csvEscape(value) {
        const text = (value === null || value === undefined ? "" : value).toString();
        return `"${text.replace(/"/g, "\"\"")}"`;
    }

    function quotesPlain(quotes) {
        return parseQuotes(quotes || "")
            .sort(sortByScore)
            .map((quote) => {
                const score = quote.score ? `, relevance_score=${quote.score.toFixed(3)}` : "";
                return `[${quote.id}${score}] ${(quote.text || "").trim()}`;
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
                if (state.dimIndex >= state.items.length) state.dimIndex = 0;
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
            if (els.statusText) els.statusText.textContent = "Sample report";
            render();
        } catch (_error) {
            setStatus("FAILURE", "Could not load the sample report");
        }
    }

    /* ── global keyboard ─────────────────────────────────────────────────── */

    function onKey(event) {
        if (event.target && (event.target.tagName === "INPUT" || event.target.tagName === "TEXTAREA")) return;
        const dims = dimensions();
        if (!dims.length) return;
        const dim = currentDim(dims);
        if (state.view === "poster") {
            if (event.key === "ArrowLeft" && dim.index > 0) { event.preventDefault(); setDimIndex(dim.index - 1); }
            else if (event.key === "ArrowRight" && dim.index < dims.length - 1) { event.preventDefault(); setDimIndex(dim.index + 1); }
            else if (event.key === "Escape") { event.preventDefault(); setView("board"); }
        } else if (state.view === "trace") {
            if (event.key === "Escape") { event.preventDefault(); setView("poster"); }
            else if (event.key === "ArrowLeft") { event.preventDefault(); stepQuote(-1); }
            else if (event.key === "ArrowRight") { event.preventDefault(); stepQuote(1); }
        }
    }

    /* ── wire up ─────────────────────────────────────────────────────────── */

    if (els.copyLink) els.copyLink.addEventListener("click", copyReportLink);
    if (els.csv) els.csv.addEventListener("click", downloadCsv);
    window.addEventListener("keydown", onKey);
    window.addEventListener("beforeunload", () => {
        if (state.pollHandle) window.clearTimeout(state.pollHandle);
    });

    render();
    if (state.demoSrc) loadDemo();
    else pollTaskStatus();
})();
