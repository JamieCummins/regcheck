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
        // Signature of the inputs the Evidence (documents) view is built from, so
        // background polling doesn't needlessly rebuild it — a rebuild resets the
        // reader's scroll and re-triggers the scroll-to-active-quote every tick.
        docsSignature: null,
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
        download: document.getElementById("report-download-btn"),
        focusBtn: document.getElementById("report-focus-btn"),
        settingsBtn: document.getElementById("report-settings-btn"),
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
        if (normalized === "yes") return { label: "Deviation", tone: "flag", desc: "The paper deviates from what was registered for this dimension." };
        if (normalized === "no") return { label: "No deviation", tone: "ok", desc: "The paper matches what was registered for this dimension." };
        return { label: "Insufficient evidence", tone: "warn", desc: "Not enough information in the provided text to judge this dimension reliably." };
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

    // Evidence IDs the judge cited in its rationale/summaries for this dimension.
    // These can include chunks it did NOT extract as a quote — so their reference
    // would be a dead click unless we surface them from the manifest (below).
    function referencedEvidenceIds(item) {
        const text = [
            item.deviation_information,
            item.registration_content_summary,
            item.paper_content_summary,
        ].join("\n");
        const ids = new Set();
        const re = /\b((?:PREREG|PAPER)_\d+)\b/g;
        let m;
        while ((m = re.exec(text)) !== null) ids.add(m[1]);
        return ids;
    }

    // Full, unlimited evidence set for a role: the extracted quotes PLUS any chunk
    // the rationale/summaries reference but the judge didn't quote (synthesised from
    // the manifest, flagged `referenced`). Without this, a ref like [PREREG_0006] that
    // appears only in the rationale of one dimension can't be located there, even
    // though it's a real chunk — the cause of the "works under X, not under Y" bug.
    function allQuotesForRole(item, role) {
        const raw = role === "reg" ? item.registration_content_quotes : item.paper_content_quotes;
        const quotes = parseQuotes(raw || "").map((q) => Object.assign({}, q, { doc: role }));
        const have = new Set(quotes.map((q) => q.id));
        const prefix = role === "reg" ? "PREREG" : "PAPER";
        const chunks = (state.manifest && state.manifest.chunks) || {};
        referencedEvidenceIds(item).forEach((id) => {
            if (!id.startsWith(prefix) || have.has(id) || !chunks[id]) return;
            const num = parseInt((id.match(/(\d+)/) || [])[1], 10) || 0;
            const text = chunks[id].text || "";
            quotes.push({ id, num, score: 0, text, raw: text, doc: role, referenced: true });
        });
        return quotes;
    }

    function quotesForRole(item, role) {
        return sortedQuotes(allQuotesForRole(item, role), state.evidenceLimit);
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
        // Keep the pill visible through completion: a green "Report complete" check
        // is a clearer end-of-run signal than the pill silently disappearing.
        els.statusPill.classList.remove("d-none");
        const running = stateValue !== "SUCCESS" && stateValue !== "FAILURE";
        if (running) state.everRunning = true;
        els.statusPill.classList.toggle("is-running", running);
        els.statusPill.classList.toggle("is-done", stateValue === "SUCCESS");
        els.statusPill.classList.toggle("is-error", stateValue === "FAILURE");
        const hasCounts = counts && Number.isFinite(counts.processed) && Number.isFinite(counts.total) && counts.total > 0;
        if (els.statusText) {
            els.statusText.textContent =
                stateValue === "SUCCESS" ? "Report complete" :
                stateValue === "FAILURE" ? "Processing failed" :
                hasCounts ? `Processing · ${counts.processed}/${counts.total}` : "Processing";
        }
        // One-off "Report complete" toast, but only for a run we actually watched
        // progress through (not the demo/offline bundle that loads straight to SUCCESS).
        if (stateValue === "SUCCESS" && state.everRunning && !state.completionNotified) {
            state.completionNotified = true;
            showToast("Report complete");
        }
        if (detailText && detailText !== state.lastWorkflowStatus) {
            state.lastWorkflowStatus = detailText;
            els.statusPill.title = detailText;
        }
        // Downloads (CSV / HTML) are only meaningful once the report is complete.
        // Gate via aria-disabled rather than the disabled attribute: a natively
        // disabled button swallows hover, so the explanatory tooltip never showed.
        if (els.download) {
            const done = stateValue === "SUCCESS";
            els.download.disabled = false;
            els.download.setAttribute("aria-disabled", done ? "false" : "true");
            els.download.title = done
                ? "Download"
                : "Report still in progress — download becomes available when it finishes";
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
        if (inDocuments) {
            // Only (re)build the Evidence view when its inputs actually change.
            // render() also runs on every status poll (~3s); rebuilding then would
            // reset the reader's scroll and re-pull the page to the active quote.
            const item = currentItem();
            const sig = [
                state.activeIndex,
                item ? item.dimension : "",
                state.evidenceLimit,
                state.manifestUnavailable ? "x" : (state.manifest ? "m" : "0"),
            ].join("::");
            if (sig !== state.docsSignature) {
                state.docsSignature = sig;
                renderDocumentsView();
            }
        } else {
            state.docsSignature = null;
        }

        if (!state.decisionResizersDone && els.viewDecision) {
            const panels = [
                els.viewDecision.querySelector(".report-dimension-rail"),
                els.viewDecision.querySelector(".report-detail"),
                els.viewDecision.querySelector(".report-quotes"),
            ].filter(Boolean);
            if (panels.length === 3) {
                installResizers(els.viewDecision, panels, "regcheck.report.decisionCols.v2",
                    [140, 360, 300], [0.8, 2.5, 2.4]);
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
                    <span class="judgement-chip judgement-chip--${info.tone}" title="${escapeHtml(info.desc || "")}">${info.label}</span>
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
                <div class="verdict-banner__body">${paragraphsHtml(item.deviation_information || "No deviation information found.")}</div>
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
        const all = allQuotesForRole(item, role);
        if (!all.some((q) => q.id === id)) return;   // unknown reference — ignore
        // Make sure the referenced quote is within the limited set shown in Evidence.
        if (!sortedQuotes(all, state.evidenceLimit).some((q) => q.id === id)) {
            persistEvidenceLimit("all");
        }
        state.view = "documents";
        state.activeQuoteId = id;
        render();   // documents view scrolls the active quote into view on render
    }

    // Like linkifyQuoteRefs, but split the text into <p> paragraphs on blank lines so
    // long blocks (esp. the appended consensus note) don't read as one dense wall in a
    // narrow column. Returns one <p> when there are no paragraph breaks.
    function paragraphsHtml(text) {
        const raw = (text || "").trim();
        if (!raw) return `<p>${linkifyQuoteRefs(text)}</p>`;
        return raw
            .split(/\n\s*\n+/)
            .map((part) => part.trim())
            .filter(Boolean)
            .map((part) => `<p>${linkifyQuoteRefs(part)}</p>`)
            .join("");
    }

    // PDF-extracted quote text carries hard line breaks, end-of-line hyphenation, and
    // whitespace runs that read badly in the callout. Normalise to flowing prose for
    // DISPLAY only — the locator still matches against the raw chunk text.
    function cleanQuoteText(text) {
        return (text || "")
            .replace(/\u00AD/g, "")                          // strip soft hyphens
            .replace(/([A-Za-z])-\s*\n\s*([a-z])/g, "$1$2")  // join words hyphenated across a line break
            .replace(/\s*\n+\s*/g, " ")                       // hard line breaks → spaces
            .replace(/[ \t\u00A0]{2,}/g, " ")                 // collapse runs of spaces
            .replace(/\s+([,.;:!?%)\]])/g, "$1")              // no space before closing punctuation
            .replace(/([(\[])\s+/g, "$1")                     // no space after opening bracket
            .trim();
    }

    // Resolve a quote's text by its ID within the current dimension (for the callout).
    function quoteTextById(id) {
        const item = currentItem();
        if (!item) return null;
        const role = /^PREREG/i.test(id) ? "reg" : "ppr";
        const q = allQuotesForRole(item, role).find((x) => x.id === id);
        if (!q) return null;
        const chunk = chunkForQuote(q);
        return { role, text: (chunk && chunk.text) || q.text || q.raw || "" };
    }

    function hideQuoteCallout() {
        const existing = document.getElementById("quote-callout");
        if (existing) existing.remove();
    }

    // Trial: clicking a reference shows the quote inline (anchored to the ref) instead of
    // jumping to the Evidence view; an "Open in Evidence" affordance keeps the deep path.
    function showQuoteCallout(refEl, id) {
        hideQuoteCallout();
        const info = quoteTextById(id);
        if (!info) { openQuoteRef(id); return; }   // can't resolve — fall back to Evidence
        const box = document.createElement("div");
        box.id = "quote-callout";
        box.className = `quote-callout quote-callout--${info.role}`;
        box.setAttribute("role", "dialog");
        box.innerHTML = `
            <div class="quote-callout__head">
                <span class="quote-callout__id quote-callout__id--${info.role}">${escapeHtml(id)}</span>
                <button type="button" class="quote-callout__close" aria-label="Close">&times;</button>
            </div>
            <div class="quote-callout__text">${escapeHtml(cleanQuoteText(info.text) || "Quote text unavailable.")}</div>
            <button type="button" class="quote-callout__open">Open in Evidence &rsaquo;</button>
        `;
        document.body.appendChild(box);
        // Anchor below the ref if there's room, otherwise above; clamp to the viewport.
        const r = refEl.getBoundingClientRect();
        const bw = box.offsetWidth, bh = box.offsetHeight;
        const left = Math.min(Math.max(8, r.left), window.innerWidth - bw - 8);
        let top = r.bottom + 6;
        if (top + bh > window.innerHeight - 8) top = Math.max(8, r.top - bh - 6);
        box.style.left = `${Math.round(left)}px`;
        box.style.top = `${Math.round(top)}px`;
        box.querySelector(".quote-callout__close").addEventListener("click", hideQuoteCallout);
        box.querySelector(".quote-callout__open").addEventListener("click", () => {
            hideQuoteCallout();
            openQuoteRef(id);
        });
    }

    function bindQuoteRefs(scope) {
        if (!scope) return;
        scope.querySelectorAll(".quote-ref").forEach((btn) => {
            btn.addEventListener("click", (event) => {
                event.stopPropagation();   // don't let the document dismiss close it immediately
                showQuoteCallout(btn, btn.dataset.quoteRef);
            });
        });
    }

    /* ── decision view: right pane (quotes list) ─────────────────────────── */

    function limitControlHtml() {
        const opts = [...EVIDENCE_LIMIT_OPTIONS, "all"];
        return `
            <label class="evidence-limit-control">
                <span class="evidence-limit-control__label">Quotes shown</span>
                <select class="evidence-limit-select" aria-label="Number of quotes shown per document">
                    ${opts.map((option) => `<option value="${option}"${option === state.evidenceLimit ? " selected" : ""}>${escapeHtml(option === "all" ? "All" : String(option))}</option>`).join("")}
                </select>
            </label>`;
    }

    function bindLimitControl(scope, rerender) {
        const select = scope.querySelector(".evidence-limit-select");
        if (!select) return;
        select.addEventListener("change", () => {
            const raw = select.value;
            const next = raw === "all" ? "all" : parseInt(raw, 10);
            if (next !== "all" && !EVIDENCE_LIMIT_OPTIONS.includes(next)) return;
            persistEvidenceLimit(next);
            rerender();
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
                    <div class="summary-box__text">${paragraphsHtml(item.registration_content_summary || "No summary available.")}</div>
                </div>
                <div class="summary-box">
                    <p class="section-title evidence-column__title--paper">Paper summary</p>
                    <div class="summary-box__text">${paragraphsHtml(item.paper_content_summary || "No summary available.")}</div>
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
                    <span class="evidence-card__score" title="Relevance to this dimension (0–1; higher means more relevant)">${hasScore ? quote.score.toFixed(2) : "&mdash;"}</span>
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
                <select class="docs-bar__dim" id="docs-dim-select" aria-label="Switch dimension">${
                    state.items.map((it, i) => `<option value="${i}"${i === state.activeIndex ? " selected" : ""}>${escapeHtml(it.dimension || ("Dimension " + (i + 1)))}</option>`).join("")
                }</select>
                <span class="judgement-chip judgement-chip--${info.tone}">${info.label}</span>
            </div>
            <div class="docs-body">
                <section class="docs-panel docs-panel--quotes">
                    <div class="docs-panel__head">
                        <span class="section-title">Quotes</span>
                        ${limitControlHtml()}
                    </div>
                    <p class="docs-quotes-hint">Click a quote to locate and highlight it in the document.</p>
                    <div class="docs-panel__scroll" id="docs-quotes"></div>
                </section>
                <section class="docs-panel docs-panel--doc">
                    <div class="docs-panel__head">
                        <span class="docs-tag docs-tag--reg">Registration</span>
                    </div>
                    <div class="docs-toolbar" id="docs-reg-tools"></div>
                    <div class="docs-panel__scroll" id="docs-reg-scroll"><div class="source-placeholder">Loading…</div></div>
                </section>
                <section class="docs-panel docs-panel--doc">
                    <div class="docs-panel__head">
                        <span class="docs-tag docs-tag--ppr">Paper</span>
                    </div>
                    <div class="docs-toolbar" id="docs-ppr-tools"></div>
                    <div class="docs-panel__scroll" id="docs-ppr-scroll"><div class="source-placeholder">Loading…</div></div>
                </section>
            </div>
        `;

        // Dimension dropdown — switch dimensions without leaving the Evidence view.
        const dimSelect = els.viewDocuments.querySelector("#docs-dim-select");
        if (dimSelect) dimSelect.addEventListener("change", () => {
            const idx = parseInt(dimSelect.value, 10);
            if (Number.isFinite(idx) && idx !== state.activeIndex) {
                state.activeIndex = idx;
                state.activeQuoteId = null;
                render();
            }
        });
        bindLimitControl(els.viewDocuments, renderDocumentsView);

        const body = els.viewDocuments.querySelector(".docs-body");
        // DOM order is now Quotes | Registration | Paper (quotes moved to the left).
        const panels = [...els.viewDocuments.querySelectorAll(".docs-panel")];
        installResizers(body, panels, "regcheck.report.docsCols.v2", [200, 220, 220], [1.3, 2, 2]);

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
        if (!scroll) return;
        // Capture a fixed base width once (the pane's width at first render = fit-to-pane at
        // 100%). The page is then sized as base x zoom, so resizing the pane no longer
        // rescales the document — only the zoom control changes its size.
        if (!scroll.dataset.baseWidth) {
            const w = scroll.clientWidth;
            if (w > 0) scroll.dataset.baseWidth = String(w);
        }
        if (scroll.dataset.baseWidth) {
            scroll.style.setProperty("--doc-base-width", scroll.dataset.baseWidth + "px");
        }
        scroll.style.setProperty("--doc-zoom", String(zoom || 1));
    }

    // Diagnose *why* the whole evidence set is unavailable, from the status the
    // worker recorded — so a blank pane becomes an actionable message.
    function evidenceUnavailableMessage() {
        const status = state.evidenceStatus;
        const err = (state.evidenceError || "").trim();
        if (status === "missing") {
            return "Evidence wasn’t generated for this report. If this persists, the worker may be on a different release than the site — re-running the comparison should regenerate it.";
        }
        if (status === "error") {
            return err ? ("The source documents couldn’t be prepared: " + err) : "The source documents couldn’t be prepared for this report.";
        }
        if (status === "preparing") {
            return "The source documents are still being prepared — refresh in a moment.";
        }
        // ready/unknown, but the manifest or its artifacts are gone (expired or evicted from storage).
        return "The source documents for this report are no longer available (they may have expired).";
    }

    // Last-resort panel state: explain + offer the original file when we can
    // neither render pages nor show extracted text.
    function renderUnavailable(scroll, source, message) {
        const raw = source && source.raw_url;
        const link = raw
            ? `<a class="source-raw-link" href="${escapeHtml(raw)}" target="_blank" rel="noopener">Open the original file &rsaquo;</a>`
            : "";
        // On a live report where the whole evidence set is missing/errored, offer a
        // one-click "Regenerate" that re-runs from the ORIGINAL uploaded files (kept
        // for the report window), with "Run a new comparison" as the fallback.
        const rerun = (state.taskId && state.manifestUnavailable)
            ? `<button type="button" class="source-rerun-btn" data-regenerate>Regenerate report</button>`
              + `<a class="source-raw-link" href="/compare">Run a new comparison &rsaquo;</a>`
            : "";
        scroll.innerHTML = `<div class="source-placeholder source-placeholder--fallback"><p>${escapeHtml(message)}</p>${rerun}${link}</div>`;
        scroll.dataset.mode = "unavailable";
    }

    // Re-run the comparison from its original uploaded files (kept for the report
    // window). On success the task resets to PENDING and we resume polling; if the
    // inputs have expired (409) or the user can't manage it (403), explain and leave
    // the "Run a new comparison" fallback.
    async function regenerateReport(btn) {
        if (!state.taskId || state.regenerating) return;
        state.regenerating = true;
        if (btn) { btn.disabled = true; btn.textContent = "Re-running…"; }
        try {
            const resp = await fetch(`/reports/${encodeURIComponent(state.taskId)}/regenerate`, { method: "POST" });
            if (resp.ok) {
                showToast("Re-running the comparison…");
                state.manifestUnavailable = false;
                state.manifest = null;
                state.completionNotified = false;
                if (state.pollHandle) window.clearTimeout(state.pollHandle);
                pollTaskStatus();
                return;
            }
            const body = await resp.json().catch(() => ({}));
            showToast(body.detail || "Couldn't regenerate this report.");
        } catch (_error) {
            showToast("Couldn't regenerate this report.");
        } finally {
            state.regenerating = false;
            if (btn) { btn.disabled = false; btn.textContent = "Regenerate report"; }
        }
    }

    async function renderDocPanel(item, role, quotes) {
        const scroll = els.viewDocuments.querySelector(`#docs-${role}-scroll`);
        const tools = els.viewDocuments.querySelector(`#docs-${role}-tools`);
        if (!scroll) return;
        const ui = state.docPanelUI[role] || (state.docPanelUI[role] = { mode: null, zoom: 1, query: "" });
        if (tools) tools.innerHTML = "";
        const sourceId = roleSourceId(item, role);
        const source = sourceId ? manifestSources()[sourceId] : null;
        if (state.manifestUnavailable || !source) {
            const message = state.manifestUnavailable ? evidenceUnavailableMessage() : "This document is unavailable.";
            renderUnavailable(scroll, source, message);
            return;
        }

        const renderData = await getRenderData(source.id);
        // Prefer the reflowed display text (PDF sources keep `text` raw for page/rect
        // mapping); older bundles without display_text fall back to the raw text.
        const text = (renderData && (renderData.display_text || renderData.text)) || "";
        const hasText = !!text.trim();
        const canPdf = source.kind === "pdf" && source.render_mode === "pdf" && !!source.page_url_template;
        const mode = canPdf ? (ui.mode || "text") : "text";   // text view is the default
        const pageCount = source.page_count || (source.pages || []).length || 1;

        const rebuildTools = () => {
            if (!tools) return;
            if (scroll.dataset.mode === "unavailable") { tools.innerHTML = ""; return; }
            tools.innerHTML = buildDocToolbar(role, canPdf, scroll.dataset.mode, pageCount, ui.zoom);
            wireDocToolbar(role, scroll, pageCount);
        };

        // Robust fallback chain: page images → extracted text → "open original".
        const showText = () => {
            if (!hasText) {
                renderUnavailable(scroll, source, "We couldn’t extract readable text from this document.");
            } else {
                scroll.innerHTML = `<div class="source-text source-text--doc ${role === "reg" ? "is-reg" : ""}">${buildTextDoc(text, quotes)}</div>`;
                scroll.dataset.mode = "text";
            }
        };

        if (mode === "page") {
            scroll.innerHTML = buildPdfDoc(source, quotes, role, renderData);
            scroll.dataset.mode = "page";
            const imgs = [...scroll.querySelectorAll(".pdf-stage img")];
            const fallbackFromPage = () => {
                if (hasText) { ui.mode = "text"; refreshDocPanel(role); }      // re-render in text mode
                else { renderUnavailable(scroll, source, "This document couldn’t be rendered."); applyDocZoom(scroll, ui.zoom); rebuildTools(); }
            };
            if (!imgs.length) {
                fallbackFromPage();
            } else {
                let fellBack = false;
                imgs.forEach((img) => img.addEventListener("error", () => {
                    if (fellBack) return;
                    fellBack = true;
                    fallbackFromPage();
                }));
            }
        } else {
            showText();
        }

        applyDocZoom(scroll, ui.zoom);
        bindQuoteClicks(scroll);
        rebuildTools();
        if (scroll.dataset.mode === "page") setupPageNav(role, scroll, pageCount);
        if (ui.query && scroll.dataset.mode === "text") applyDocSearch(role);
    }

    async function refreshDocPanel(role) {
        const item = currentItem();
        if (!item) return;
        await renderDocPanel(item, role, quotesForRole(item, role));
        refreshActiveHighlight();
        markUnlocatedQuotes();
        scrollActiveIntoView();
    }

    function buildDocToolbar(role, canPdf, mode, pageCount, zoom) {
        const pct = Math.round((zoom || 1) * 100);
        const sep = `<span class="docs-tb__sep" aria-hidden="true"></span>`;
        const page = mode === "page" ? `<div class="docs-tb__grp">
                <button type="button" class="docs-tb__btn" data-doc-page="prev" title="Previous page" aria-label="Previous page">&lsaquo;</button>
                <span class="docs-tb__page"><span class="docs-page__ind">1</span><span class="docs-tb__page-sep">/</span>${pageCount}</span>
                <button type="button" class="docs-tb__btn" data-doc-page="next" title="Next page" aria-label="Next page">&rsaquo;</button>
            </div>${sep}` : "";
        const zoomg = `<div class="docs-tb__grp">
                <button type="button" class="docs-tb__btn" data-doc-zoom="out" title="Zoom out" aria-label="Zoom out">&minus;</button>
                <span class="docs-zoom__pct">${pct}%</span>
                <button type="button" class="docs-tb__btn" data-doc-zoom="in" title="Zoom in" aria-label="Zoom in">+</button>
            </div>${sep}`;
        const search = `<div class="docs-tb__grp docs-tb__search">
                <svg class="docs-tb__icon" viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg>
                <input type="search" class="docs-search__input" placeholder="Search document" aria-label="Search document">
                <span class="docs-search__count"></span>
                <button type="button" class="docs-tb__btn" data-doc-search="prev" title="Previous match" aria-label="Previous match">&lsaquo;</button>
                <button type="button" class="docs-tb__btn" data-doc-search="next" title="Next match" aria-label="Next match">&rsaquo;</button>
            </div>`;
        const modes = canPdf ? `<span class="docs-tb__spacer"></span>
            <div class="docs-tb__seg-group" role="group" aria-label="View as">
                <button type="button" class="docs-tb__seg ${mode === "text" ? "is-active" : ""}" data-doc-mode="text">Text</button>
                <button type="button" class="docs-tb__seg ${mode === "page" ? "is-active" : ""}" data-doc-mode="page">Doc</button>
            </div>` : "";
        return page + zoomg + search + modes;
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
                const pct = tools.querySelector(".docs-zoom__pct");
                if (pct) pct.textContent = `${Math.round(ui.zoom * 100)}%`;
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
            if (ind) ind.textContent = String(cur);
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
        // Reflect selection in the middle quote cards. The active card also briefly
        // flashes so that clicking a passage in a document visibly "lights up" the
        // quote it maps to (reverse tracing), not just scrolls the list.
        els.viewDocuments.querySelectorAll("#docs-quotes .evidence-card").forEach((card) => {
            const isActive = card.dataset.quote === id;
            card.classList.toggle("is-active", isActive);
            if (isActive && !prefersReducedMotion) {
                card.classList.remove("is-flash");
                void card.offsetWidth;   // restart the animation if it was already active
                card.classList.add("is-flash");
            }
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

    function centerInScroll(container, el) {
        if (!container || !el) return;
        window.requestAnimationFrame(() => {
            const cr = container.getBoundingClientRect();
            const er = el.getBoundingClientRect();
            const top = container.scrollTop + (er.top - cr.top) - (container.clientHeight / 2) + (er.height / 2);
            container.scrollTo({ top, behavior: prefersReducedMotion ? "auto" : "smooth" });
        });
    }

    function scrollActiveIntoView() {
        if (!state.activeQuoteId) return;
        // Scroll the active quote's CARD in the quotes column into view, so opening a
        // different reference visibly moves to that quote rather than leaving the list
        // where it was (otherwise the user has to hunt for the highlighted card).
        const card = els.viewDocuments.querySelector(
            `#docs-quotes .evidence-card[data-quote="${state.activeQuoteId}"]`
        );
        if (card) centerInScroll(card.closest(".docs-panel__scroll"), card);
        // Scroll the document panel to the located highlight (if the quote was anchored).
        const el = document.getElementById(`docmark-${state.activeQuoteId}`);
        if (el) centerInScroll(el.closest(".docs-panel__scroll"), el);
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
        // Earlier start first; on a tie, the longer span first so the shorter one
        // becomes the clamped tail/anchor rather than swallowing the longer highlight.
        ranges.sort((a, b) => (a.start - b.start) || (b.end - a.end));

        let out = "";
        let cursor = 0;
        ranges.forEach((r) => {
            const active = r.quote.id === state.activeQuoteId ? " is-active" : "";
            const approx = r.approximate ? ` title="Approximate location"` : "";
            // Clamp so marks never overlap already-emitted text (nested <mark>s are
            // invalid HTML). Crucially, EVERY quote still gets an element carrying its
            // id=docmark-<id>, so its reference is always clickable/locatable. With
            // overlap-based chunking + neighbour expansion, adjacent cited chunks often
            // overlap; previously the later one was dropped outright and its quote ref
            // silently failed to locate (the PAPER_0016 report).
            const markStart = Math.max(r.start, cursor);
            if (markStart >= r.end) {
                // Fully inside an already-marked span: emit a zero-width locatable anchor.
                out += `<span class="doc-anchor" data-quote="${escapeHtml(r.quote.id)}" id="${highlightId(r.quote)}"></span>`;
                return;
            }
            if (cursor < markStart) out += escapeHtml(text.slice(cursor, markStart));
            out += `<mark class="doc-mark${active}" data-quote="${escapeHtml(r.quote.id)}" id="${highlightId(r.quote)}"${approx}>`
                + escapeHtml(text.slice(markStart, r.end))
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

    // Download the report as a single self-contained interactive HTML file (the same
    // viewer, with the evidence bundle inlined server-side). Only meaningful in the live
    // app; the offline export itself has no server to call.
    function canExportHtml() {
        // Only real, server-backed reports can be exported: not the inlined offline
        // export itself (__REGCHECK_BUNDLE__), nor the demo/local pseudo-reports.
        return !!state.taskId && state.taskId !== "demo" && state.taskId !== "local"
            && !window.__REGCHECK_BUNDLE__;
    }
    function downloadHtml() {
        if (!canExportHtml()) return;
        window.location.href = `/report/${encodeURIComponent(state.taskId)}/export.html`;
    }

    /* ── download menu (HTML / CSV) ───────────────────────────────────────────── */
    function setupDownloadMenu() {
        const trigger = els.download;
        if (!trigger) return;
        let menu = null;
        const close = () => {
            if (!menu) return;
            menu.remove(); menu = null;
            document.removeEventListener("click", onDocClick, true);
            window.removeEventListener("keydown", onKeyDown, true);
            trigger.setAttribute("aria-expanded", "false");
        };
        const onDocClick = (e) => { if (menu && !menu.contains(e.target) && e.target !== trigger) close(); };
        const onKeyDown = (e) => { if (e.key === "Escape") close(); };
        const open = () => {
            menu = document.createElement("div");
            menu.className = "report-download-menu";
            const canHtml = canExportHtml();
            // PDF export was deprecated (print-to-PDF dropped bullets, quote links
            // and forced landscape). The self-contained HTML is the recommended
            // download — it preserves the interactive viewer, quote-tracing and layout.
            menu.innerHTML = `
                ${canHtml ? '<button type="button" data-dl="html">Interactive report (HTML)<span class="report-download-menu__hint">Recommended</span></button>' : ""}
                <button type="button" data-dl="csv">Data table (CSV)</button>`;
            document.body.appendChild(menu);
            const r = trigger.getBoundingClientRect();
            menu.style.top = `${Math.round(r.bottom + 6)}px`;
            menu.style.right = `${Math.round(window.innerWidth - r.right)}px`;
            if (canHtml) menu.querySelector('[data-dl="html"]').addEventListener("click", () => { close(); downloadHtml(); });
            menu.querySelector('[data-dl="csv"]').addEventListener("click", () => { close(); downloadCsv(); });
            trigger.setAttribute("aria-expanded", "true");
            // Defer listener attach so this opening click doesn't immediately close it.
            window.setTimeout(() => {
                document.addEventListener("click", onDocClick, true);
                window.addEventListener("keydown", onKeyDown, true);
            }, 0);
        };
        trigger.addEventListener("click", () => {
            if (trigger.getAttribute("aria-disabled") === "true") return;
            menu ? close() : open();
        });
    }

    async function copyReportLink() {
        const url = window.location.href;
        let copied = false;
        if (navigator.clipboard && navigator.clipboard.writeText) {
            try {
                await navigator.clipboard.writeText(url);
                copied = true;
            } catch (_error) {
                copied = false;
            }
        }
        if (!copied) {
            // Fallback for browsers/contexts where the async Clipboard API is blocked
            // (notably Safari): use an editable, focused, range-selected field + execCommand.
            try {
                const temp = document.createElement("textarea");
                temp.value = url;
                temp.contentEditable = "true";
                temp.readOnly = false;
                temp.style.position = "fixed";
                temp.style.top = "0";
                temp.style.opacity = "0";
                document.body.appendChild(temp);
                temp.focus();
                temp.setSelectionRange(0, url.length);
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
            applyReportSettings(data.settings);
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


    // Hydrate the viewer from a static {items, manifest, render_data} bundle —
    // shared by the /demo fixture and the CLI's self-contained offline HTML report
    // (which inlines the bundle as window.__REGCHECK_BUNDLE__). No polling/fetch.
    function applyStaticBundle(data, statusLabel) {
        state.items = Array.isArray(data.items) ? data.items : [];
        state.manifest = data.manifest || null;
        state.manifestUnavailable = !state.manifest;
        if (data.render_data) {
            Object.keys(data.render_data).forEach((id) => {
                state.renderDataCache.set(id, data.render_data[id]);
            });
        }
        applyReportSettings(data.settings);   // offline HTML export / demo may carry settings
        setStatus("SUCCESS", statusLabel);
        // Keep the status badge visible (setStatus hides the pill on SUCCESS for
        // live reports); a static bundle has no live state to track.
        if (els.statusPill) els.statusPill.classList.remove("d-none");
        if (els.statusText) els.statusText.textContent = statusLabel;
        render();
    }

    async function loadDemo() {
        try {
            const response = await fetch(state.demoSrc);
            if (!response.ok) throw new Error("demo fixture unavailable");
            const data = await response.json();
            applyStaticBundle(data, "Sample report");
        } catch (_error) {
            setStatus("FAILURE", "Could not load the sample report");
        }
    }

    /* ── keyboard ────────────────────────────────────────────────────────── */

    function onKey(event) {
        if (event.target && (event.target.tagName === "INPUT" || event.target.tagName === "TEXTAREA")) return;
        // Esc leaves focus mode first. When focus mode rides on the native
        // Fullscreen API the browser consumes Esc itself (and fullscreenchange
        // clears the class), so this only fires for the CSS-only fallback.
        if (event.key === "Escape" && document.body.classList.contains("report-focus") && !nativeFullscreenElement()) {
            event.preventDefault();
            exitFocusMode();
            return;
        }
        if (state.view !== "documents") return;
        if (event.key === "Escape") { event.preventDefault(); backToDecision(); }
        else if (event.key === "ArrowLeft") { event.preventDefault(); stepQuote(-1); }
        else if (event.key === "ArrowRight") { event.preventDefault(); stepQuote(1); }
    }

    /* ── focus / full-screen mode ────────────────────────────────────────── */

    function nativeFullscreenElement() {
        return document.fullscreenElement || document.webkitFullscreenElement || null;
    }

    function setFocusButtonState(active) {
        if (!els.focusBtn) return;
        els.focusBtn.setAttribute("aria-pressed", active ? "true" : "false");
        els.focusBtn.title = active ? "Exit full screen" : "Full screen (more room for documents)";
        els.focusBtn.setAttribute("aria-label", active ? "Exit full screen" : "Full screen");
        const label = els.focusBtn.querySelector(".report-focus-btn__label");
        if (label) label.textContent = active ? "Exit full screen" : "Full screen";
    }

    function enterFocusMode() {
        document.body.classList.add("report-focus");
        setFocusButtonState(true);
        // Best-effort native fullscreen (also hides the browser chrome). If it is
        // unsupported or rejected, the CSS class alone still gives the full-bleed,
        // chrome-free layout.
        const el = document.documentElement;
        const request = el.requestFullscreen || el.webkitRequestFullscreen;
        if (request && !nativeFullscreenElement()) {
            try {
                const result = request.call(el);
                if (result && typeof result.catch === "function") result.catch(() => {});
            } catch (_error) { /* keep CSS-only focus mode */ }
        }
    }

    function exitFocusMode() {
        document.body.classList.remove("report-focus");
        setFocusButtonState(false);
        const exit = document.exitFullscreen || document.webkitExitFullscreen;
        if (exit && nativeFullscreenElement()) {
            try {
                const result = exit.call(document);
                if (result && typeof result.catch === "function") result.catch(() => {});
            } catch (_error) { /* class already removed */ }
        }
    }

    function toggleFocusMode() {
        if (document.body.classList.contains("report-focus")) exitFocusMode();
        else enterFocusMode();
    }

    function onFullscreenChange() {
        // Keep the CSS state in sync when the user exits native fullscreen via Esc
        // or the browser's own control rather than our button.
        if (!nativeFullscreenElement() && document.body.classList.contains("report-focus")) {
            document.body.classList.remove("report-focus");
            setFocusButtonState(false);
        }
    }

    /* ── report settings popover (read-only "View settings") ─────────────── */

    const SETTINGS_MODEL_LABELS = { openai: "ChatGPT (GPT-5)", claude: "Claude Opus 4.8", gpustack: "Local model (GPUStack)" };
    const SETTINGS_PARSER_LABELS = { pymupdf: "PyMuPDF", grobid: "GROBID", dpt2: "DPT-2", external: "External (bibr)" };
    const SETTINGS_TYPE_LABELS = { preregistration: "Preregistration", clinical_trials: "Clinical trial (ClinicalTrials.gov)", registered_report: "Registered report" };
    const titleCase = (v) => String(v || "").replace(/^\w/, (c) => c.toUpperCase());

    function settingsRows(s) {
        const rows = [["Comparison", SETTINGS_TYPE_LABELS[s.comparison_type] || titleCase(s.comparison_type) || "Preregistration"]];
        rows.push(["Model", SETTINGS_MODEL_LABELS[s.client] || s.client || "—"]);
        if (s.client === "openai" && s.reasoning_effort) rows.push(["Reasoning effort", titleCase(s.reasoning_effort)]);
        rows.push(["Document parser", SETTINGS_PARSER_LABELS[s.parser_choice] || s.parser_choice || "—"]);
        if (s.multiple_experiments) rows.push(["Multi-study", s.experiment_number ? `Study ${s.experiment_number}` : "Yes"]);
        rows.push(["Append prior outputs", s.append_previous_output ? "Yes" : "No"]);
        const dims = Array.isArray(s.dimensions) ? s.dimensions : [];
        rows.push([`Dimensions (${dims.length})`, dims.length ? dims.join(", ") : "—"]);
        return rows;
    }

    function closeSettingsPopover() {
        const existing = document.getElementById("report-settings-popover");
        if (existing) existing.remove();
        if (els.settingsBtn) els.settingsBtn.setAttribute("aria-expanded", "false");
    }

    function openSettingsPopover() {
        closeSettingsPopover();
        const s = state.reportSettings;
        if (!s || !els.settingsBtn) return;
        const pop = document.createElement("div");
        pop.id = "report-settings-popover";
        pop.className = "report-settings-popover";
        pop.setAttribute("role", "dialog");
        pop.setAttribute("aria-label", "Report settings");
        const rows = settingsRows(s)
            .map(([k, v]) => `<div class="report-settings-popover__row"><dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd></div>`)
            .join("");
        pop.innerHTML = `<div class="report-settings-popover__head">Report settings</div><dl class="report-settings-popover__list">${rows}</dl>`;
        document.body.appendChild(pop);
        // Anchor under the button, right-aligned, clamped to the viewport.
        const r = els.settingsBtn.getBoundingClientRect();
        pop.style.top = `${r.bottom + 8}px`;
        const width = pop.offsetWidth || 280;
        pop.style.left = `${Math.max(8, Math.min(r.right - width, window.innerWidth - width - 8))}px`;
        els.settingsBtn.setAttribute("aria-expanded", "true");
    }

    function toggleSettingsPopover() {
        if (document.getElementById("report-settings-popover")) closeSettingsPopover();
        else openSettingsPopover();
    }

    // Reveal the (otherwise hidden) settings button once the status poll delivers
    // the settings used for this report. Offline/demo bundles carry no settings,
    // so the button stays hidden there.
    function applyReportSettings(settings) {
        if (!els.settingsBtn || !settings || typeof settings !== "object") return;
        state.reportSettings = settings;
        els.settingsBtn.classList.remove("d-none");
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
    setupDownloadMenu();
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
    if (els.focusBtn) els.focusBtn.addEventListener("click", toggleFocusMode);
    if (els.settingsBtn) els.settingsBtn.addEventListener("click", (event) => { event.stopPropagation(); toggleSettingsPopover(); });
    // "Regenerate report" lives in dynamically-rendered evidence placeholders, so bind via delegation.
    document.addEventListener("click", (event) => {
        const regen = event.target.closest("[data-regenerate]");
        if (regen) regenerateReport(regen);
    });
    document.addEventListener("click", (event) => {
        if (!event.target.closest("#report-settings-popover") && !event.target.closest("#report-settings-btn")) closeSettingsPopover();
    });
    document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeSettingsPopover(); });
    window.addEventListener("resize", closeSettingsPopover);
    window.addEventListener("scroll", closeSettingsPopover, true);
    document.addEventListener("fullscreenchange", onFullscreenChange);
    document.addEventListener("webkitfullscreenchange", onFullscreenChange);
    window.addEventListener("keydown", onKey);
    // Dismiss the quote callout on outside-click, Escape, or scrolling the page underneath
    // (it's fixed-positioned). Scrolling INSIDE the callout's own text must NOT dismiss it.
    document.addEventListener("click", (event) => {
        if (!event.target.closest(".quote-callout") && !event.target.closest(".quote-ref")) hideQuoteCallout();
    });
    document.addEventListener("keydown", (event) => { if (event.key === "Escape") hideQuoteCallout(); });
    window.addEventListener("scroll", (event) => {
        const target = event.target;
        if (target && target.closest && target.closest(".quote-callout")) return;  // scrolling within the popup
        hideQuoteCallout();
    }, true);
    window.addEventListener("beforeunload", () => {
        if (state.pollHandle) window.clearTimeout(state.pollHandle);
    });

    if (window.__REGCHECK_BUNDLE__) applyStaticBundle(window.__REGCHECK_BUNDLE__, "Report");
    else if (state.demoSrc) loadDemo();
    else pollTaskStatus();
})();
