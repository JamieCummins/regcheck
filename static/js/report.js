(function () {
    "use strict";

    const app = document.getElementById("report-app");
    if (!app) return;

    const EVIDENCE_LIMIT_KEY = "regcheck.report.evidenceLimit";
    const DEFAULT_EVIDENCE_LIMIT = 3;
    const EVIDENCE_LIMIT_OPTIONS = [3, 5, 10, "all"];

    function loadEvidenceLimit() {
        const stored = window.localStorage ? window.localStorage.getItem(EVIDENCE_LIMIT_KEY) : null;
        if (stored === "all") return "all";
        const parsed = parseInt(stored || "", 10);
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
        activeIndex: 0,
        activeEvidence: null,
        currentSourceId: null,
        currentPage: 1,
        evidenceLimit: loadEvidenceLimit(),
        renderDataCache: new Map(),
        lastWorkflowStatus: null,
        logSeen: "",
        pollHandle: null,
    };

    const els = {
        list: document.getElementById("dimension-list"),
        count: document.getElementById("dimension-count"),
        railSummary: document.getElementById("rail-summary"),
        detail: document.getElementById("dimension-detail"),
        empty: document.getElementById("report-empty-state"),
        viewer: document.getElementById("source-viewer"),
        sourceTitle: document.getElementById("source-title"),
        sourceTabs: document.getElementById("source-tabs"),
        sourceCanvas: document.getElementById("source-canvas"),
        rawLink: document.getElementById("source-raw-link"),
        prevPage: document.getElementById("source-prev-page"),
        nextPage: document.getElementById("source-next-page"),
        pageLabel: document.getElementById("source-page-label"),
        closeViewer: document.getElementById("source-close-btn"),
        statusPill: document.getElementById("report-status-pill"),
        statusDot: document.getElementById("report-status-dot"),
        statusText: document.getElementById("report-status-text"),
        copyLink: document.getElementById("copy-report-link-btn"),
        copyLinkLabel: document.getElementById("copy-report-link-label"),
        csv: document.getElementById("download-report-csv-btn"),
        toast: document.getElementById("report-toast"),
        log: document.getElementById("report-log"),
        logList: document.getElementById("report-log-list"),
        logProgress: document.getElementById("report-log-progress"),
    };

    const READY_SOURCE_HTML = els.sourceCanvas ? els.sourceCanvas.innerHTML : "";

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

    function appendLog(text, counts) {
        if (els.logProgress && counts && Number.isFinite(counts.processed) && Number.isFinite(counts.total) && counts.total > 0) {
            els.logProgress.textContent = `${counts.processed}/${counts.total} dimensions`;
        }
        if (!text || !els.logList || text === state.logSeen) return;
        state.logSeen = text;
        const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
        // Single line: the latest status replaces the previous one.
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

    function render() {
        renderDimensionList();
        renderDimensionDetail();
        renderSourceTabs();
        if (state.manifestUnavailable) {
            renderSourceViewer();
        }
    }

    function verdictCounts() {
        const counts = { flag: 0, ok: 0, warn: 0 };
        state.items.forEach((item) => {
            counts[judgementInfo(item.deviation_judgement).tone] += 1;
        });
        return counts;
    }

    function renderRailSummary() {
        if (!els.railSummary) return;
        if (!state.items.length) {
            els.railSummary.innerHTML = "";
            return;
        }
        const counts = verdictCounts();
        const segments = [
            { tone: "flag", value: counts.flag, label: "deviation" },
            { tone: "warn", value: counts.warn, label: "missing" },
            { tone: "ok", value: counts.ok, label: "no deviation" },
        ].filter((segment) => segment.value > 0);
        els.railSummary.innerHTML = segments.map((segment) => `
            <span class="rail-summary__item rail-summary__item--${segment.tone}">
                <span class="verdict-dot verdict-dot--${segment.tone}"></span>
                <strong>${segment.value}</strong> ${escapeHtml(segment.label)}
            </span>
        `).join("");
    }

    function renderDimensionList() {
        if (!els.list) return;
        els.count.textContent = String(state.items.length);
        renderRailSummary();
        els.list.innerHTML = "";
        state.items.forEach((item, index) => {
            const info = judgementInfo(item.deviation_judgement);
            const regCount = parseQuotes(item.registration_content_quotes || "").length;
            const paperCount = parseQuotes(item.paper_content_quotes || "").length;
            const button = document.createElement("button");
            button.type = "button";
            button.className = `dimension-card ${index === state.activeIndex ? "is-active" : ""}`;
            button.innerHTML = `
                <span class="dimension-card__top">
                    <span class="verdict-dot verdict-dot--${info.tone}" title="${escapeHtml(info.label)}"></span>
                    <span class="dimension-card__title">${escapeHtml(item.dimension || `Dimension ${index + 1}`)}</span>
                </span>
                <span class="dimension-card__meta">
                    <span class="judgement-chip judgement-chip--${info.tone}">${info.label}</span>
                    <span class="dimension-card__count">${regCount + paperCount} quotes</span>
                </span>
            `;
            button.addEventListener("click", () => {
                state.activeIndex = index;
                state.activeEvidence = null;
                render();
                const detail = document.querySelector(".report-detail");
                if (detail) detail.scrollTop = 0;
                autoShowTopEvidence();
            });
            els.list.appendChild(button);
        });
    }

    function renderDimensionDetail() {
        if (!els.detail || !els.empty) return;
        if (!state.items.length) {
            els.detail.classList.add("d-none");
            els.empty.classList.remove("d-none");
            return;
        }
        els.empty.classList.add("d-none");
        els.detail.classList.remove("d-none");
        const item = state.items[Math.min(state.activeIndex, state.items.length - 1)] || {};
        const info = judgementInfo(item.deviation_judgement);
        els.detail.innerHTML = `
            <div class="detail-head">
                <p class="detail-eyebrow">Dimension ${state.activeIndex + 1} of ${state.items.length}</p>
                <h2>${escapeHtml(item.dimension || "Dimension")}</h2>
            </div>
            <div class="verdict-banner verdict-banner--${info.tone}">
                <div class="verdict-banner__head">
                    <span class="verdict-dot verdict-dot--${info.tone}"></span>
                    <span class="verdict-banner__label">${info.label}</span>
                </div>
                <p class="verdict-banner__body">${escapeHtml(item.deviation_information || "No deviation information found.")}</p>
            </div>
            <div class="summary-grid">
                <div class="summary-box">
                    <p class="section-title">Registration summary</p>
                    <p>${escapeHtml(item.registration_content_summary || "No summary available.")}</p>
                </div>
                <div class="summary-box">
                    <p class="section-title">Paper summary</p>
                    <p>${escapeHtml(item.paper_content_summary || "No summary available.")}</p>
                </div>
            </div>
            <div class="evidence-section">
                <div class="evidence-section__header">
                    <div>
                        <p class="section-title">Evidence</p>
                        <p class="evidence-limit-note">Showing ${escapeHtml(evidenceLimitLabel(state.evidenceLimit).toLowerCase())} by similarity. Click a quote to view it in context. CSV export includes all chunks.</p>
                    </div>
                    <div class="evidence-limit-control" role="group" aria-label="Evidence chunks shown">
                        ${EVIDENCE_LIMIT_OPTIONS.map((option) => `
                            <button type="button" class="evidence-limit-btn ${option === state.evidenceLimit ? "is-active" : ""}" data-evidence-limit="${option}">
                                ${escapeHtml(option === "all" ? "All" : String(option))}
                            </button>
                        `).join("")}
                    </div>
                </div>
                <div class="evidence-grid">
                    <div class="evidence-column">
                        <p class="evidence-column__title evidence-column__title--reg">Registration</p>
                        <div class="evidence-list" id="registration-evidence-list"></div>
                    </div>
                    <div class="evidence-column">
                        <p class="evidence-column__title evidence-column__title--paper">Paper</p>
                        <div class="evidence-list" id="paper-evidence-list"></div>
                    </div>
                </div>
            </div>
        `;
        els.detail.querySelectorAll("[data-evidence-limit]").forEach((button) => {
            button.addEventListener("click", () => {
                const raw = button.dataset.evidenceLimit;
                const next = raw === "all" ? "all" : parseInt(raw, 10);
                if (!EVIDENCE_LIMIT_OPTIONS.includes(next)) return;
                persistEvidenceLimit(next);
                renderDimensionDetail();
            });
        });
        renderEvidenceList("registration", parseQuotes(item.registration_content_quotes || ""));
        renderEvidenceList("paper", parseQuotes(item.paper_content_quotes || ""));
    }

    function renderEvidenceList(section, quotes) {
        const container = document.getElementById(`${section}-evidence-list`);
        if (!container) return;
        container.innerHTML = "";
        const list = sortedQuotes(quotes, state.evidenceLimit);
        if (!list.length) {
            container.innerHTML = `<div class="evidence-empty">No evidence found</div>`;
            return;
        }
        list.forEach((quote) => {
            const chunk = chunkForQuote(quote);
            const isActive = state.activeEvidence && state.activeEvidence.quote.id === quote.id;
            const hasScore = quote.score && quote.score > 0;
            const pct = hasScore ? Math.max(6, Math.min(100, Math.round(quote.score * 100))) : 0;
            const button = document.createElement("button");
            button.type = "button";
            button.className = `evidence-card evidence-card--${section} ${isActive ? "is-active" : ""}`;
            button.innerHTML = `
                <span class="evidence-card__head">
                    <span class="evidence-card__id">${escapeHtml(quote.id || "Evidence")}</span>
                    ${hasScore ? `
                        <span class="evidence-card__strength" title="Similarity ${quote.score.toFixed(3)}" aria-label="Similarity ${quote.score.toFixed(3)}">
                            <span class="evidence-card__strength-bar" style="width:${pct}%"></span>
                        </span>` : ""}
                </span>
                <span class="evidence-card__text">${escapeHtml((chunk && chunk.text) || quote.text || quote.raw || "")}</span>
                <span class="evidence-card__cue">View in context &rsaquo;</span>
            `;
            button.addEventListener("click", () => openEvidence(quote));
            container.appendChild(button);
        });
    }

    function chunkForQuote(quote) {
        if (!state.manifest || !state.manifest.chunks) return null;
        return state.manifest.chunks[quote.id] || null;
    }

    function renderSourceTabs() {
        if (!els.sourceTabs) return;
        const sources = state.manifest && state.manifest.sources ? state.manifest.sources : {};
        els.sourceTabs.innerHTML = "";
        if (state.manifestUnavailable) {
            return;
        }
        Object.values(sources).forEach((source) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = `source-tab ${source.id === state.currentSourceId ? "is-active" : ""}`;
            button.textContent = source.label || source.id;
            button.addEventListener("click", () => {
                state.currentSourceId = source.id;
                state.activeEvidence = null;
                state.currentPage = 1;
                renderSourceViewer();
                renderSourceTabs();
            });
            els.sourceTabs.appendChild(button);
        });
    }

    function firstLocation(chunk, kind) {
        if (!chunk || !Array.isArray(chunk.locations)) return null;
        return chunk.locations.find((location) => location.kind === kind) || null;
    }

    function openEvidence(quote, opts) {
        opts = opts || {};
        const chunk = chunkForQuote(quote);
        state.activeEvidence = { quote, chunk };
        if (chunk && chunk.source_id) {
            state.currentSourceId = chunk.source_id;
            const pdfLocation = firstLocation(chunk, "pdf");
            state.currentPage = pdfLocation ? pdfLocation.page : 1;
        }
        renderDimensionDetail();
        renderSourceTabs();
        renderSourceViewer();
        // Auto-flick (e.g. on dimension select) updates the docked pane but
        // must not pop the mobile bottom-sheet open.
        if (els.viewer && !opts.auto) els.viewer.classList.add("is-open");
    }

    // A source document is the "registration" if its chunks carry PREREG ids.
    function isRegistrationSource(sourceId) {
        if (!sourceId || !state.manifest || !state.manifest.chunks) return false;
        return Object.values(state.manifest.chunks)
            .some((c) => c.source_id === sourceId && /^PREREG/i.test(c.id || ""));
    }

    // On dimension select / load, pre-open the highest-similarity quote so the
    // source pane flicks to the most relevant page automatically.
    function autoShowTopEvidence() {
        if (state.activeEvidence) return;
        if (state.manifestUnavailable || !state.manifest) return;
        const item = state.items[Math.min(state.activeIndex, state.items.length - 1)];
        if (!item) return;
        const quotes = parseQuotes(item.registration_content_quotes || "")
            .concat(parseQuotes(item.paper_content_quotes || ""))
            .filter((q) => chunkForQuote(q))
            .sort((a, b) => (b.score || 0) - (a.score || 0));
        if (quotes.length) openEvidence(quotes[0], { auto: true });
    }

    function renderUnavailableSourceState() {
        els.sourceTitle.textContent = "Sources unavailable";
        const status = state.evidenceStatus ? `<p>Status: ${escapeHtml(state.evidenceStatus)}</p>` : "";
        const error = state.evidenceError ? `<p>${escapeHtml(state.evidenceError)}</p>` : "";
        const fallback = status || error
            ? ""
            : "<p>Evidence artifacts were not created for this report.</p>";
        els.sourceCanvas.innerHTML = `
            <div class="source-placeholder">
                ${fallback}
                ${status}
                ${error}
            </div>
        `;
        els.pageLabel.textContent = "-";
        els.prevPage.disabled = true;
        els.nextPage.disabled = true;
        els.rawLink.classList.add("d-none");
    }

    function showReadySourceState() {
        els.sourceTitle.textContent = "Context";
        els.sourceCanvas.innerHTML = READY_SOURCE_HTML;
        els.pageLabel.textContent = "-";
        els.prevPage.disabled = true;
        els.nextPage.disabled = true;
        els.rawLink.classList.add("d-none");
    }

    async function renderSourceViewer() {
        if (!els.sourceCanvas || !els.sourceTitle) return;
        if (state.manifestUnavailable) {
            renderUnavailableSourceState();
            return;
        }
        const sources = state.manifest && state.manifest.sources ? state.manifest.sources : {};
        const source = sources[state.currentSourceId] || null;
        if (!source) {
            showReadySourceState();
            return;
        }
        els.sourceTitle.textContent = source.label || source.id;
        if (source.raw_url) {
            els.rawLink.href = source.raw_url;
            els.rawLink.classList.remove("d-none");
        } else {
            els.rawLink.classList.add("d-none");
        }
        const chunk = state.activeEvidence && state.activeEvidence.chunk && state.activeEvidence.chunk.source_id === source.id
            ? state.activeEvidence.chunk
            : null;
        if (source.kind === "pdf" && source.render_mode === "pdf") {
            renderPdfSource(source, chunk);
            return;
        }
        await renderTextSource(source, chunk);
    }

    function renderPdfSource(source, chunk) {
        const pageCount = source.page_count || (source.pages || []).length || 1;
        const template = source.page_url_template || "";
        const isReg = isRegistrationSource(source.id);
        const hlClass = "pdf-highlight" + (isReg ? " pdf-highlight--reg" : "");
        const highlightLocations = chunk && Array.isArray(chunk.locations)
            ? chunk.locations.filter((location) => location.kind === "pdf")
            : [];
        const targetPage = highlightLocations.length
            ? highlightLocations[0].page
            : Math.min(Math.max(state.currentPage || 1, 1), pageCount);

        // Render every page stacked so the source can be scrolled continuously.
        let html = "";
        for (let page = 1; page <= pageCount; page += 1) {
            const src = template.replace("{page_number}", String(page));
            const pageInfo = (source.pages || []).find((p) => p.page_number === page) || {};
            const highlights = highlightLocations
                .filter((location) => location.page === page)
                .flatMap((location) => {
                    const width = location.page_width || pageInfo.width || 1;
                    const height = location.page_height || pageInfo.height || 1;
                    return (location.rects || []).map((rect) => {
                        const left = (rect.x0 / width) * 100;
                        const top = (rect.y0 / height) * 100;
                        const rectWidth = ((rect.x1 - rect.x0) / width) * 100;
                        const rectHeight = ((rect.y1 - rect.y0) / height) * 100;
                        return `<span class="${hlClass} is-active" style="left:${left}%; top:${top}%; width:${rectWidth}%; height:${rectHeight}%;"></span>`;
                    });
                }).join("");
            html += `<div class="pdf-stage" data-page="${page}"><img src="${escapeHtml(src)}" alt="Page ${page}" loading="lazy">${highlights}</div>`;
        }
        els.sourceCanvas.innerHTML = html;
        state.currentPage = targetPage;
        els.pageLabel.textContent = `${targetPage} / ${pageCount}`;
        els.prevPage.disabled = false;
        els.nextPage.disabled = false;
        // Flick to the page that holds the highlight (or the current page).
        const targetEl = els.sourceCanvas.querySelector(`.pdf-stage[data-page="${targetPage}"]`);
        if (targetEl) {
            window.requestAnimationFrame(() => {
                targetEl.scrollIntoView({ block: "start", behavior: "smooth" });
            });
        }
    }

    function scrollToPage(page) {
        const stage = els.sourceCanvas && els.sourceCanvas.querySelector(`.pdf-stage[data-page="${page}"]`);
        if (stage) stage.scrollIntoView({ block: "start", behavior: "smooth" });
    }

    async function renderTextSource(source, chunk) {
        const renderData = await getRenderData(source.id);
        const text = renderData && renderData.text ? renderData.text : "";
        const location = chunk ? (firstLocation(chunk, "text") || firstLocation(chunk, "json")) : null;
        const fallbackStart = chunk && chunk.text ? text.indexOf(chunk.text) : -1;
        const start = location && Number.isFinite(location.start) ? location.start : fallbackStart;
        const end = location && Number.isFinite(location.end) ? location.end : (fallbackStart >= 0 && chunk ? fallbackStart + chunk.text.length : -1);
        els.pageLabel.textContent = source.kind === "json" ? "JSON" : "Text";
        els.prevPage.disabled = true;
        els.nextPage.disabled = true;
        const regClass = isRegistrationSource(source.id) ? " is-reg" : "";
        els.sourceCanvas.innerHTML = `<div class="source-text${regClass}">${highlightText(text, start, end)}</div>`;
        const mark = els.sourceCanvas.querySelector("mark");
        if (mark) {
            window.requestAnimationFrame(() => {
                mark.scrollIntoView({ block: "center", behavior: "smooth" });
            });
        }
    }

    async function getRenderData(sourceId) {
        if (state.renderDataCache.has(sourceId)) return state.renderDataCache.get(sourceId);
        const response = await fetch(`/report/${state.taskId}/sources/${sourceId}/render-data`);
        if (!response.ok) return null;
        const data = await response.json();
        state.renderDataCache.set(sourceId, data);
        return data;
    }

    function highlightText(text, start, end) {
        if (!text) return "";
        if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end <= start) {
            return escapeHtml(text);
        }
        return [
            escapeHtml(text.slice(0, start)),
            "<mark>",
            escapeHtml(text.slice(start, end)),
            "</mark>",
            escapeHtml(text.slice(end)),
        ].join("");
    }

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
        if (els.copyLinkLabel) {
            els.copyLinkLabel.textContent = copied ? "Copied" : "Copy failed";
        }
        if (els.copyLink) els.copyLink.classList.toggle("is-copied", copied);
        showToast(copied ? "Report link copied to clipboard" : "Could not copy link");
        window.setTimeout(() => {
            if (els.copyLinkLabel) els.copyLinkLabel.textContent = "Share";
            if (els.copyLink) els.copyLink.classList.remove("is-copied");
        }, 1800);
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
            renderSourceTabs();
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
            if (data.evidence_available === true) {
                await loadManifest();
            } else if (data.evidence_available === false) {
                state.manifest = null;
                state.manifestUnavailable = data.state === "SUCCESS";
            } else {
                await loadManifest();
            }
            render();
            autoShowTopEvidence();
            if (data.state === "SUCCESS" && els.log) {
                els.log.classList.add("d-none");
            }
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
            autoShowTopEvidence();
        } catch (_error) {
            setStatus("FAILURE", "Could not load the sample report");
        }
    }

    if (els.prevPage) {
        els.prevPage.addEventListener("click", () => {
            state.currentPage = Math.max(1, (state.currentPage || 1) - 1);
            scrollToPage(state.currentPage);
        });
    }
    if (els.nextPage) {
        els.nextPage.addEventListener("click", () => {
            const stages = els.sourceCanvas.querySelectorAll(".pdf-stage[data-page]");
            const max = stages.length || ((state.currentPage || 1) + 1);
            state.currentPage = Math.min(max, (state.currentPage || 1) + 1);
            scrollToPage(state.currentPage);
        });
    }
    if (els.sourceCanvas) {
        els.sourceCanvas.addEventListener("scroll", () => {
            const stages = els.sourceCanvas.querySelectorAll(".pdf-stage[data-page]");
            if (!stages.length) return;
            const canvasTop = els.sourceCanvas.getBoundingClientRect().top;
            let current = 1;
            stages.forEach((st) => {
                if (st.getBoundingClientRect().top - canvasTop <= 80) current = Number(st.dataset.page);
            });
            state.currentPage = current;
            els.pageLabel.textContent = `${current} / ${stages.length}`;
        });
    }
    if (els.closeViewer) {
        els.closeViewer.addEventListener("click", () => {
            els.viewer.classList.remove("is-open");
        });
    }
    if (els.copyLink) els.copyLink.addEventListener("click", copyReportLink);
    if (els.csv) els.csv.addEventListener("click", downloadCsv);

    window.addEventListener("beforeunload", () => {
        if (state.pollHandle) window.clearTimeout(state.pollHandle);
    });

    if (state.demoSrc) {
        loadDemo();
    } else {
        pollTaskStatus();
    }
})();
