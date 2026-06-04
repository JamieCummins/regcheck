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
        pollHandle: null,
    };

    const els = {
        list: document.getElementById("dimension-list"),
        count: document.getElementById("dimension-count"),
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
        logLine: document.getElementById("workflow-log-line"),
        spinner: document.getElementById("workflow-spinner"),
        statusNote: document.getElementById("processing-status-note"),
        copyLink: document.getElementById("copy-report-link-btn"),
        csv: document.getElementById("download-report-csv-btn"),
    };

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
        if (normalized === "yes") return { label: "Deviation", cls: "judgement-chip--yes" };
        if (normalized === "no") return { label: "No Deviation", cls: "judgement-chip--no" };
        return { label: "Missing", cls: "judgement-chip--missing" };
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

    function appendWorkflowLog(statusText) {
        if (!statusText || statusText === state.lastWorkflowStatus) return;
        state.lastWorkflowStatus = statusText;
        if (!els.logLine) return;
        const timestamp = new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
        });
        els.logLine.textContent = `[${timestamp}] ${statusText}`;
    }

    function updateStatus(stateValue) {
        if (els.spinner) {
            if (stateValue === "SUCCESS" || stateValue === "FAILURE") {
                els.spinner.classList.add("d-none");
            } else {
                els.spinner.classList.remove("d-none");
            }
        }
        if (els.statusNote) {
            els.statusNote.textContent = stateValue === "FAILURE" ? "Processing failed" : "";
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
            renderSourceTabs();
        } catch (_error) {
            state.manifest = state.manifest || null;
        }
    }

    function render() {
        renderDimensionList();
        renderDimensionDetail();
        renderSourceTabs();
        if (state.manifestUnavailable) {
            renderSourceViewer();
        }
    }

    function renderDimensionList() {
        if (!els.list) return;
        els.count.textContent = String(state.items.length);
        els.list.innerHTML = "";
        state.items.forEach((item, index) => {
            const info = judgementInfo(item.deviation_judgement);
            const regCount = parseQuotes(item.registration_content_quotes || "").length;
            const paperCount = parseQuotes(item.paper_content_quotes || "").length;
            const button = document.createElement("button");
            button.type = "button";
            button.className = `dimension-card ${index === state.activeIndex ? "is-active" : ""}`;
            button.innerHTML = `
                <span class="dimension-card__title">${escapeHtml(item.dimension || `Dimension ${index + 1}`)}</span>
                <span class="dimension-card__meta">
                    <span class="judgement-chip ${info.cls}">${info.label}</span>
                    <span>${regCount + paperCount} chunks</span>
                </span>
            `;
            button.addEventListener("click", () => {
                state.activeIndex = index;
                state.activeEvidence = null;
                render();
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
            <div class="detail-meta">
                <span class="judgement-chip ${info.cls}">${info.label}</span>
            </div>
            <h2>${escapeHtml(item.dimension || "Dimension")}</h2>
            <div class="deviation-box">
                <p class="section-title">Judgement</p>
                <p>${escapeHtml(item.deviation_information || "No deviation information found.")}</p>
            </div>
            <div class="summary-grid">
                <div class="summary-box">
                    <p class="section-title">Registration Summary</p>
                    <p>${escapeHtml(item.registration_content_summary || "No summary available.")}</p>
                </div>
                <div class="summary-box">
                    <p class="section-title">Paper Summary</p>
                    <p>${escapeHtml(item.paper_content_summary || "No summary available.")}</p>
                </div>
            </div>
            <div class="evidence-section">
                <div class="evidence-section__header">
                    <div>
                        <p class="section-title">Evidence</p>
                        <p class="evidence-limit-note">Showing ${escapeHtml(evidenceLimitLabel(state.evidenceLimit).toLowerCase())} by similarity. CSV export includes all chunks.</p>
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
                    <div>
                        <p class="section-title">Registration</p>
                        <div class="evidence-list" id="registration-evidence-list"></div>
                    </div>
                    <div>
                        <p class="section-title">Paper</p>
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
            container.innerHTML = `<div class="source-placeholder">No evidence found</div>`;
            return;
        }
        list.forEach((quote) => {
            const chunk = chunkForQuote(quote);
            const isActive = state.activeEvidence && state.activeEvidence.quote.id === quote.id;
            const button = document.createElement("button");
            button.type = "button";
            button.className = `evidence-card ${isActive ? "is-active" : ""}`;
            button.innerHTML = `
                <span class="evidence-card__head">
                    <span>${escapeHtml(quote.id || "Evidence")}</span>
                    <span class="evidence-card__score">${quote.score ? quote.score.toFixed(3) : ""}</span>
                </span>
                <span class="evidence-card__text">${escapeHtml((chunk && chunk.text) || quote.text || quote.raw || "")}</span>
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

    function openEvidence(quote) {
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
        if (els.viewer) els.viewer.classList.add("is-open");
    }

    function renderUnavailableSourceState() {
        els.sourceTitle.textContent = "Sources Unavailable";
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

    async function renderSourceViewer() {
        if (!els.sourceCanvas || !els.sourceTitle) return;
        if (state.manifestUnavailable) {
            renderUnavailableSourceState();
            return;
        }
        const sources = state.manifest && state.manifest.sources ? state.manifest.sources : {};
        const source = sources[state.currentSourceId] || null;
        if (!source) {
            els.sourceTitle.textContent = "No Source Selected";
            els.sourceCanvas.innerHTML = `<div class="source-placeholder">No source selected</div>`;
            els.pageLabel.textContent = "-";
            els.rawLink.classList.add("d-none");
            els.prevPage.disabled = true;
            els.nextPage.disabled = true;
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
        if (!state.currentPage || state.currentPage < 1) state.currentPage = 1;
        if (state.currentPage > pageCount) state.currentPage = pageCount;
        const template = source.page_url_template || "";
        const src = template.replace("{page_number}", String(state.currentPage));
        const pageInfo = (source.pages || []).find((page) => page.page_number === state.currentPage) || {};
        const highlightLocations = chunk && Array.isArray(chunk.locations)
            ? chunk.locations.filter((location) => location.kind === "pdf" && location.page === state.currentPage)
            : [];
        const highlights = highlightLocations.flatMap((location) => {
            const width = location.page_width || pageInfo.width || 1;
            const height = location.page_height || pageInfo.height || 1;
            return (location.rects || []).map((rect) => {
                const left = (rect.x0 / width) * 100;
                const top = (rect.y0 / height) * 100;
                const rectWidth = ((rect.x1 - rect.x0) / width) * 100;
                const rectHeight = ((rect.y1 - rect.y0) / height) * 100;
                return `<span class="pdf-highlight" style="left:${left}%; top:${top}%; width:${rectWidth}%; height:${rectHeight}%;"></span>`;
            });
        }).join("");
        els.pageLabel.textContent = `${state.currentPage} / ${pageCount}`;
        els.prevPage.disabled = state.currentPage <= 1;
        els.nextPage.disabled = state.currentPage >= pageCount;
        els.sourceCanvas.innerHTML = `
            <div class="pdf-stage">
                <img src="${escapeHtml(src)}" alt="${escapeHtml(source.label || "PDF page")}">
                ${highlights}
            </div>
        `;
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
        els.sourceCanvas.innerHTML = `<div class="source-text">${highlightText(text, start, end)}</div>`;
        const mark = els.sourceCanvas.querySelector("mark");
        if (mark) mark.scrollIntoView({ block: "center" });
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
        const blob = new Blob(["\uFEFF", csvText], { type: "text/csv;charset=utf-8;" });
        const link = document.createElement("a");
        link.download = "regcheck-report.csv";
        link.href = window.URL.createObjectURL(blob);
        link.style.display = "none";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    async function copyReportLink() {
        try {
            await navigator.clipboard.writeText(window.location.href);
            els.copyLink.textContent = "Copied";
        } catch (_error) {
            els.copyLink.textContent = "Copy Failed";
        }
        window.setTimeout(() => {
            els.copyLink.textContent = "Copy link";
        }, 1400);
    }

    async function pollTaskStatus() {
        try {
            const response = await fetch(`/task_status/${state.taskId}`);
            const data = await response.json();
            state.evidenceStatus = data.evidence_status || null;
            state.evidenceError = data.evidence_error || null;
            appendWorkflowLog(data.status || data.state || "");
            updateStatus(data.state);
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
            if (data.state !== "SUCCESS" && data.state !== "FAILURE") {
                state.pollHandle = window.setTimeout(pollTaskStatus, 3000);
            }
        } catch (_error) {
            appendWorkflowLog("Failed to fetch report status");
            updateStatus("FAILURE");
        }
    }

    if (els.prevPage) {
        els.prevPage.addEventListener("click", () => {
            state.currentPage = Math.max(1, state.currentPage - 1);
            renderSourceViewer();
        });
    }
    if (els.nextPage) {
        els.nextPage.addEventListener("click", () => {
            state.currentPage += 1;
            renderSourceViewer();
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

    pollTaskStatus();
})();
