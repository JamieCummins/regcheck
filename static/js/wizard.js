(function () {
    "use strict";

    document.addEventListener("DOMContentLoaded", function () {
        const form = document.getElementById("wizard-form");
        if (!form) return;
        // One wizard script, two flows: "compare" (registration vs paper, the
        // default) and "quality" (single-document registration-quality
        // assessment; fewer steps, no paper upload). Set via data-flow on the form.
        const FLOW = form.dataset.flow || "compare";

        const steps = Array.from(document.querySelectorAll(".form-step"));
        const panel = document.querySelector(".wizard-panel");
        const progress = document.querySelector(".wizard-progress");
        const progressFill = document.getElementById("wizard-progress-fill");
        const stepCount = document.getElementById("wizard-step-count");

        const backButton = document.getElementById("step-back");
        const forwardButton = document.getElementById("step-forward");
        const submitButton = document.getElementById("app-compare-btn");

        const defaultModeButton = document.getElementById("default-mode");
        const customModeButton = document.getElementById("custom-mode");

        const parserSelect = document.getElementById("parser_choice");
        const modelSelect = document.getElementById("client");
        const reasoningEffortGroup = document.getElementById("reasoning_effort_group");
        const reasoningEffortSelect = document.getElementById("reasoning_effort");
        const appendSelect = document.getElementById("append_previous_output");

        const multipleExperimentsSelect = document.getElementById("multiple_experiments");
        const experimentNumberGroup = document.getElementById("experiment_number_group");
        const experimentNumberInput = document.getElementById("experiment_number");
        const experimentTextInput = document.getElementById("experiment_text");

        const comparisonSelect = document.getElementById("prereg_source");
        const comparisonTypeInput = document.getElementById("comparison_type");
        const generalPane = document.getElementById("general-upload");
        const clinicalPane = document.getElementById("clinical-upload");
        const osfPane = document.getElementById("osf-upload");

        const preregFileInput = document.getElementById("prereg_file");
        const paperFileInput = document.getElementById("paper_file");
        const registrationInput = document.getElementById("registration_id");
        const osfUrlInput = document.getElementById("osf_url");

        const dimensionsDataInput = document.getElementById("dimensions-data");

        let currentStep = 1;
        let defaultModeActive = false;

        function escapeHtml(value) {
            return (value === null || value === undefined ? "" : String(value))
                .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
        }

        const ICON = {
            trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/></svg>',
            check: '<svg class="po-check" viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 12l5 5L20 6"/></svg>',
            empty: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2.5"/><path d="M3 9h18M8 14h8"/></svg>'
        };
        const GLYPH = {
            brain: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 5a3 3 0 0 0-3 3 3 3 0 0 0-1 5.5A2.5 2.5 0 0 0 7 18a2.5 2.5 0 0 0 2 1V5zM15 5a3 3 0 0 1 3 3 3 3 0 0 1 1 5.5A2.5 2.5 0 0 1 17 18a2.5 2.5 0 0 1-2 1V5z"/></svg>',
            pulse: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12h4l2-6 4 12 2-6h6"/></svg>',
            trend: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 17l6-6 4 4 7-8M21 7v5h-5"/></svg>',
            mouse: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 15c0-3.9 3.1-6.4 6.9-6.4 3.4 0 6.2 2.1 7 5.2.2.8-.4 1.4-1.2 1.4"/><path d="M5 15c0 1.7 1.3 3 3 3h7.6c1.7 0 3-1.3 3-3"/><circle cx="15.4" cy="9.6" r="2.3"/><circle cx="13.4" cy="13" r="0.7" fill="currentColor"/><path d="M5 15c-1.8 0-3.1.9-3.1 2.4"/></svg>'
        };

        /* Discipline presets are defined ONCE in the backend
           (backend/data/discipline_dimensions.json) and injected by the /compare
           template as window.REGCHECK_DISCIPLINES, so the wizard, API, and CLI stay
           in sync. Glyphs are UI-only and stay here, keyed by preset. */
        const GLYPH_BY_KEY = { psychology: "brain", clinical: "pulse", economics: "trend", preclinical: "mouse" };
        const DISCIPLINES = (window.REGCHECK_DISCIPLINES || []).map((d) => ({
            key: d.key,
            label: d.label,
            meta: (d.dims ? d.dims.length : 0) + " dimensions",
            glyph: GLYPH_BY_KEY[d.key] || "brain",
            dims: d.dims || []
        }));

        /* ---- dimensions state (two-pane editor) --------------------------- */
        const rowsEl = document.getElementById("dimension-rows");
        const detailEl = document.getElementById("dimension-detail-pane");
        const presetEl = document.getElementById("dim-preset");
        const presetBtn = document.getElementById("dim-preset-btn");
        const presetMenu = document.getElementById("dim-preset-menu");
        const presetLabel = document.getElementById("dim-preset-label");
        const addDimensionButton = document.getElementById("add-dimension");

        let dimensions = [];
        let selectedId = null;
        let activeDiscipline = DISCIPLINES[0] ? DISCIPLINES[0].key : "psychology";
        let dimensionsTouched = false;
        let dragFromId = null;
        let pendingFocusName = false;

        let _uid = 0;
        function uid() { _uid += 1; return "dim-" + _uid + "-" + Math.random().toString(36).slice(2, 7); }

        function loadDiscipline(key) {
            const d = DISCIPLINES.find((x) => x.key === key) || DISCIPLINES[0];
            return d.dims.map((x) => ({ id: uid(), name: x.name, definition: x.definition }));
        }

        function buildPresetMenu() {
            if (!presetMenu) return;
            presetMenu.innerHTML = '<div class="pm-cap">Load defaults for…</div>' +
                DISCIPLINES.map((d) =>
                    '<button type="button" class="preset-opt" data-key="' + d.key + '" role="option">' +
                        '<span class="po-ico">' + (GLYPH[d.glyph] || GLYPH.brain) + '</span>' +
                        '<span class="po-text"><span class="po-name">' + escapeHtml(d.label) + '</span>' +
                        '<span class="po-meta">' + escapeHtml(d.meta) + '</span></span>' +
                        ICON.check +
                    '</button>'
                ).join("");
            presetMenu.querySelectorAll(".preset-opt").forEach((opt) => {
                opt.addEventListener("click", () => {
                    applyPreset(opt.dataset.key, true);
                    toggleMenu(false);
                });
            });
        }

        function updatePresetActive() {
            const cur = DISCIPLINES.find((d) => d.key === activeDiscipline) || DISCIPLINES[0];
            if (presetLabel) presetLabel.textContent = cur.label;
            if (presetMenu) {
                presetMenu.querySelectorAll(".preset-opt").forEach((opt) => {
                    opt.classList.toggle("on", opt.dataset.key === activeDiscipline);
                });
            }
        }

        function toggleMenu(force) {
            if (!presetMenu || !presetBtn) return;
            const open = force !== undefined ? force : presetMenu.hidden;
            presetMenu.hidden = !open;
            presetBtn.classList.toggle("open", open);
            presetBtn.setAttribute("aria-expanded", String(open));
        }

        function renderRows() {
            if (!rowsEl) return;
            rowsEl.innerHTML = "";
            dimensions.forEach((d, index) => {
                const row = document.createElement("div");
                row.className = "tp-row" + (d.id === selectedId ? " sel" : "");
                row.dataset.id = d.id;
                row.innerHTML =
                    '<span class="grip" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i></span>' +
                    '<span class="num">' + (index + 1) + '</span>' +
                    '<span class="tp-name">' + (d.name ? escapeHtml(d.name) : '<span class="tp-name--empty">Untitled</span>') + '</span>' +
                    '<button type="button" class="icon-btn danger tp-del" aria-label="Remove dimension">' + ICON.trash + '</button>';

                row.addEventListener("click", () => selectDimension(d.id));
                row.querySelector(".tp-del").addEventListener("click", (event) => {
                    event.stopPropagation();
                    deleteDimension(d.id);
                });

                const grip = row.querySelector(".grip");
                grip.addEventListener("mousedown", () => { row.draggable = true; });
                grip.addEventListener("touchstart", () => { row.draggable = true; }, { passive: true });

                row.addEventListener("dragstart", (event) => {
                    dragFromId = d.id;
                    row.classList.add("drag-ghost");
                    event.dataTransfer.effectAllowed = "move";
                    try { event.dataTransfer.setData("text/plain", d.id); } catch (_e) { /* no-op */ }
                });
                row.addEventListener("dragend", () => {
                    row.draggable = false;
                    row.classList.remove("drag-ghost");
                    dragFromId = null;
                    clearOver();
                });
                row.addEventListener("dragover", (event) => {
                    if (dragFromId === null) return;
                    event.preventDefault();
                    setOver(row);
                });
                row.addEventListener("drop", (event) => {
                    event.preventDefault();
                    if (dragFromId !== null && dragFromId !== d.id) reorder(dragFromId, d.id);
                    clearOver();
                });

                rowsEl.appendChild(row);
            });
        }

        function setOver(row) {
            clearOver();
            if (row !== rowsEl.querySelector(".drag-ghost")) row.classList.add("is-over");
        }
        function clearOver() {
            if (!rowsEl) return;
            rowsEl.querySelectorAll(".tp-row.is-over").forEach((r) => r.classList.remove("is-over"));
        }

        function updateRowName(id, name) {
            if (!rowsEl) return;
            const nameEl = rowsEl.querySelector('.tp-row[data-id="' + id + '"] .tp-name');
            if (nameEl) nameEl.innerHTML = name ? escapeHtml(name) : '<span class="tp-name--empty">Untitled</span>';
        }

        function renderEditor() {
            if (!detailEl) return;
            const current = dimensions.find((d) => d.id === selectedId);
            if (!current) {
                detailEl.classList.add("tp-empty");
                detailEl.innerHTML = ICON.empty + "<div>Add a dimension to get started.</div>";
                return;
            }
            detailEl.classList.remove("tp-empty");
            const index = dimensions.findIndex((d) => d.id === current.id);
            detailEl.innerHTML =
                '<div class="td2-top"><span class="num">' + (index + 1) + '</span>' +
                '<span class="td2-meta">of ' + dimensions.length + '</span></div>' +
                '<input type="text" class="q-name" placeholder="Untitled dimension">' +
                '<textarea class="q-def" rows="1" placeholder="Add a definition — what should RegCheck look for on this dimension?"></textarea>';

            const nameEl = detailEl.querySelector(".q-name");
            const defEl = detailEl.querySelector(".q-def");
            nameEl.value = current.name;
            defEl.value = current.definition;

            nameEl.addEventListener("input", () => {
                current.name = nameEl.value;
                dimensionsTouched = true;
                updateRowName(current.id, current.name);
                if (nameEl.value.trim()) showDimWarning(false);
            });
            defEl.addEventListener("input", () => {
                current.definition = defEl.value;
                dimensionsTouched = true;
            });

            if (pendingFocusName) {
                pendingFocusName = false;
                nameEl.focus();
            }
        }

        function selectDimension(id) {
            selectedId = id;
            if (rowsEl) {
                rowsEl.querySelectorAll(".tp-row").forEach((r) => r.classList.toggle("sel", r.dataset.id === id));
            }
            renderEditor();
        }

        function addDimension() {
            const nd = { id: uid(), name: "", definition: "" };
            dimensions.push(nd);
            selectedId = nd.id;
            dimensionsTouched = true;
            pendingFocusName = true;
            renderRows();
            renderEditor();
            if (rowsEl) rowsEl.scrollTop = rowsEl.scrollHeight;
        }

        function deleteDimension(id) {
            const index = dimensions.findIndex((d) => d.id === id);
            if (index < 0) return;
            dimensions.splice(index, 1);
            dimensionsTouched = true;
            if (selectedId === id) selectedId = dimensions[0] ? dimensions[0].id : null;
            renderRows();
            renderEditor();
        }

        function reorder(fromId, toId) {
            const from = dimensions.findIndex((d) => d.id === fromId);
            const to = dimensions.findIndex((d) => d.id === toId);
            if (from < 0 || to < 0 || from === to) return;
            const moved = dimensions.splice(from, 1)[0];
            dimensions.splice(to, 0, moved);
            dimensionsTouched = true;
            renderRows();
            renderEditor();
        }

        /* Quick-start field question: the default flow loads the chosen field's
           preset dimensions (the custom flow has the full two-pane editor instead). */
        const defaultDisciplineSelect = document.getElementById("default_discipline");
        if (defaultDisciplineSelect && !defaultDisciplineSelect.options.length) {
            DISCIPLINES.forEach((d) => {
                const opt = document.createElement("option");
                opt.value = d.key;
                opt.textContent = d.label;
                defaultDisciplineSelect.appendChild(opt);
            });
        }
        function applyDefaultDiscipline() {
            if (!defaultDisciplineSelect || !defaultDisciplineSelect.value) return;
            applyPreset(defaultDisciplineSelect.value, false);
        }
        if (defaultDisciplineSelect) {
            defaultDisciplineSelect.addEventListener("change", applyDefaultDiscipline);
        }

        function applyPreset(key, userInitiated) {
            activeDiscipline = key;
            dimensions = loadDiscipline(key);
            selectedId = dimensions[0] ? dimensions[0].id : null;
            if (userInitiated) dimensionsTouched = true;
            if (countDimensions() >= 1) showDimWarning(false);
            updatePresetActive();
            renderRows();
            renderEditor();
        }

        if (presetBtn) {
            presetBtn.addEventListener("click", (event) => {
                event.stopPropagation();
                toggleMenu();
            });
        }
        document.addEventListener("click", (event) => {
            if (presetEl && !presetEl.contains(event.target)) toggleMenu(false);
        });
        if (addDimensionButton) addDimensionButton.addEventListener("click", addDimension);

        /* ---- step navigation ---------------------------------------------- */
        function countDimensions() {
            return dimensions.filter((d) => (d.name || "").trim() !== "").length;
        }

        function showDimWarning(show) {
            const el = document.getElementById("dim-empty-warning");
            if (el) el.hidden = !show;
        }

        function getActiveSequence() {
            let base;
            if (FLOW === "quality") {
                // Single-document flow: no RR tiles, no field question, no multi-study.
                base = defaultModeActive ? [1, 7, 8] : [1, 2, 3, 4, 5, 7, 8];
            } else {
                // Step 10 (preregistration vs Registered Report) opens BOTH modes.
                base = defaultModeActive ? [1, 10, 9, 6, 7, 8] : [1, 10, 2, 3, 4, 5, 6, 7, 8];
            }
            // "Append previous outputs?" (step 5) only makes sense with >= 2 dimensions.
            return countDimensions() < 2 ? base.filter((s) => s !== 5) : base;
        }

        function goToStep(step) {
            const sequence = getActiveSequence();
            const target = sequence.includes(step) ? step : sequence[sequence.length - 1];
            steps.forEach(function (el) {
                el.classList.toggle("active", Number(el.dataset.step) === target);
            });
            currentStep = target;
            updateProgress();
            updateNav();
            if (panel) panel.scrollTop = 0;
            if (typeof window.scrollTo === "function") window.scrollTo({ top: 0, behavior: "smooth" });
            if (currentStep === 8) checkFiles();
        }

        function goToNext() {
            // The comparison needs at least one dimension; block leaving step 4 empty.
            if (currentStep === 4 && countDimensions() < 1) {
                showDimWarning(true);
                return;
            }
            const sequence = getActiveSequence();
            const idx = sequence.indexOf(currentStep);
            if (idx === -1) { goToStep(sequence[0]); return; }
            if (idx < sequence.length - 1) goToStep(sequence[idx + 1]);
        }

        function goToPrevious() {
            const sequence = getActiveSequence();
            const idx = sequence.indexOf(currentStep);
            if (idx > 0) goToStep(sequence[idx - 1]);
        }

        function updateProgress() {
            const sequence = getActiveSequence();
            const position = sequence.indexOf(currentStep) + 1;
            const intro = currentStep === 1;
            if (progress) progress.classList.toggle("is-hidden", intro);
            if (progressFill) progressFill.style.width = intro ? "0%" : Math.round((position / sequence.length) * 100) + "%";
            if (stepCount) stepCount.textContent = intro ? "" : "Step " + position + " of " + sequence.length;
        }

        function updateNav() {
            const sequence = getActiveSequence();
            const finalStep = sequence[sequence.length - 1];
            const isIntro = currentStep === 1;
            const isFinal = currentStep === finalStep;
            // Tile-choice steps advance by clicking a tile, not via Next.
            const isTileStep = currentStep === 10;
            if (backButton) {
                backButton.classList.toggle("is-hidden", isIntro);
                backButton.disabled = isIntro;
            }
            if (forwardButton) {
                forwardButton.classList.toggle("is-hidden", isIntro || isFinal || isTileStep);
                forwardButton.disabled = isIntro || isFinal || isTileStep;
            }
            if (submitButton) submitButton.classList.toggle("d-none", !isFinal);
        }

        /* ---- reasoning effort --------------------------------------------- */
        function updateReasoningEffortVisibility() {
            if (!reasoningEffortGroup || !reasoningEffortSelect) return;
            // ChatGPT is the reasoning model that takes a reasoning effort.
            const isReasoning = modelSelect && modelSelect.value === "openai";
            reasoningEffortGroup.hidden = !isReasoning;
            reasoningEffortSelect.disabled = !isReasoning;
            if (isReasoning && !reasoningEffortSelect.value) reasoningEffortSelect.value = "medium";
        }

        /* ---- uploads ------------------------------------------------------ */
        function fileDisplay(input) {
            const dropzone = input.closest(".dropzone");
            return dropzone ? dropzone.querySelector(".dropzone__file") : null;
        }

        function refreshFileDisplay(input) {
            const display = fileDisplay(input);
            const dropzone = input.closest(".dropzone");
            if (!display) return;
            const files = input.files;
            if (files && files.length > 0) {
                const names = Array.from(files).map((f) => f.name);
                // Multiple files are combined server-side into one document; show that.
                display.textContent = files.length === 1 ? names[0] : `${files.length} files (combined): ${names.join(", ")}`;
                display.title = names.join("\n");
                if (dropzone) dropzone.classList.add("has-file");
            } else {
                display.textContent = display.dataset.empty || "No file selected";
                display.removeAttribute("title");
                if (dropzone) dropzone.classList.remove("has-file");
            }
        }

        function fileExt(name) {
            const i = name.lastIndexOf(".");
            return i >= 0 ? name.slice(i).toLowerCase() : "";
        }

        // Reject unsupported file types at upload time (rather than failing the run).
        // Every selected file is checked, since several can be combined into one.
        function validateFile(input) {
            if (!input.files || !input.files.length) return true;
            const allowed = (input.getAttribute("accept") || "")
                .split(",").map((s) => s.trim().toLowerCase()).filter(Boolean);
            const dropzone = input.closest(".dropzone");
            const bad = allowed.length
                ? Array.from(input.files).find((f) => allowed.indexOf(fileExt(f.name)) === -1)
                : null;
            if (bad) {
                const display = fileDisplay(input);
                if (display) {
                    display.textContent = "Unsupported file (" + (fileExt(bad.name) || "no extension") + "). Use PDF, DOCX, TXT, or HTML.";
                    display.removeAttribute("title");
                }
                if (dropzone) { dropzone.classList.add("has-error"); dropzone.classList.remove("has-file"); }
                input.value = "";
                return false;
            }
            if (dropzone) dropzone.classList.remove("has-error");
            return true;
        }

        function setupDropzone(input) {
            if (!input) return;
            const dropzone = input.closest(".dropzone");
            input.addEventListener("change", function () {
                if (validateFile(input)) refreshFileDisplay(input);
                checkFiles();
            });
            if (!dropzone) return;
            ["dragenter", "dragover"].forEach(function (type) {
                dropzone.addEventListener(type, function (event) {
                    event.preventDefault();
                    dropzone.classList.add("is-dragover");
                });
            });
            ["dragleave", "dragend"].forEach(function (type) {
                dropzone.addEventListener(type, function () {
                    dropzone.classList.remove("is-dragover");
                });
            });
            dropzone.addEventListener("drop", function (event) {
                event.preventDefault();
                dropzone.classList.remove("is-dragover");
                if (input.disabled) return;
                if (event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files.length) {
                    input.files = event.dataTransfer.files;
                    if (validateFile(input)) refreshFileDisplay(input);
                    checkFiles();
                }
            });
        }

        function selectedSource() {
            return comparisonSelect ? comparisonSelect.value : "upload";
        }

        function setPreregSource(source) {
            const isClinical = source === "clinical";
            const isOsf = source === "osf";
            if (comparisonTypeInput) {
                comparisonTypeInput.value = isClinical ? "clinical_trials" : "general_preregistration";
            }
            if (generalPane) generalPane.hidden = isClinical || isOsf;
            if (clinicalPane) clinicalPane.hidden = !isClinical;
            if (osfPane) osfPane.hidden = !isOsf;
            if (preregFileInput) preregFileInput.disabled = isClinical || isOsf;
            if (registrationInput) registrationInput.disabled = !isClinical;
            if (osfUrlInput) osfUrlInput.disabled = !isOsf;
            if (!dimensionsTouched && FLOW !== "quality") {
                applyPreset(isClinical ? "clinical" : "psychology", false);
            }
            checkFiles();
        }

        function checkFiles() {
            if (!submitButton) return;
            // The quality flow assesses the registration alone — no paper upload.
            const paperOk = FLOW === "quality" || (paperFileInput && paperFileInput.files.length > 0);
            const source = selectedSource();
            let ready;
            if (source === "clinical") {
                ready = registrationInput && registrationInput.value.trim().length > 0 && paperOk;
            } else if (source === "osf") {
                ready = osfUrlInput && osfUrlInput.value.trim().length > 0 && paperOk;
            } else {
                ready = preregFileInput && preregFileInput.files.length > 0 && paperOk;
            }
            submitButton.disabled = !ready;
            submitButton.classList.toggle("is-ready", !!ready);
        }

        /* ---- experiment text ---------------------------------------------- */
        function updateExperimentText() {
            if (!experimentTextInput) return;
            if (multipleExperimentsSelect && multipleExperimentsSelect.value === "yes") {
                const num = experimentNumberInput ? experimentNumberInput.value : "";
                experimentTextInput.value = num
                    ? "This was a multi-study paper. The preregistration here pertains to Study " + num + " only."
                    : "";
            } else {
                experimentTextInput.value = "";
            }
        }

        /* ---- wire-up ------------------------------------------------------ */
        if (backButton) backButton.addEventListener("click", goToPrevious);
        if (forwardButton) forwardButton.addEventListener("click", goToNext);

        if (defaultModeButton) {
            defaultModeButton.addEventListener("click", function () {
                if (parserSelect) parserSelect.value = "pymupdf";
                if (modelSelect) modelSelect.value = "openai";
                if (reasoningEffortSelect) reasoningEffortSelect.value = "medium";
                if (appendSelect) appendSelect.value = "yes";
                updateReasoningEffortVisibility();
                defaultModeActive = true;
                applyDefaultDiscipline();
                goToStep(getActiveSequence()[1]);
            });
        }
        if (customModeButton) {
            customModeButton.addEventListener("click", function () {
                defaultModeActive = false;
                goToStep(getActiveSequence()[1]);
            });
        }

        if (modelSelect) modelSelect.addEventListener("change", updateReasoningEffortVisibility);

        // Progression is manual: making a selection never advances the step on
        // its own — only the Next button (and the explicit defaults/customize
        // buttons) moves the wizard forward. Selection handlers below only
        // reveal/update controls on the current step.

        if (multipleExperimentsSelect) {
            multipleExperimentsSelect.addEventListener("change", function () {
                const yes = multipleExperimentsSelect.value === "yes";
                if (experimentNumberGroup) experimentNumberGroup.hidden = !yes;
                updateExperimentText();
            });
        }
        if (experimentNumberInput) experimentNumberInput.addEventListener("input", updateExperimentText);

        if (comparisonSelect) {
            comparisonSelect.addEventListener("change", function () {
                setPreregSource(selectedSource());
            });
        }

        // Registered Report mode: chosen on the step-10 tiles (posted via the
        // hidden comparison_mode input). RR relabels the two upload panes as
        // Stage 1 / Stage 2, retitles the source step, and drops the
        // ClinicalTrials.gov source (an RR Stage 1 is a manuscript, uploaded or
        // fetched from OSF — never a CT.gov record).
        const comparisonModeInput = document.getElementById("comparison_mode");
        const comparisonModeCards = Array.from(document.querySelectorAll(".comparison-mode-card"));
        function applyComparisonMode() {
            // Only the compare flow has a comparison mode; the quality page's own
            // labels must not be overwritten with comparison-flow wording.
            if (!comparisonModeInput) return;
            const rr = comparisonModeInput.value === "registered_report";
            const regTitle = document.querySelector("#registration-slot .dropzone__title");
            const regHint = document.querySelector("#registration-slot .dropzone__hint");
            const paperSlot = document.getElementById("paper_file");
            const paperZone = paperSlot ? paperSlot.closest(".dropzone") : null;
            if (regTitle) regTitle.textContent = rr ? "Stage 1 manuscript" : "Preregistration";
            if (regHint) regHint.textContent = rr
                ? "The in-principle-accepted Stage 1 manuscript (one or more files)"
                : "Provide one or more files (e.g., registration, statistical analysis plan)";
            if (paperZone) {
                const t = paperZone.querySelector(".dropzone__title");
                const h = paperZone.querySelector(".dropzone__hint");
                if (t) t.textContent = rr ? "Stage 2 manuscript" : "Paper";
                if (h) h.textContent = rr
                    ? "The completed Stage 2 manuscript (one or more files)"
                    : "Provide one or more files (e.g., paper, supplementary materials)";
            }
            const sourceStep = document.querySelector('.form-step[data-step="7"]');
            if (sourceStep) {
                const heading = sourceStep.querySelector(".step-heading");
                const sub = sourceStep.querySelector(".step-sub");
                const srcLabel = sourceStep.querySelector('label[for="prereg_source"]');
                if (heading) heading.textContent = rr ? "Where is your Stage 1 manuscript?" : "Where is your preregistration?";
                if (sub) sub.textContent = rr
                    ? "Upload files or paste a link to the Open Science Framework."
                    : "Upload files, provide a ClinicalTrials.gov identifier, or paste a link to the Open Science Framework.";
                if (srcLabel) srcLabel.textContent = rr ? "Stage 1 source" : "Preregistration source";
            }
            if (comparisonSelect) {
                const clinicalOption = comparisonSelect.querySelector('option[value="clinical"]');
                if (clinicalOption) clinicalOption.hidden = rr;
                if (rr && comparisonSelect.value === "clinical") {
                    comparisonSelect.value = "upload";
                    setPreregSource(selectedSource());
                }
            }
        }
        function setComparisonMode(mode) {
            if (comparisonModeInput) comparisonModeInput.value = mode;
            comparisonModeCards.forEach((card) => {
                const on = card.dataset.comparisonMode === mode;
                card.classList.toggle("is-selected", on);
                card.setAttribute("aria-pressed", String(on));
            });
            applyComparisonMode();
        }
        comparisonModeCards.forEach((card) => {
            card.addEventListener("click", function () {
                setComparisonMode(card.dataset.comparisonMode || "standard");
                goToNext();
            });
        });
        applyComparisonMode();
        if (registrationInput) registrationInput.addEventListener("input", checkFiles);
        if (osfUrlInput) osfUrlInput.addEventListener("input", checkFiles);

        setupDropzone(preregFileInput);
        setupDropzone(paperFileInput);

        let preflightAcknowledged = false;

        function setSubmitBusy(busy, label) {
            if (!submitButton) return;
            if (busy) {
                if (!submitButton.dataset.idleLabel) submitButton.dataset.idleLabel = submitButton.textContent;
                submitButton.disabled = true;
                submitButton.textContent = label || "Checking…";
                submitButton.classList.add("is-busy");
            } else {
                submitButton.disabled = false;
                if (submitButton.dataset.idleLabel) submitButton.textContent = submitButton.dataset.idleLabel;
                submitButton.classList.remove("is-busy");
            }
        }

        function proceedWithSubmit() {
            preflightAcknowledged = true;
            if (form) form.submit();   // native submit; bypasses this submit handler
        }

        function closeThinModal() {
            const existing = document.getElementById("preflight-thin-modal");
            if (existing) existing.remove();
        }

        function showThinRegistrationWarning(result, source) {
            closeThinModal();
            const chars = (result && typeof result.chars === "number") ? result.chars : 0;
            const hint = source === "osf"
                ? "If your OSF link points to a registration page whose actual content is an attached file (e.g. a PDF), go back and paste the direct link to that file instead of the general project link."
                : "If this is a scanned PDF, its text may not be extractable — try a text-based PDF, DOCX, or TXT.";
            const overlay = document.createElement("div");
            overlay.id = "preflight-thin-modal";
            overlay.className = "rc-modal-overlay";
            overlay.setAttribute("role", "dialog");
            overlay.setAttribute("aria-modal", "true");
            overlay.setAttribute("aria-labelledby", "preflight-thin-title");
            overlay.innerHTML =
                '<div class="rc-modal">' +
                '  <h2 class="rc-modal__title" id="preflight-thin-title">Very little text found in your registration</h2>' +
                '  <p class="rc-modal__body">We could only extract about <strong>' + chars + ' characters</strong> from your registration. ' + hint + '</p>' +
                '  <p class="rc-modal__body">You can run the comparison anyway, but with so little registration text the results may be unreliable.</p>' +
                '  <div class="rc-modal__actions">' +
                '    <button type="button" class="btn btn-secondary" data-preflight="back">Go back and fix</button>' +
                '    <button type="button" class="btn btn-primary" data-preflight="anyway">Run anyway</button>' +
                '  </div>' +
                '</div>';
            document.body.appendChild(overlay);
            const back = () => closeThinModal();
            overlay.querySelector('[data-preflight="back"]').addEventListener("click", back);
            overlay.querySelector('[data-preflight="anyway"]').addEventListener("click", () => { closeThinModal(); proceedWithSubmit(); });
            overlay.addEventListener("click", (e) => { if (e.target === overlay) back(); });
            document.addEventListener("keydown", function esc(e) {
                if (e.key === "Escape") { closeThinModal(); document.removeEventListener("keydown", esc); }
            });
            const anywayBtn = overlay.querySelector('[data-preflight="anyway"]');
            if (anywayBtn) anywayBtn.focus();
        }

        async function runRegistrationPreflight(source) {
            setSubmitBusy(true, "Checking registration…");
            let result = null;
            try {
                const fd = new FormData();
                fd.append("prereg_source", source);
                if (source === "osf") {
                    fd.append("osf_url", (osfUrlInput && osfUrlInput.value.trim()) || "");
                } else {
                    const file = preregFileInput && preregFileInput.files && preregFileInput.files[0];
                    if (file) fd.append("preregistration", file);
                }
                const resp = await fetch("/preflight/registration", { method: "POST", body: fd });
                if (resp.ok) result = await resp.json();
            } catch (_err) {
                result = null;   // probe failed → don't get in the user's way
            }
            setSubmitBusy(false);
            if (result && result.ok && result.thin) showThinRegistrationWarning(result, source);
            else proceedWithSubmit();
        }

        if (form) {
            form.addEventListener("submit", function (event) {
                // Re-assert the prereg-source field state right before submission:
                // inactive branches are excluded by being `disabled`, and disabled
                // inputs are never sent — so this guarantees the active source's
                // field (e.g. the OSF link) is enabled and actually submitted.
                setPreregSource(selectedSource());
                if (!dimensionsDataInput) return;
                const out = dimensions
                    .map((d) => ({ dimension: (d.name || "").trim(), definition: (d.definition || "").trim() }))
                    .filter((d) => d.dimension !== "");
                if (out.length === 0) {
                    event.preventDefault();
                    goToStep(4);
                    window.alert("Please specify at least one dimension to compare.");
                    return;
                }
                dimensionsDataInput.value = JSON.stringify(out);

                // Pre-flight (#5): warn — never block — when little/no registration
                // text can be parsed. CT.gov registries are structured, so skip them.
                // Any probe failure falls through to a normal submit.
                if (preflightAcknowledged) return;
                const source = selectedSource();
                if (source !== "osf" && source !== "upload") return;
                // Skip the thin-parse probe when several registration files are
                // combined — it only probes one file, so the signal would be misleading.
                if (source === "upload" && preregFileInput && preregFileInput.files.length > 1) return;
                event.preventDefault();
                runRegistrationPreflight(source);
            });
        }

        /* ---- init --------------------------------------------------------- */
        buildPresetMenu();
        applyPreset(activeDiscipline, false);
        updateReasoningEffortVisibility();
        setPreregSource(selectedSource());
        goToStep(1);
    });
})();
