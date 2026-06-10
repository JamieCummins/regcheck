(function () {
    "use strict";

    document.addEventListener("DOMContentLoaded", function () {
        const form = document.getElementById("wizard-form");
        if (!form) return;

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

        const comparisonSelect = document.getElementById("clinical_registration");
        const comparisonTypeInput = document.getElementById("comparison_type");
        const generalPane = document.getElementById("general-upload");
        const clinicalPane = document.getElementById("clinical-upload");

        const preregFileInput = document.getElementById("prereg_file");
        const paperFileInput = document.getElementById("paper_file");
        const registrationInput = document.getElementById("registration_id");

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
            globe: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.5 2.6 2.5 15.4 0 18M12 3c-2.5 2.6-2.5 15.4 0 18"/></svg>'
        };

        /* discipline default sets (verbatim from design handoff data.js) */
        const DISCIPLINES = [
            {
                key: "psychology", label: "Psychology", meta: "7 dimensions", glyph: "brain",
                dims: [
                    { name: "Sample size", definition: "The planned number of participants to be included in the study. This could include a precise number of units or observations that the researchers intend to sample, or a range, minimum, or maximum. It could also include a stopping rule (that is, how the decision to terminate data collection will be made) if one is specified." },
                    { name: "Data source", definition: "Where the data will be / was obtained, and the processes therein. For example, a student sample, online (e.g., m-turk, prolific), the community, an existing archival dataset, etc." },
                    { name: "Hypotheses", definition: "The specific, directional predictions the study set out to test, as stated prior to data collection — including which effects are confirmatory versus exploratory." },
                    { name: "Outcome measures", definition: "The primary dependent variables or instruments used to operationalise the constructs of interest, including how each is scored." },
                    { name: "Analysis plan", definition: "The statistical models, tests, and any covariates specified to evaluate each hypothesis, including corrections for multiple comparisons." },
                    { name: "Exclusion criteria", definition: "Rules determining which participants, trials, or observations are removed before analysis (e.g., attention checks, outliers, incomplete responses)." },
                    { name: "Conditions & assignment", definition: "The experimental conditions and the procedure used to assign participants to them (e.g., random assignment, counterbalancing)." }
                ]
            },
            {
                key: "clinical", label: "Clinical / Medical", meta: "7 dimensions", glyph: "pulse",
                dims: [
                    { name: "Sample size & power", definition: "The target enrolment and the power analysis justifying it, including the assumed effect size, alpha, and power." },
                    { name: "Eligibility criteria", definition: "Inclusion and exclusion criteria defining the study population, including age, diagnosis, and comorbidity constraints." },
                    { name: "Intervention & comparator", definition: "The treatment, dose, route, and schedule, together with the comparator or control arm." },
                    { name: "Primary endpoint", definition: "The principal outcome used to judge efficacy, with its precise definition and measurement timepoint." },
                    { name: "Secondary endpoints", definition: "Additional pre-specified outcomes that support or contextualise the primary analysis." },
                    { name: "Blinding", definition: "Who is masked to allocation — participants, providers, outcome assessors — and the method used to maintain it." },
                    { name: "Analysis plan", definition: "The pre-specified statistical methods, analysis population (ITT / per-protocol), and handling of missing data." }
                ]
            },
            {
                key: "economics", label: "Economics", meta: "7 dimensions", glyph: "trend",
                dims: [
                    { name: "Data & sample", definition: "The dataset, unit of observation, time period, and how the analysis sample is constructed and restricted." },
                    { name: "Identification strategy", definition: "The research design used to identify the causal effect (RCT, IV, difference-in-differences, RDD, etc.) and the assumptions it relies on." },
                    { name: "Treatment & instruments", definition: "The treatment variable and any instruments used, including the exclusion restriction being invoked." },
                    { name: "Outcome variables", definition: "The dependent variables of interest and how each is measured or constructed." },
                    { name: "Estimation method", definition: "The estimator and specification, including fixed effects, controls, and functional form." },
                    { name: "Inference & standard errors", definition: "How statistical inference is conducted, including clustering, robust corrections, and any multiple-testing adjustments." },
                    { name: "Robustness checks", definition: "Pre-specified alternative specifications, placebo tests, and sensitivity analyses." }
                ]
            },
            {
                key: "social", label: "Social science (general)", meta: "5 dimensions", glyph: "globe",
                dims: [
                    { name: "Research question", definition: "The central question or claim the study is designed to address, framed as it was prior to data collection." },
                    { name: "Sample & data", definition: "The population, sampling frame, and data collection procedure, including target sample size." },
                    { name: "Key variables", definition: "The independent and dependent variables, and how each construct is measured." },
                    { name: "Analysis plan", definition: "The analytic approach and models specified to answer the research question." },
                    { name: "Exclusions & robustness", definition: "Pre-specified data exclusions and any sensitivity or robustness checks." }
                ]
            }
        ];

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
        let activeDiscipline = "psychology";
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

        function applyPreset(key, userInitiated) {
            activeDiscipline = key;
            dimensions = loadDiscipline(key);
            selectedId = dimensions[0] ? dimensions[0].id : null;
            if (userInitiated) dimensionsTouched = true;
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
        function getActiveSequence() {
            return defaultModeActive ? [1, 6, 7, 8] : [1, 2, 3, 4, 5, 6, 7, 8];
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
            if (backButton) {
                backButton.classList.toggle("is-hidden", isIntro);
                backButton.disabled = isIntro;
            }
            if (forwardButton) {
                forwardButton.classList.toggle("is-hidden", isIntro || isFinal);
                forwardButton.disabled = isIntro || isFinal;
            }
            if (submitButton) submitButton.classList.toggle("d-none", !isFinal);
        }

        /* ---- reasoning effort --------------------------------------------- */
        function updateReasoningEffortVisibility() {
            if (!reasoningEffortGroup || !reasoningEffortSelect) return;
            const isOpenAI = modelSelect && modelSelect.value === "openai";
            reasoningEffortGroup.hidden = !isOpenAI;
            reasoningEffortSelect.disabled = !isOpenAI;
            if (isOpenAI && !reasoningEffortSelect.value) reasoningEffortSelect.value = "medium";
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
            if (input.files && input.files.length > 0) {
                display.textContent = input.files[0].name;
                display.title = input.files[0].name;
                if (dropzone) dropzone.classList.add("has-file");
            } else {
                display.textContent = display.dataset.empty || "No file selected";
                display.removeAttribute("title");
                if (dropzone) dropzone.classList.remove("has-file");
            }
        }

        function setupDropzone(input) {
            if (!input) return;
            const dropzone = input.closest(".dropzone");
            input.addEventListener("change", function () {
                refreshFileDisplay(input);
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
                    refreshFileDisplay(input);
                    checkFiles();
                }
            });
        }

        function isClinicalSelected() {
            return comparisonSelect && comparisonSelect.value === "yes";
        }

        function setComparisonMode(isClinical) {
            if (comparisonTypeInput) {
                comparisonTypeInput.value = isClinical ? "clinical_trials" : "general_preregistration";
            }
            if (generalPane) generalPane.hidden = isClinical;
            if (clinicalPane) clinicalPane.hidden = !isClinical;
            if (preregFileInput) preregFileInput.disabled = isClinical;
            if (registrationInput) registrationInput.disabled = !isClinical;
            if (!dimensionsTouched) {
                applyPreset(isClinical ? "clinical" : "psychology", false);
            }
            checkFiles();
        }

        function checkFiles() {
            if (!submitButton) return;
            const hasPaper = paperFileInput && paperFileInput.files.length > 0;
            let ready;
            if (isClinicalSelected()) {
                ready = registrationInput && registrationInput.value.trim().length > 0 && hasPaper;
            } else {
                ready = preregFileInput && preregFileInput.files.length > 0 && hasPaper;
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
                    ? "This was a multistudy paper. The preregistration here pertains to Experiment " + num + " only."
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
                if (parserSelect) parserSelect.value = "grobid";
                if (modelSelect) modelSelect.value = "openai";
                if (reasoningEffortSelect) reasoningEffortSelect.value = "medium";
                if (appendSelect) appendSelect.value = "yes";
                updateReasoningEffortVisibility();
                defaultModeActive = true;
                goToStep(6);
            });
        }
        if (customModeButton) {
            customModeButton.addEventListener("click", function () {
                defaultModeActive = false;
                goToStep(2);
            });
        }

        if (modelSelect) modelSelect.addEventListener("change", updateReasoningEffortVisibility);

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
                setComparisonMode(isClinicalSelected());
            });
        }
        if (registrationInput) registrationInput.addEventListener("input", checkFiles);

        setupDropzone(preregFileInput);
        setupDropzone(paperFileInput);

        if (form) {
            form.addEventListener("submit", function (event) {
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
            });
        }

        /* ---- init --------------------------------------------------------- */
        buildPresetMenu();
        applyPreset(activeDiscipline, false);
        updateReasoningEffortVisibility();
        setComparisonMode(isClinicalSelected());
        goToStep(1);
    });
})();
