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
            mouse: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 15c0-3.9 3.1-6.4 6.9-6.4 3.4 0 6.2 2.1 7 5.2.2.8-.4 1.4-1.2 1.4"/><path d="M5 15c0 1.7 1.3 3 3 3h7.6c1.7 0 3-1.3 3-3"/><circle cx="15.4" cy="9.6" r="2.3"/><circle cx="13.4" cy="13" r="0.7" fill="currentColor"/><path d="M5 15c-1.8 0-3.1.9-3.1 2.4"/></svg>'
        };

        /* discipline default sets (verbatim from design handoff data.js) */
        const DISCIPLINES = [
            {
                key: "psychology", label: "Psychology", meta: "9 dimensions", glyph: "brain",
                dims: [
                    { name: "Hypotheses", definition: "Specific, directional (verbal) predictions the study set to test out (usually stated in the Introduction [frequently at the end of the introduction] or right before the introduction of a new study), including effects which are confirmatory vs. exploratory. Relevant keywords: 'expect(ed)', 'predict(ed)', or 'hypothesise(d)'. for primary hypotheses: 'key', 'leading', 'main', 'major', 'primary', or 'principal'. for secondary hypotheses: 'additional', 'auxiliary', 'minor', or 'secondary'." },
                    { name: "Data source", definition: "Where the data will be / was obtained, and the processes therein. For example, a student sample, online (e.g., m-turk, prolific), the community, an existing archival dataset, etc." },
                    { name: "Inclusion and exclusion criteria", definition: "Inclusion criteria specify how subjects (or units of analysis otherwise specified) are selected for eligibility in the study.\nExclusion criteria specify how data exclusions are determined (e.g., how will outliers be determined? Will awareness checks be implemented?)." },
                    { name: "Sample size", definition: "The precise number of participants that the researchers intend to sample or a range, minimum, or maximum. This can also include a stopping rule if one is specified." },
                    { name: "Manipulated variables", definition: "Variables that are experimentally manipulated, that is, the experimenter controlled the level / treatment that participants received." },
                    { name: "Measured variables", definition: "Variables that are observed/ recorded in a study. This can include outcome measures, as well as any measured predictors or covariates." },
                    { name: "Statistical models", definition: "The statistical tests that are used to analyse the data, based on the manipulated and measured variables, to address the research questions and hypotheses." },
                    { name: "Transformations", definition: "Approaches to processing data from one form, structure, or scale to another to prepare it for analysis (e.g., log transforming, centering, or recoding the data)." },
                    { name: "Missing data", definition: "Information about how missing data will be handled in the processing and analysis of the study." }
                ]
            },
            {
                key: "clinical", label: "Clinical / Medical", meta: "11 dimensions", glyph: "pulse",
                dims: [
                    { name: "Eligibility – inclusion criteria", definition: "All explicitly stated conditions, characteristics, or thresholds that must be satisfied for a participant or study unit (e.g., individual, cluster, site) to be enrolled in the study. This includes required demographic characteristics (e.g., age range, sex/gender, pregnancy/menopausal status), diagnosis or target condition (including staging, severity, symptom duration, biomarker status), clinical status (e.g., performance status, treatment-naïve status), prior or current treatments that are required or permitted, contextual factors (e.g., recruitment setting, healthcare setting, geography), and other positive requirements (e.g., language proficiency, ability to comply with study procedures, capacity to consent, access to required technology or devices), typically expressed using terms such as “must have,” “must meet,” “only participants with,” or “eligible if.” Commonly appears as bullet points or prose under headings such as “Inclusion criteria,” “Eligibility criteria,” or “Participants” in trial registries, protocols, and manuscripts." },
                    { name: "Eligibility – exclusion criteria", definition: "All explicitly stated conditions or characteristics that disqualify a participant from enrollment or continued participation in the study. This includes safety-related exclusions (e.g., contraindications, relevant comorbidities, allergies, pregnancy where specified as exclusion, severe organ dysfunction), protocol- or confounding-related exclusions (e.g., concurrent trial participation, use of prohibited concomitant treatments, prior exposure to the study intervention beyond allowed limits), feasibility-related exclusions (e.g., inability or unwillingness to adhere to required visits, procedures, or follow-up), and history or condition-based exclusions indicated by phrases such as “must not,” “excluded if,” “no history of,” or “not eligible if.” Commonly appears under “Exclusion criteria” or within “Eligibility”/“Participants” sections in trial registries, protocols, and manuscripts." },
                    { name: "Intervention/treatment and control/placebo", definition: "The detailed specification of each study arm, including both experimental and comparator arms. This encompasses the name or identity of the intervention (e.g., drug, biologic, device, surgical procedure, behavioural or psychological program, digital/app-based intervention), the dose or intensity (e.g., strength, amount, frequency, duration of sessions), route or mode of administration (e.g., oral, IV, subcutaneous, inhaled, implanted, online, in-person, telephone), schedule and timing (including dosing schedules, titration, tapering, timing relative to baseline or randomisation), planned duration of administration, and a clear description of the control or comparator condition, such as placebo (including route and appearance), sham procedure, active control (with its own dose and schedule), or usual/standard care (with defining elements). It also includes any mandated co-interventions (e.g., background therapy) and any explicitly prohibited concomitant treatments. Commonly appears under headings such as “Interventions,” “Study treatments,” “Arm description,” “Trial arms,” or “Treatment protocol.”" },
                    { name: "Ethical approval – number", definition: "The specific identifier(s) associated with ethical approval of the study, including ethics committee/IRB/REC/REB approval numbers, protocol codes linked to ethics approval, national or institutional reference numbers, and identifiers for substantial amendments where explicitly reported. These identifiers are typically presented in sections or statements labelled “Ethics approval,” “Ethical considerations,” “IRB (Institutional Review Board) approval,” or similar, and may appear in trial registries, protocols, and manuscripts." },
                    { name: "Ethical approval – committee", definition: "The official name(s) of the ethics body or bodies responsible for reviewing and approving the study, such as Institutional Review Boards (IRBs), Research Ethics Committees (RECs), Research Ethics Boards (REBs), hospital or university ethics committees, and national or regional competent authorities performing ethics review. Names often include the institution and country or region (e.g., “University X Research Ethics Committee, Country Y”) and are typically reported in ethics-related sections or statements in trial registries, protocols, and manuscripts." },
                    { name: "Ethical approval – date", definition: "The calendar date on which ethics approval was granted for the primary study protocol, and, where explicitly reported, the dates on which major amendments were approved. Dates may be presented in full (e.g., “15 March 2022”) or in partial formats (e.g., month/year) and are usually mentioned alongside the ethics committee name or approval number in sections titled “Ethics approval,” “Ethical considerations,” or similar in trial registries, protocols, and manuscripts." },
                    { name: "Sample size", definition: "The numerical specification of the number of participants associated with the study, including planned or target total sample size, planned or target per-arm sample size (where reported), and actual numbers enrolled, randomized, allocated, or analyzed, overall and by arm (where reported). Expressions may include phrases such as “target sample size,” “planned enrollment,” “anticipated enrollment,” “a total of N participants,” “we aimed to recruit,” or “N participants were randomized/analyzed,” and typically appear in “Sample size,” “Methods,” “Participants,” “Study design,” “Statistical methods,” enrollment fields, or CONSORT-style flow diagrams." },
                    { name: "Date recruitment started", definition: "The planned or actual date on which enrollment of the first participant began. This may be expressed as an exact date (e.g., “Recruitment started on 01 June 2021”) or a less specific time reference (e.g., “Recruitment commenced in June 2021”) and is commonly found in fields or text labelled “Study start date,” “Date of first enrollment,” “Recruitment,” or within “Methods” and “Study design” sections in trial registries, protocols, and manuscripts." },
                    { name: "Outcomes – primary", definition: "All outcome measures explicitly designated as “primary,” “main,” or “primary endpoint(s)” that define the principal endpoints of interest for evaluating the intervention. A well-specified primary outcome typically includes: (a) the outcome domain or construct (e.g., overall survival, hospitalization, pain, HbA1c, depression severity), (b) the measurement instrument or operational definition (e.g., named clinical scale, laboratory test, imaging modality, diagnostic or response criteria, event definition algorithm), (c) the metric or scale (e.g., absolute value, change from baseline, proportion meeting a threshold, time-to-event, rate, composite index, including units where applicable), and (d) the assessment timepoint(s) (e.g., at 12 weeks, day 28, at discharge, 6 months post-randomisation). Such information is typically reported in “Primary outcome(s)” or “Endpoints” fields, “Outcomes” or “Endpoints” sections, abstracts, and statistical analysis descriptions. Some primary outcomes may be composites of other primary or secondary outcomes." },
                    { name: "Outcomes – secondary", definition: "All outcome measures explicitly labelled as “secondary,” “key secondary,” or similar that are intended to provide supportive, exploratory, mechanistic, or safety-related information beyond the primary endpoints. A well-specified secondary outcome similarly includes: (a) the outcome domain or construct, (b) the measurement instrument or operational definition, (c) the metric or scale, and (d) the assessment timepoint(s). These outcomes may include safety endpoints, adverse events, laboratory parameters, quality of life, functional status, biomarker changes, intermediate clinical outcomes, or subgroup-specific effects, and are typically documented in “Secondary outcome(s)” fields and “Secondary outcomes” or “Endpoints” sections. Some secondary outcomes may be composites of other primary or secondary outcomes." },
                    { name: "Method of randomisation and allocation", definition: "The described method used to generate and implement the allocation of participants or clusters to study arms. This includes the sequence generation approach (e.g., computer-generated random numbers, random-number tables, permuted block randomisation with or without specified block sizes, stratified randomisation by site or baseline characteristics, minimiz(s)ation or dynamic allocation procedures), the allocation ratio between arms (e.g., 1:1, 2:1, 3:1:1), and, where described in the same context, the allocation concealment mechanism (e.g., central randomisation, secure web- or telephone-based systems such as IWRS/IVRS, pharmacy-controlled allocation, sealed opaque sequentially numbered envelopes). Common textual indicators include phrases such as “participants were randomiz(s)ed using…,” “random sequence generated by…,” “block size…,” “stratified by…,” and “allocation was concealed using…,” and this information typically appears in “Randomiz(s)ation,” “Study design,” or “Methods” sections and corresponding fields in trial documentation." }
                ]
            },
            {
                key: "economics", label: "Economics", meta: "10 dimensions", glyph: "trend",
                dims: [
                    { name: "Research question & hypotheses", definition: "The central question or claim the study is designed to address and the specific predictions it sets out to test, as stated prior to data collection. This includes the direction of each predicted effect where applicable, the distinction between confirmatory (pre-specified, hypothesis-testing) and exploratory (hypothesis-generating) analyses, and any primary versus secondary hypotheses. Note whether the questions and predictions tested in the paper match those registered in the pre-analysis plan, and whether any were added, dropped, or reframed after seeing the data. Typically stated in the introduction, 'Research questions', 'Hypotheses', or a pre-analysis-plan summary." },
                    { name: "Data & sample", definition: "The data underlying the analysis and how the analysis sample is built. This includes the dataset(s) or data source (e.g., administrative records, survey, experiment, panel, archival/secondary data), the unit of observation (individual, household, firm, region, country-year), the time period and geographic scope, and the sample-construction rules (inclusion/exclusion filters, trimming, the population frame, and for primary data the collection procedure and target sample). Note any difference between the planned sample and the realised analysis sample. Typically reported in 'Data', 'Sample', 'Setting', or the early 'Methods'/'Empirical strategy' sections." },
                    { name: "Identification & research design", definition: "The research design used to identify the effect of interest and the assumptions it relies on. This includes the design type (randomised controlled trial / field experiment, instrumental variables, difference-in-differences, regression discontinuity, event study, synthetic control, matching, panel/fixed-effects, or a structural model), the source of identifying variation, the unit or level of randomisation or treatment assignment, and the key identifying assumptions (e.g., exclusion restriction and instrument relevance, parallel trends, continuity at the threshold, conditional independence/selection on observables). Note whether the strategy and assumptions in the paper match the pre-specified design. Typically reported in 'Empirical strategy', 'Identification', 'Research design', or 'Methods'." },
                    { name: "Treatment/intervention & comparison", definition: "The treatment or causal variable of interest and the comparison against which its effect is estimated. For experiments this includes the treatment arm(s) and intervention content (what was delivered, dose/intensity, delivery mode, timing, and duration), the control or comparison arm, the randomisation ratio, and expected take-up/compliance. For observational designs this includes the definition of the treatment/exposure variable, any instrument(s) and the exclusion restriction invoked, and the comparison group or counterfactual (e.g., not-yet-treated units, below-threshold units). Note any mandated co-interventions or excluded conditions. Typically reported in 'Intervention', 'Treatment', 'Experimental design', or 'Empirical strategy'." },
                    { name: "Outcome variables (primary & secondary)", definition: "The dependent variables used to evaluate the effect, distinguishing primary from secondary outcomes. A well-specified outcome includes (a) the construct or domain (e.g., earnings, employment, test scores, consumption, take-up), (b) the operational definition and data source, (c) how it is constructed and scaled (level, log, change, share, standardised z-score, an index of several measures, winsorised/top-coded), and (d) the measurement timing or horizon. Note any pre-specified primary outcome or index, whether the paper's primary outcome matches the registered one, and any outcomes reported but not pre-specified (or vice versa). Typically reported in 'Outcomes', 'Variables', 'Measurement', or the pre-analysis plan." },
                    { name: "Sample size, power & MDE", definition: "The basis for the study's statistical precision, as planned. This includes the target sample size (overall and per arm or cell), any a priori power calculation or minimum detectable effect (MDE), and the assumptions behind it (assumed effect size, baseline variance, significance level and power, intra-cluster correlation and number of clusters for clustered designs, and expected take-up/attrition). Note whether the realised sample and design achieve the planned precision and whether the power analysis was pre-specified. Typically reported in 'Power', 'Sample size', 'Experimental design', or the pre-analysis plan; it is often absent in observational work, which is itself worth noting." },
                    { name: "Estimation & specification", definition: "The estimating equation(s) and model used to obtain the main results, as pre-specified. This includes the estimator (e.g., OLS, 2SLS/IV, logit/probit, Poisson, maximum likelihood, GMM, local polynomial), the exact specification (outcome and treatment variables, control variables/covariates, fixed effects, interaction terms, functional form, and any weighting), the level of aggregation, and how the treatment-effect parameter is defined (intention-to-treat versus treatment-on-the-treated/LATE). Note any divergence between the planned and reported specification, including added or dropped controls or a changed functional form. Typically reported in 'Empirical strategy', 'Estimation', 'Specification', or 'Methods'." },
                    { name: "Inference & multiple-hypothesis testing", definition: "How statistical inference is conducted and how multiple comparisons are handled, as planned. This includes the method for computing standard errors and the level of clustering (e.g., heteroskedasticity-robust, cluster-robust at a stated level, HAC/Newey-West, wild-cluster bootstrap, randomisation/permutation inference), and the approach to multiple-hypothesis testing across outcomes, subgroups, or treatment arms (e.g., pre-specified outcome families, summary indices, family-wise error-rate or false-discovery-rate corrections, sharpened q-values). Note whether the inference and corrections in the paper match what was pre-specified. Typically reported in 'Inference', 'Standard errors', 'Statistical analysis', or notes to the main tables." },
                    { name: "Heterogeneity & subgroup analyses", definition: "Pre-specified analyses of how the effect varies across units or contexts. This includes the subgroups or moderators to be examined (e.g., by sex, age, baseline status, region, treatment intensity), the dimensions of treatment-effect heterogeneity of interest, and how these will be tested (interaction terms, split samples, or pre-specified machine-learning/causal-forest approaches), including any correction for the additional tests. Note whether the heterogeneity analyses in the paper were pre-registered or appear exploratory, and whether subgroups were added or dropped. Typically reported in 'Heterogeneity', 'Subgroup analysis', or the pre-analysis plan." },
                    { name: "Attrition, missing data & robustness", definition: "How the study handles incomplete data and shows that results are not artefacts of analytic choices, as planned. This includes the treatment of attrition and non-response (e.g., differential-attrition tests, bounds such as Lee bounds, inverse-probability weighting), the handling of missing values and outliers (imputation, winsorising, trimming), and the set of pre-specified robustness and sensitivity checks (alternative specifications, alternative samples or RDD bandwidths, placebo/falsification tests, controlling for additional covariates). Note any divergence between planned and reported handling, and any robustness checks added after the fact. Typically reported in 'Robustness', 'Sensitivity analysis', 'Attrition', or appendices." }
                ]
            },
            {
                key: "preclinical", label: "Preclinical / Animal", meta: "9 dimensions", glyph: "mouse",
                dims: [
                    { name: "Study type (exploratory vs. confirmatory)", definition: "Whether the study is described as exploratory (hypothesis-generating, with outcomes and analyses not fully pre-specified) or confirmatory (hypothesis-testing, with pre-specified predictions, outcomes, and analyses), and whether this designation is consistent between the protocol/preregistration and the publication. Confirmatory work states specific predictions before data collection and tests them with pre-specified analyses; exploratory work investigates patterns without prespecified hypotheses. Look for explicit statements (e.g., 'this study was exploratory/confirmatory', 'hypothesis-generating', 'pre-specified primary analysis') and for whether the paper retrospectively reframes exploratory analyses as confirmatory (or vice versa). Commonly indicated in the title, abstract, 'Study design', or 'Statistical analysis' sections." },
                    { name: "Total number of animals", definition: "The total number of animals planned for the study, the sum across all experimental and control groups as stated in the protocol, and, where reported, the number of animals actually used, randomised, or analysed overall. May be described as total sample size, number of animals, total N, planned cohort size, or the estimated number of animals declared to an ethics or animal-welfare body. The experimental unit (animal, litter, cage, or cohort) should be considered, and discrepancies may arise from attrition, mortality, humane endpoints, or exclusions; note any difference between the planned total and the number ultimately analysed. Typically reported in 'Animals', 'Sample size', 'Methods', ethics/licence sections, or a study-flow diagram." },
                    { name: "Number of animals per group", definition: "The planned number of animals allocated to each experimental and control group (group size, per-group sample size, n per arm/condition) as stated in the protocol, and, where reported, the actual numbers per group. This is often justified by an a priori power or sample-size calculation (assumed effect size, variability, alpha, power) or by a resource-equation/3Rs rationale. Note any divergence between planned and realised group sizes and whether the experimental unit is the individual animal or a higher-level unit (litter, cage). Typically reported in 'Animals', 'Sample size', 'Experimental design', or 'Statistical methods' sections, or in figure legends." },
                    { name: "Intervention and control", definition: "The specific intervention(s) planned and the control or comparator condition against which they are assessed. For the intervention this includes its identity (e.g., compound/drug, biologic, genetic or surgical manipulation, dietary or behavioural manipulation, device), the dose or intensity (amount, concentration, frequency), the route or mode of administration (e.g., oral gavage, intraperitoneal, intravenous, subcutaneous, inhaled, surgical), and the schedule, timing, and planned duration of administration. For the control this includes the comparator condition (e.g., vehicle, sham surgery, untreated, wild-type or littermate control, naive animals) specified with the same level of detail (route, volume, timing). It also includes any mandated co-interventions and explicitly prohibited concomitant treatments. Typically reported under 'Interventions', 'Treatments', 'Experimental groups', 'Study design', or 'Methods'." },
                    { name: "Measures to reduce bias", definition: "Methods planned to reduce the risk of bias in conducting and analysing the study. This includes randomisation of animals (or higher-level units) to groups and the sequence-generation method (e.g., computer-generated random numbers, block or stratified randomisation), allocation concealment, and blinding/masking of those administering interventions, caring for animals, assessing outcomes, and analysing data. It also covers other bias-mitigation steps such as pre-specified inclusion/exclusion rules for animals or data points, randomised order of treatment and measurement, and steps to control for cage, litter, batch, or experimenter effects. Note whether each measure is stated as planned and whether the paper reports it as implemented. Commonly reported in 'Randomisation', 'Blinding', 'Experimental design', or 'Methods' sections (aligned with ARRIVE 2.0 items)." },
                    { name: "Primary outcomes", definition: "All outcome measures explicitly designated as primary, main, or the principal endpoint(s) used to evaluate the intervention. A well-specified primary outcome includes (a) the outcome domain or construct (e.g., tumour volume, infarct size, lesion area, survival, a behavioural readout, a physiological or molecular marker), (b) the measurement instrument or operational definition (e.g., named assay, imaging modality, behavioural test, histological scoring method), (c) the metric or scale (e.g., absolute value, change from baseline, proportion meeting a threshold, time-to-event, including units), and (d) the assessment timepoint(s) (e.g., at day 14, 6 weeks post-induction, terminal). Note whether the primary outcome named in the paper matches the one pre-specified in the protocol. Typically reported in 'Primary outcome', 'Outcomes', 'Endpoints', or the statistical-analysis description. Some primary outcomes may be composites of other outcomes." },
                    { name: "Secondary outcomes", definition: "All outcome measures explicitly labelled as secondary, additional, or exploratory that provide supportive, mechanistic, or safety/welfare-related information beyond the primary endpoint. A well-specified secondary outcome similarly includes (a) the outcome domain or construct, (b) the measurement instrument or operational definition, (c) the metric or scale, and (d) the assessment timepoint(s). These may include additional behavioural, physiological, biochemical, histological, or molecular measures, adverse events or tolerability/welfare indicators, and subgroup or time-course analyses. Note any secondary outcomes reported in the paper that were not pre-specified, or pre-specified ones not reported. Typically reported in 'Secondary outcomes', 'Outcomes', or 'Endpoints' sections." },
                    { name: "Statistical analyses", definition: "The statistical methods planned to test each hypothesis or estimate each effect, as specified before data collection. This includes the specific tests or models (e.g., t-test, ANOVA, mixed-effects/multilevel models, survival analysis, regression), the experimental unit used for analysis (animal, litter, cage), the handling of nested or repeated-measures data, covariates and factors included, planned corrections for multiple comparisons, the criterion for statistical significance, and rules for handling missing data, outliers, and excluded animals. Note any divergence between the planned analysis and the analysis reported, including added, removed, or changed tests. Typically reported in 'Statistical analysis', 'Data analysis', or 'Methods' sections." },
                    { name: "Hypotheses", definition: "The specific predictions the study set out to test, as stated prior to data collection, including their direction (e.g., which group is expected to differ and in which direction) and whether each is designated confirmatory or exploratory. A well-specified hypothesis links a predicted effect to the intervention and the primary outcome. Note whether the hypotheses tested in the paper match those registered in direction and operationalisation, and whether any predictions appear to have been added, dropped, or reversed after seeing the data. Typically reported in the introduction, 'Objectives/Aims', 'Hypotheses', or 'Study design' sections." }
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
            const base = defaultModeActive ? [1, 6, 7, 8] : [1, 2, 3, 4, 5, 6, 7, 8];
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
            // ChatGPT (OpenAI reasoning models) takes a reasoning effort; GPT-OSS runs via Groq without one.
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

        function fileExt(name) {
            const i = name.lastIndexOf(".");
            return i >= 0 ? name.slice(i).toLowerCase() : "";
        }

        // Reject unsupported file types at upload time (rather than failing the run).
        function validateFile(input) {
            if (!input.files || !input.files.length) return true;
            const allowed = (input.getAttribute("accept") || "")
                .split(",").map((s) => s.trim().toLowerCase()).filter(Boolean);
            const ext = fileExt(input.files[0].name);
            const dropzone = input.closest(".dropzone");
            if (allowed.length && allowed.indexOf(ext) === -1) {
                const display = fileDisplay(input);
                if (display) {
                    display.textContent = "Unsupported file (" + (ext || "no extension") + "). Use PDF, DOCX, or TXT.";
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
                if (parserSelect) parserSelect.value = "pymupdf";
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

        // Auto-advance to the next step when a choice is made, except when the
        // selection reveals more controls on the same step (reasoning effort,
        // experiment number). Native selects can't fire on re-picking the same
        // value, so the Back/Next buttons stay available as a fallback.
        function autoAdvance(select, stayWhen) {
            if (!select) return;
            select.addEventListener("change", function () {
                if (select.disabled) return;
                if (typeof stayWhen === "function" && stayWhen(select.value)) return;
                window.setTimeout(function () { goToNext(); }, 240);
            });
        }
        autoAdvance(parserSelect);
        autoAdvance(modelSelect, function (v) { return v === "openai"; });
        autoAdvance(reasoningEffortSelect);
        autoAdvance(appendSelect);
        autoAdvance(multipleExperimentsSelect, function (v) { return v === "yes"; });
        autoAdvance(comparisonSelect);

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
