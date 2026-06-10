/* Discipline default dimension sets for RegCheck comparison */
(function(){
  let _id = 0;
  const D = (name, definition) => ({ id: ++_id, name, definition });

  const DISCIPLINES = [
    {
      key: "psychology",
      label: "Psychology",
      meta: "7 dimensions",
      glyph: "brain",
      dims: [
        D("Sample size",
          "The planned number of participants to be included in the study. This could include a precise number of units or observations that the researchers intend to sample, or a range, minimum, or maximum. It could also include a stopping rule (that is, how the decision to terminate data collection will be made) if one is specified."),
        D("Data source",
          "Where the data will be / was obtained, and the processes therein. For example, a student sample, online (e.g., m-turk, prolific), the community, an existing archival dataset, etc."),
        D("Hypotheses",
          "The specific, directional predictions the study set out to test, as stated prior to data collection — including which effects are confirmatory versus exploratory."),
        D("Outcome measures",
          "The primary dependent variables or instruments used to operationalise the constructs of interest, including how each is scored."),
        D("Analysis plan",
          "The statistical models, tests, and any covariates specified to evaluate each hypothesis, including corrections for multiple comparisons."),
        D("Exclusion criteria",
          "Rules determining which participants, trials, or observations are removed before analysis (e.g., attention checks, outliers, incomplete responses)."),
        D("Conditions & assignment",
          "The experimental conditions and the procedure used to assign participants to them (e.g., random assignment, counterbalancing).")
      ]
    },
    {
      key: "clinical",
      label: "Clinical / Medical",
      meta: "7 dimensions",
      glyph: "pulse",
      dims: [
        D("Sample size & power",
          "The target enrolment and the power analysis justifying it, including the assumed effect size, alpha, and power."),
        D("Eligibility criteria",
          "Inclusion and exclusion criteria defining the study population, including age, diagnosis, and comorbidity constraints."),
        D("Intervention & comparator",
          "The treatment, dose, route, and schedule, together with the comparator or control arm."),
        D("Primary endpoint",
          "The principal outcome used to judge efficacy, with its precise definition and measurement timepoint."),
        D("Secondary endpoints",
          "Additional pre-specified outcomes that support or contextualise the primary analysis."),
        D("Blinding",
          "Who is masked to allocation — participants, providers, outcome assessors — and the method used to maintain it."),
        D("Analysis plan",
          "The pre-specified statistical methods, analysis population (ITT / per-protocol), and handling of missing data.")
      ]
    },
    {
      key: "economics",
      label: "Economics",
      meta: "7 dimensions",
      glyph: "trend",
      dims: [
        D("Data & sample",
          "The dataset, unit of observation, time period, and how the analysis sample is constructed and restricted."),
        D("Identification strategy",
          "The research design used to identify the causal effect (RCT, IV, difference-in-differences, RDD, etc.) and the assumptions it relies on."),
        D("Treatment & instruments",
          "The treatment variable and any instruments used, including the exclusion restriction being invoked."),
        D("Outcome variables",
          "The dependent variables of interest and how each is measured or constructed."),
        D("Estimation method",
          "The estimator and specification, including fixed effects, controls, and functional form."),
        D("Inference & standard errors",
          "How statistical inference is conducted, including clustering, robust corrections, and any multiple-testing adjustments."),
        D("Robustness checks",
          "Pre-specified alternative specifications, placebo tests, and sensitivity analyses.")
      ]
    },
    {
      key: "social",
      label: "Social science (general)",
      meta: "5 dimensions",
      glyph: "globe",
      dims: [
        D("Research question",
          "The central question or claim the study is designed to address, framed as it was prior to data collection."),
        D("Sample & data",
          "The population, sampling frame, and data collection procedure, including target sample size."),
        D("Key variables",
          "The independent and dependent variables, and how each construct is measured."),
        D("Analysis plan",
          "The analytic approach and models specified to answer the research question."),
        D("Exclusions & robustness",
          "Pre-specified data exclusions and any sensitivity or robustness checks.")
      ]
    }
  ];

  // clone helper so editing one discipline's defaults doesn't mutate the source
  function loadDiscipline(key){
    const d = DISCIPLINES.find(x => x.key === key) || DISCIPLINES[0];
    let n = 100000;
    return d.dims.map(x => ({ id: ++n + Math.random(), name: x.name, definition: x.definition }));
  }

  window.RC_DISCIPLINES = DISCIPLINES;
  window.RC_loadDiscipline = loadDiscipline;
  window.RC_newDim = () => ({ id: Date.now() + Math.random(), name: "", definition: "" });
})();
