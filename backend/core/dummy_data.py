from backend.models import User, DimensionComparisonResult, ComparisonResult, Divergence


user = User(
    name="Mr Dummy",
    hashed_password="$argon2id$v=19$m=65536,t=3,p=4$LDgp8yshjuMNSAe2km1vNA$e7ihLGpNiSzc7adq2Q4OZgvPa8pb/tQm0Y2jlwKQwO8",
    comparisons=[],
)
comparison = ComparisonResult(
    user=user,
    dimensions=[
        DimensionComparisonResult(
            dimension_name="Sample Size",
            dimension_description="The planned number of participants to be included in the study. This could include a precise number of units or observations that the researchers intend to sample or a range, minimum, or maximum. It could also include a stopping rule (that is, how the decision to terminate data collection will be made) if one is specified.",
            divergence=Divergence.YES,
            divergence_summary="""
The preregistration identifies 21,060 participants and precommits to reserving 10% of the total sample for code development, implying analyses on ~90% of the data. The paper instead reports 21,060 observations in the analytic sample and 23,413 unique individuals overall, without mentioning any 10% holdout. This creates two discrepancies: (1) the unit and count differ (registration: 21,060 participants; paper: 21,060 observations and 23,413 unique participants), and (2) the preregistered 10% training/holdout split is not reported or reflected in the paper’s analytic sample size. These inconsistencies suggest the paper may have deviated from the preregistered sample handling and counting, which could affect the effective sample size used for analyses and the independence/unit-of-analysis assumptions.
""",
            preregistration_quotes="""
"The data consisted of 21060 participants in total who completed and met the screening criteria for at least one measure in the overall study." "we reserved 10% of the total sample for exclusive use in the code development process (i.e., the “training” dataset)." "Sample The sample used for these analyses was taken from Bar-Anan & Nosek’s (2014) data, collected via the Project Implicit website."
""",
            paper_quotes="""
"The data used in our analytic sample, composed of participants who completed at least one measure in the overall study and met common accuracy and latency performance exclusion criteria (full details in supplementary materials), leading to 21060 observations in total (i.e., some participants may have completed more than one of the measures)." "A total of 23,413 unique individuals participated in this study (63% women, 36% men, 1% unknown; mean age = 29.1, SD = 12.0)." "Sample The sample used for these analyses was taken from Bar-Anan & Nosek’s (2014) data, collected via the Project Implicit website."
""",
        ),
        DimensionComparisonResult(
            dimension_name="Data source",
            dimension_description="Where the data will be / was obtained, and the processes therein. For example, a student sample, online (e.g., m-turk, prolific), the community, etc.",
            divergence=Divergence.NO,
            divergence_summary="""
The registration and paper both describe using an existing dataset collected via Project Implicit (Bar‑Anan & Nosek, 2014) and do not indicate any new participant recruitment. The paper further notes the data are openly available and that consent is not applicable due to secondary use. These statements are consistent with the preregistered data source.
""",
            preregistration_quotes="""
"The sample used for these analyses was taken from Bar-Anan & Nosek’s (2014) data, collected via the Project Implicit website. Detailed information regarding the collection of these data can be found in Bar-Anan and Nosek (2014)."
""",
            paper_quotes="""
"This study uses openly available data collected on Project Implicit (https://implicit.harvard.edu), originally collected by Bar-Anan and Nosek (2014; data available from osf.io/qf9jx)." "Detailed information regarding the collection of these data can be found in Bar-Anan and Nosek (2014)." "Consent to participate: Not applicable (use of existing, openly-available data)." "Additionally, it is certainly the norm in modern implicit measures research that data be collected via online samples; indeed, the data analysed here were from Project Implicit, which is the largest source of data for implicit measures in the field."
""",
        ),
        DimensionComparisonResult(
            dimension_name="Inclusion criteria",
            dimension_description="How subjects (or units of analysis otherwise specified) will be selected for eligibility in the study.",
            divergence=Divergence.NO,
            divergence_summary="""
Both documents define inclusion similarly: participants must have completed at least one measure and satisfy performance-based screening/exclusion criteria. The paper specifies these as common accuracy and latency criteria (details in the supplement), while the preregistration refers generally to screening criteria and points to preregistered code for specifics. Although the paper reports counts in terms of observations and the registration mentions participants, the eligibility criteria themselves are aligned; no substantive deviation is evident for inclusion criteria.
""",
            preregistration_quotes="""
"The data consisted of 21060 participants in total who completed and met the screening criteria for at least one measure in the overall study." "The sample used for these analyses was taken from Bar-Anan & Nosek’s (2014) data, collected via the Project Implicit website." "However, we also invite the reader to inspect our fully preregistered code for precise specifications on all aspects of the analyses."
""",
            paper_quotes="""
"The data used in our analytic sample, composed of participants who completed at least one measure in the overall study and met common accuracy and latency performance exclusion criteria (full details in supplementary materials), leading to 21060 observations in total (i.e., some participants may have completed more than one of the measures)."
""",
        ),
        DimensionComparisonResult(
            dimension_name="Exclusion criteria",
            dimension_description="How data exclusions will be determined (e.g., how will outliers be determined? Will awareness checks be implemented?).",
            divergence=Divergence.UNSURE,
            divergence_summary="""
The paper explicitly reports using common accuracy and latency performance exclusion criteria (details in the supplement), whereas the preregistration text only refers to generic screening criteria and a processing script without detailing specific exclusion rules. Because the preregistration excerpt does not enumerate the exclusion criteria, it is not possible to confirm whether the exclusions applied in the paper match the preregistered plan.
""",
            preregistration_quotes="""
"The data consisted of 21060 participants in total who completed and met the screening criteria for at least one measure in the overall study." "Other than for the purposes of data processing (i.e., running the processing.Rmd script to produce the full processed dataset before splitting it 10%/90%), we did not look at or run any analyses on the testing dataset prior to preregistration (i.e., the analyses.Rmd has been developed and run only on the training dataset)." "For this and all subsequent analyses, if proportions of 0 or 1 or variances of 0 were present they were offset by 0.001."
""",
            paper_quotes="""
"The data used in our analytic sample, composed of participants who completed at least one measure in the overall study and met common accuracy and latency performance exclusion criteria (full details in supplementary materials), leading to 21060 observations in total (i.e., some participants may have completed more than one of the measures)." "Data processing Scoring algorithm The implicit measures we compared typically use different methods and metrics for scoring."
""",
        ),
        DimensionComparisonResult(
            dimension_name="Incomplete and missing data",
            dimension_description="A description of how incomplete and/or missing data will be handled.",
            divergence=Divergence.UNSURE,
            divergence_summary="""
Both documents mention the same boundary adjustment (offsetting 0/1 proportions and zero variances by 0.001), which is consistent. The paper further implies an available-case approach by including participants who completed at least one measure, which the registration also mentions at a high level. However, neither the preregistration nor the paper provides explicit procedures for handling incomplete or missing data beyond this (e.g., item- or trial-level missingness, listwise vs. pairwise deletion, imputation). Because the preregistration lacks a clear plan and the paper does not provide sufficient detail, there is insufficient information to evaluate consistency on missing-data handling.
""",
            preregistration_quotes="""
"For this and all subsequent analyses, if proportions of 0 or 1 or variances of 0 were present they were offset by 0.001." "The data consisted of 21060 participants in total who completed and met the screening criteria for at least one measure in the overall study."
""",
            paper_quotes="""
"The data used in our analytic sample, composed of participants who completed at least one measure in the overall study and met common accuracy and latency performance exclusion criteria (full details in supplementary materials), leading to 21060 observations in total (i.e., some participants may have completed more than one of the measures)." "For this and all subsequent analyses, if proportions of 0 or 1 or variances of 0 were obtained, these values were offset by 0.001 in order to allow for meta-analysis." "There were minor deviations from the preregistration (for details see Supplementary Table 1)."
""",
        ),
        DimensionComparisonResult(
            dimension_name="Hypotheses",
            dimension_description="A prediction or set of predictions about the result(s) of a study.",
            divergence=Divergence.NO,
            divergence_summary="""
Both the preregistration and the paper frame the work around the same three primary questions: detectability of non-zero effects within individuals, discriminability between individuals, and CI coverage/width. Neither source articulates specific directional hypotheses favoring particular measures; the paper’s statement that “hypotheses” were preregistered appears to refer to these aims/questions. Given that the substantive expectations are non-directional and aligned across documents, there is no evident deviation regarding hypotheses.
""",
            preregistration_quotes="""
"We addressed three primary research questions in this study. We investigated, for each measure and across domains, the proportion of participants who demonstrated a detectable effect (i.e., whose scores defected from the neutral point)." "The first research question related to the relative ability of different measures to detect effects for individual participants." "95% CIs on individuals’ scores were used to assess whether each individual excluded the neutral point of zero effect on the task (i.e., PI = 0.50). Meta-analytic model The individual level proportions were logit transformed and entered into a similar linear mixed-effects model as the previous one: proportion_discriminable_logit ~ 1 + measure + (1 | domain), weights = 1/variance."
""",
            paper_quotes="""
"Preregistration information: The hypotheses and analysis plan/code were preregistered (https://osf.io/qk9ar) on 31/08/2022, prior to the commencement of the analysis of the data." "In this preregistered study, we specifically set out to determine (i) how well measures can detect non-zero effects within individuals; (ii) how well measures could discriminate between individuals, and (iii) the width of the range of scores that the confidence intervals of individual’s scores tended to cover." "The upper third of the plot shows the meta-analytic model for the proportion of participants whose scores differed detectably from zero; the middle third of the plot shows the meta-analytic model for the probability of detectable difference between two participants; and the lower third shows the meta-analytic model for the coverage of the confidence intervals."
""",
        ),
        DimensionComparisonResult(
            dimension_name="Manipulated variables",
            dimension_description="Refers to variables that are experimentally manipulated, that is, the experimenter controlled the level / treatment that the subject/ unit of analysis received.",
            divergence=Divergence.NO,
            divergence_summary="""
Both the preregistration and the paper indicate that there are no experimentally manipulated variables. The study uses archival Project Implicit data and compares outcomes across existing measures via meta-analytic models, treating measure as an analytic fixed effect rather than a manipulated treatment. Therefore, there is no deviation regarding manipulated variables (none were planned, and none were implemented).
""",
            preregistration_quotes="""
"proportion_diff_zero_logit ~ 1 + measure + (1 | domain), weights = 1/variance" "That is, we entered measure as a fixed effect in order to estimate differences between these specific measures (i.e., measures are an exhaustive set for our purposes). We investigated, for each measure and across domains, the proportion of participants who demonstrated a detectable effect (i.e., whose scores defected from the neutral point)." "Method Sample The sample used for these analyses was taken from Bar-Anan & Nosek’s (2014) data, collected via the Project Implicit website."
""",
            paper_quotes="""
"Meta-analytic model In order to compare the proportion of detectable effects between measures, the data from individuals was meta-analyzed." "Specifically, recall that Project Implicit uses the values of 0, 0.15, 0.35, and 0.65 to denote, no bias, small bias, moderate bias, and strong bias, respectively."
""",
        ),
        DimensionComparisonResult(
            dimension_name="Measured variables",
            dimension_description="""Variables that are observed/ recorded in a study. This will include outcome measures, as well as any measured predictors or covariates. Also related is how the variable(s) will be measured (e.g., "The primary outcome variable will be the perceived tastiness of the single brownie each participant will eat. We will measure this by asking participants 'How much did you enjoy eating the brownie' (on a scale of 1-7, 1 being 'not at all', 7 being 'a great deal')."). If authors mention that measures will be combined into an index/ composite (or a mean), this is also relevant.""",
            divergence=Divergence.UNSURE,
            divergence_summary="""
Measured variables across preregistration and paper largely align (PI scoring; measure and domain factors). The paper explicitly reports CI width/coverage, which the preregistration excerpts only imply, leaving some uncertainty about complete preregistered coverage.
""",
            preregistration_quotes="""
"The Wilkinson notation for the model was as follows: proportion_diff_zero_logit ~ 1 + measure + (1 | domain), weights = 1/variance That is, we entered measure as a fixed effect in order to estimate differences between these specific measures (i.e., measures are an exhaustive set for our purposes)." "We investigated, for each measure and across domains, the proportion of participants who demonstrated a detectable effect (i.e., whose scores defected from the neutral point)." "Calculation of scores 95% CIs on individuals’ scores were used to assess whether each individual excluded the neutral point of zero effect on the task (i.e., PI = 0.50)."
""",
            paper_quotes="""
"Measures For more detailed descriptions, see Bar-Anan and Nosek (2014) and the associated references provided under each measure. In this preregistered study, we specifically set out to determine (i) how well measures can detect non-zero effects within individuals; (ii) how well measures could discriminate between individuals, and (iii) the width of the range of scores that the confidence intervals of individual’s scores tended to cover." "PI scores also provide a standardized method of scoring data from tasks that are typically derived from different properties of participants' responses (e.g., accuracy, response times), providing an ideal scoring method to compare multiple measures (see also Cummins et al., 2021)." "Proportion of effects detectable from zero effect"
""",
        ),
        DimensionComparisonResult(
            dimension_name="Transformations",
            dimension_description="Steps, decisions, or approaches that relate to processing data from one form, structure, or scale to another to prepare it for analysis (e.g., log transforming, centering, or recoding the data).",
            divergence=Divergence.YES,
            divergence_summary="""
Both sources apply the 0.001 boundary adjustment. Only the preregistration explicitly specifies logit-transforming and back-transforming proportions; the paper omits these transformation details, indicating a possible deviation or unreported step.
""",
            preregistration_quotes="""
"Results were back transformed from logits to proportions for plotting and reporting." "For this and all subsequent analyses, if proportions of 0 or 1 or variances of 0 were present they were offset by 0.001."
""",
            paper_quotes="""
"proportion_discriminable ~ 1 + measure + (1 | domain), weights = 1/variance" "For this and all subsequent analyses, if proportions of 0 or 1 or variances of 0 were obtained, these values were offset by 0.001 in order to allow for meta-analysis."
""",
        ),
        DimensionComparisonResult(
            dimension_name="Statistical models",
            dimension_description="The statistical model(s) used to test the hypotheses. This includes the type of model (e.g. ANOVA, RMANOVA, MANOVA, multiple regression, SEM, etc) and the specification of the model. It also includes each variable that will be included, all interactions, subgroup analyses, pairwise or complex contrasts, and any follow-up tests from omnibus tests. If transformations are planned or reported (e.g., log transforming, centering, or recoding the data) this information should also be extracted. If any inference criteria are specified (e.g., alpha thresholds, bayes factors, specific model fit indices or other cut-off criteria), that should also be extracted. This includes details about using one-tailed vs. two-tailed tests.",
            divergence=Divergence.YES,
            divergence_summary="""
Core structure aligns: both sources describe inverse-variance-weighted linear mixed-effects meta-analytic models with measure as a fixed effect, domain as a random intercept, implemented in lme4, and both apply a 0.001 offset for boundary values. However, there are notable deviations/omissions: (1) Transformation/link: the preregistration explicitly commits to logit-transforming all proportion outcomes and back-transforming results, whereas the paper presents formulas and text on the raw proportion scale (no mention of logit or back-transformation), suggesting either a change in the link/scale or an undeclared reporting omission. (2) Planned pairwise comparisons between measures are specified in the preregistration but are not reported in the quoted paper text. (3) The preregistration states that results would be based on the 90% testing dataset after a holdout split; the paper’s model description does not mention this split. These discrepancies indicate deviations in the reported statistical modeling and inferential procedures relative to the preregistered plan.
""",
            preregistration_quotes="""
                "Meta-analytic model In order to compare the proportion of detectable effects between measures, the data from individuals was transformed and meta-analyzed. Meta-analytic model The individual level proportions were logit transformed and entered into a similar linear mixed-effects model as the previous one: proportion_discriminable_logit ~ 1 + measure + (1 | domain), weights = 1/variance Results were back transformed from logits to proportions for plotting and reporting. Meta-analytic model The proportions were logit transformed and entered into a similar linear mixed-effects model as the previous two: ci_width_proportion_mean_logit ~ 1 + measure + (1 | domain), weights = 1/variance Results were back transformed from logits to proportions for plotting and reporting. The Wilkinson notation for the model was as follows: proportion_diff_zero_logit ~ 1 + measure + (1 | domain), weights = 1/variance That is, we entered measure as a fixed effect in order to estimate differences between these specific measures (i.e., measures are an exhaustive set for our purposes). We then logit transformed this proportion and entered it into a linear mixed-effects model using the R package lme4. We weighted by inverse variance, as is common in meta-analytic models. Results reported below are based on analyses for the 90% testing dataset only. Analytic Plan Below, we briefly describe our data processing and analysis steps. Results from the forest plot (i.e., the meta-analytic estimates) were interpreted with the aid of pairwise comparisons between the measures. For each measure and domain, we calculated the proportion of detectable effects and its variance. ... For this and all subsequent analyses, if proportions of 0 or 1 or variances of 0 were present they were offset by 0.001."
                """,
            paper_quotes="""
"Meta-analytic model In order to compare the proportion of detectable effects between measures, the data from individuals was meta-analyzed. The upper third of the plot shows the meta-analytic model for the proportion of participants whose scores differed detectably from zero; the middle third of the plot shows the meta-analytic model for the probability of detectable difference between two participants; and the lower third shows the meta-analytic model for the coverage of the confidence intervals." "Meta-analytic model The individual level proportions were entered into a similar linear mixed-effects model to the previous one: proportion_discriminable ~ 1 + measure + (1 | domain), weights = 1/variance" "The Wilkinson notation for the model was as follows: proportion_diff_zero ~ 1 + measure + (1 | domain), weights = 1/variance That is, we entered measure as a fixed effect in order to estimate the proportions for each measure and make inferences about differences between them (i.e., measures are an exhaustive set for our purposes)." "Meta-analytic model The proportions were entered into a similar linear mixed-effects model to the previous two: ci_width_proportion_mean ~ 1 + measure + (1 | domain), weights = 1/variance" "We then entered the proportions into a linear mixed-effects model using the R package lme4 (Bates et al., 2015)." "For this and all subsequent analyses, if proportions of 0 or 1 or variances of 0 were obtained, these values were offset by 0.001 in order to allow for meta-analysis."
""",
        ),
    ],
)
