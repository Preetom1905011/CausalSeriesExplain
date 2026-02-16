PROMPT_INIT = r"""
Prompt:

You are given several input-output examples from a timeseries simulation model. Each example consists of a multivariate input timeseries X_t and a univariate daily output timeseries Y.

All example inputs generate the same output Y. The relationship between inputs and outputs is direct and may include temporal lag/delay effects. The baseline output Y spans {T} days. The input X_t is defined over interval length {DELTA}, so X_t contains ⌈ {T} / {DELTA} ⌉ interval values.

Your goal is to observe the input-output relationships from the examples and extract symbolic causal rules of the form:

"If trends in selected components of X_t persist over a given time interval, then the output Y will exhibit a specific phase after a delay."

Please divide the baseline output into specific output phases listed below in order of timing and infer the multivariate input trends that are responsible. Each output phase should be unique (no duplicate entries in the output column), and the rules should provide input and corresponding output trend behavior for the total time window (i.e., all input windows should concatenate to cover from T=0 to T=100). Input time windows must be multiples of {DELTA}.

Fill in the table below with symbolic causal rules. Each rule should specify:

  - Input Trend Window: the time interval where a specific trend in selected components of X_t is observed.
  - Trend Description: qualitative trends over that window.
  - Output Phase Window: the time interval where a specific phase in Y occurs, as a delayed effect of the input.
  - Phase Description: the qualitative behavior of Y.
  - Symbolic Rule: a formal statement capturing this as a logical implication.

Use the vocabulary:

  - Trends: Low, Moderate, HighStable, SharpSpike, SlowRise, DipThenRise, LowModerate
  - Output Phases: InitialGrowth, PeakFormation, PostPeak, Decline, Resurgence

Below is the list of example input-output points and Baseline Expected Output:
---- Examples------
{X_t_Y_t_multivariate_contexts}
---------------------
Baseline Y:  {BASELINE_OUTPUT}

Table format:
Rule ID | Input Trend Window | X_s Trend | X_a Trend | Output Phase Window | Y Phase | Symbolic Rule

Example:
R1 | [0, 15] | Low | SlowRise | [0, 20] | PeakFormation | Box_{[0,15]}(Low(X_s) AND SlowRise(X_a)) -> Diamond_{[0,20]}PeakFormation(Y)
R2 | [15, 30] | Low | Low | [20, 35] | InitialGrowth | Box_{[15,30]}(Low(X_s) AND Low(X_a)) -> Diamond_{[20,35]}InitialGrowth(Y)

Make sure to use the table format shown above. Do not use domain-specific terminology; instead, describe timeseries patterns using abstract trend labels. Ensure the symbolic rule column uses the logical form:

Box_{[t1,t2]}(Trend(X_{t,i}) AND Trend(X_{t,j}) AND ...) -> Diamond_{[t3,t4]}Phase(Y)

Ensure:
  - All time windows are valid.
  - Input windows are multiples of {DELTA}.
  - Output phases are unique and cover the full timeline.
  - The symbolic rule strictly follows the required logical template.
  - Only abstract trend labels are used.
"""

PROMPT_INPUT_GEN = r"""

Prompt: You are given several input-output examples from a timeseries simulation model and a causal ruleset describing input trends and their corresponding output effects.

Each example consists of:
  - A multivariate input timeseries X_t defined over interval length {DELTA}
  - A daily output timeseries Y

All example inputs generate the same output Y. The relationship between inputs and outputs is direct and may include temporal lag/delay effects. The baseline output Y spans {T} days. The input X_t is defined over interval length {DELTA}, so X_t contains ⌈ {T} / {DELTA} ⌉ interval values.

Input constraints:
  - All X_{t,1} values must lie in the range {INSERT_RANGE_1}
  - All X_{t,2} values must lie in the range {INSERT_RANGE_2}

You must use the provided ruleset to generate a novel multivariate input sequence X_t that is different from all example inputs but produces an output Y that closely follows the baseline Y trend.

Instructions:

  1. Analyze the baseline Y trend and the provided ruleset.
  2. Divide the baseline Y into output phases.
  3. For each output phase:
     - Identify the corresponding rule in the ruleset.
     - Determine the required input trends.
  4. Generate a new multivariate input sequence X_t:
     - Length must match the number of intervals determined by {DELTA}.
     - Trends must be consistent with the ruleset.
     - The full timeline must be covered.
     - The trend pattern must differ from all previous example inputs.
     - All input values must satisfy the specified bounds.
  5. For each output phase, briefly justify your input choices for the corresponding input intervals, referencing:
     - The applied rule
     - The delayed and {DIRECT_OR_INVERSE} relationship between inputs and output

Below is the list of example input-output points and Baseline Expected Output:
{Clustering Context}
---------------------
Baseline Y: {BASELINE_OUTPUT}
Rules Table: {RULES_TABLE}

Required Output Format:

  1) Generated Input Series: Provide the full multivariate input sequence:
     - X_{t,1}: [v_1, v_2, ..., v_K], X_{t,2}: [v_1, v_2, ..., v_K], ...

  2) Phase-wise Justification: For each output phase, provide:
     - Output Phase: PhaseName,
     - Output Window: [t_3, t_4],
     - Applied Rule: RuleID,
     - Required Input Trend Window: [t_1, t_2],
     - Assigned Input Trends: X_{t,i}: Trend
     - Justification: Brief explanation referencing the rule and the delayed/direct or inverse relationship.

Ensure:
  - The generated input trends differ structurally from all example inputs.
  - Input intervals are multiples of {DELTA}.
  - The ruleset is followed consistently.
  - The entire timeline is covered.
  - All input values lie within the specified range.
  - The resulting Y closely follows the baseline Y.
"""

PROMPT_REFINE = r"""
Prompt:

You previously generated a multivariate input series X_t' using a causal ruleset, simulated the system, and obtained an output Y'. The simulated output Y' does not closely match the Baseline Y.

The relationship between inputs and output is {DIRECT_OR_INVERSE} and delayed. This is critical and must guide your diagnosis and rule updates.

Generated Input:
X_t':
Your generated weekly input timeseries:
{INSERT_X_T_SIM}

corresponding Y':
{INSERT_Y_SIM}
---------------------
Baseline Y: {INSERT_Y_BASE}
Previous Rules Table: {PREVIOUS_RULES_TABLE}

Instructions:

  1. Divide the Baseline Y into distinct output phases in chronological order.
  2. Compare Y' with Baseline Y and identify mismatches, including:
     - Timing shifts, magnitude differences, shape differences (peaks, valleys, slopes), and phase misalignment.
  3. Diagnose which input trends in X_t' caused the mismatch, considering whether they were too high or too low, too early or too late, too sharp or too flat, not sustained long enough, or incorrectly capturing the direct or delayed relationship.
  4. Update the ruleset to better guide future input generation.

Constraints for Updated Rules:

  - Each output phase must be unique.
  - Input windows must be multiples of {DELTA}.
  - Input windows must concatenate to cover the full timeline (T = 0 to T = {T}).
  - Use only abstract trend labels.
  - Do not use domain-specific terminology.
  - When appropriate, refine only the input variable most responsible for a mismatch.

Allowed Vocabulary:
  - Input Trends: Low, Moderate, HighStable, SharpSpike, SlowRise, DipThenRise, LowModerate
  - Output Phases: InitialGrowth, PeakFormation, PostPeak, Decline, Resurgence

Required Output Format:
Return the updated rules in the following Markdown table format:

Rule ID | Input Trend Window | X_s Trend | X_a Trend | Output Phase Window | Y Phase | Symbolic Rule

Example:
R1 | [t1,t2] | Low | SlowRise | [t3,t4] | PeakFormation | Box_{[t1,t2]}(Low(X_{t,1}) AND SlowRise(X_{t,2})) -> Diamond_{[t3,t4]}PeakFormation(Y)

Symbolic Rule Format:
Each rule must follow:

Box_{[t1,t2]}(Trend(X_{t,i}) AND Trend(X_{t,j}) AND ...) -> Diamond_{[t3,t4]}Phase(Y)

Ensure:
  - Rules better explain the mismatch between Y' and Baseline Y.
  - The direct and delayed relationship must be properly encoded.
  - The updated rules should improve future alignment with Baseline Y.
"""
