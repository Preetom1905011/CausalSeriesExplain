# Prompt: Initial Ruleset Generation

## Goal
Generate initial causal rules \(R_0\) from clustering context.

## Prompt Description
You are given several input-output examples from a time-series simulation model. Each example consists of two weekly input time-series `X_s` and `X_a`, and a daily output time series `Y`. For example: if `Y` is for 100 days, `X_s` and `X_a` will have 15 values. The relationship between inputs and outputs is inverse and contains temporal lag/delay.

The goal is to observe the input-output relations from the examples and extract causal rules of the form:
> If trends `X_s` and `X_a` persist over some time interval, then the output `Y` will exhibit a specific phase with a delay.

## Instructions
Divide the baseline output into specific output phases listed below in order of timing and infer the input trends that are responsible. Each output phase should be unique (no duplicate entries in output column) and the rules should cover the full time window from `T=0` to `T=100`. The input time window interval should be a multiple of 7.

Each rule must specify:
- **Input Trend Window**: the time interval where a specific trend in `X_s` and `X_a` is observed.
- **Trend Description**: qualitative trends over that window.
- **Output Phase Window**: the time interval where a specific phase in `Y` occurs, as a delayed effect of the input.
- **Phase Description**: the qualitative behavior of `Y`.
- **Symbolic Rule**: a formal logical implication capturing the relation.

### Trend Vocabulary
- `Low`, `Moderate`, `HighStable`, `SharpSpike`, `SlowRise`, `DipThenRise`, `LowModerate`

### Output Phases
- `InitialGrowth`, `PeakFormation`, `PostPeak`, `Decline`, `Resurgence`

---

## Example Input-Output Points and Baseline Output

```
---- Example 0 ------
X_s: [0.018, 0.14, ... K values]
X_a: [0.06, 0.23, ... K values]
Y:   [15, 12, ... T values]

---- Example 1 ------
X_s: [0.025, 0.083, ... K values]
X_a: [0.05, 0.19, ... K values]
Y:   [15, 12, ... T values]

<Insert More ...>

---------------------
Baseline Y: [15, 12, ... T values]
```

---

## Table Format

| Rule ID | Input Trend Window [t1, t2] | X_s Trend | X_a Trend | Output Phase Window [t3, t4] | Y Phase        | Symbolic Rule |
|---------|------------------------------|------------|------------|-------------------------------|----------------|---------------|
| R1      | [5, 15]                      | Low        | SlowRise   | [20, 30]                      | PeakFormation  | □[5,15](Low(X_s) ∧ SlowRise(X_a)) → ◇[20,30]PeakFormation(Y) |
| R2      | [0, 10]                      | Low        | Low        | [10, 20]                      | InitialGrowth  | □[0,10](Low(X_s) ∧ Low(X_a)) → ◇[10,20]InitialGrowth(Y)      |
| R3      | [30, 40]                     | SharpSpike | Moderate   | [45, 50]                      | PostPeak       | □[30,40](SharpSpike(X_s) ∧ Moderate(X_a)) → ◇[45,50]PostPeak(Y) |

---

## JSON Schema Representation

```json
{
  "RuleID": "R1",
  "InputTrendWindow": [5, 15],
  "Trends": {"X_s": "Low", "X_a": "SlowRise"},
  "OutputPhaseWindow": [20, 30],
  "Y_Phase": "PeakFormation",
  "SymbolicRule": "□[5,15](Low(X_s) ∧ SlowRise(X_a)) → ◇[20,30]PeakFormation(Y)"
}
```

> ⚠️ Avoid using domain-specific terminology. Use abstract labels to describe time-series trends.



### Goal: Novel input set $(X_{s}', X_{a}')$ generation using ruleset $\mathcal{R}_i$

**Prompt:**

You are given several input-output examples from a time-series simulation model and a causal ruleset describing input trends and corresponding output effect. Each example consists of two weekly input time-series $X_s$ and $X_a$, and a daily output time series $Y$. All example inputs generate the same $Y$. The relationship between inputs and outputs are inverse and contains temporal lag/delay. You will use these to generate novel input sequence for a baseline output. the baseline output is for 100 days, so both $X_a$ and $X_s$ will have 15 values. The possible range of the two input lists are (0.001, 0.20). Divide and identify the output trends into chunks and map which output trend it matches with in the ruleset and apply the input values accordingly. 

Below is the list of example input-output points and Baseline Expected Output:

---- Example 0 ------


$X\_s$: [0.018, 0.14, ... $K$ values]
$X\_a$: [0.06, 0.23, ... $K$ values]
$Y$:  [15, 12, ... $T$ values]


---- Example 1 ------


$X\_s$: [0.025, 0.083, ... $K$ values]
$X\_a$: [0.05, 0.19, ... $K$ values]
$Y$:  [15, 12, ... $T$ values]

*<Insert More ...>*

-----------------------

Baseline $Y$: [15, 12, ... $T$ values]

Rules Table: `<Insert Ruleset in JSON schema format>`

**Task:**
1.  Analyze the Baseline $Y$ trend and the rules above.
2.  Generate a new pair of **weekly input time series** ($X_a$, $X_s$), each of length 15, such that the trends in $X_a$ and $X_s$ are different from all previous example inputs. The new input trends should be consistent with the ruleset and so its resulting output $Y$ is expected to closely follow the Baseline $Y$ trend. All input values are within (0.001, 0.20).
3.  For each output phase, briefly justify your input choices for the corresponding input weeks, referencing the rules and the expected delayed/inverse relationship.


### Goal: Ruleset Refinement by evaluating $Y'$ (generated from $(X_{s}', X_a')$ and $Y_{base}$)

**Prompt:**

You have generated a new pair of weekly input time series ($X_s'$, $X_a'$) using a causal ruleset, simulated the system, and obtained an output $Y'$. $Y'$ is different from the Baseline $Y$. Your goal is to analyze why the input values did not produce an output closely matching Baseline $Y$, and to update the rules for generating better-aligned inputs in the future. The relationship is assumed to be inverse and delayed, i.e. increases in $X_a$ and $X_s$ may precede decreases in $Y$, and vice versa, with a temporal lag. This is a key information. Keep in mind.

Your generated weekly input time series:

$X\_a$ = [0.03, 0.04, ... $K$ values]
$X\_s$ = [0.02, 0.03, ... $K$ values]


corresponding $Y'$: [5, 24, ... $T$ values]

-----------------------

Baseline $Y$: [15, 12, ... $T$ values]

Previous Rules Table: *<Insert Ruleset in JSON schema Format>*

---

Please divide the baseline output into specific output phases listed below in order of timing and infer the input trends that are responsible. Each output phase should be unique (no duplicate entries in output column) and the rules should provide input and corresponding output trend behavior for the total time window (i.e all the input window should concatenate to cover from T=0 to T=100). Input time window interval should be a multiple of 7. Fill in the table below with the symbolic causal rules. Each rule should specify:

* Input Trend Window: the time interval where a specific trend in $X_s$ and $X_a$ is observed.
* Trend Description: qualitative trends over that window.
* Output Phase Window: the time interval where a specific phase in $Y$ occurs, as a delayed effect of the input.
* Phase Description: the qualitative behavior of $Y$.
* Symbolic Rule: a formal statement capturing this as a logical implication.

Use the vocabulary:

* Trends: Low, Moderate, HighStable, SharpSpike, SlowRise, DipThenRise, LowModerate
* Output Phases: InitialGrowth, PeakFormation, PostPeak, Decline, Resurgence

**Task:**

1.  Compare the simulated output $Y'$ to the Baseline $Y$. Identify where and how they differ (e.g., timing, magnitude, shape of peaks/valleys, phase alignment).
2.  Diagnose what is lacking or incorrect in the input values ($x_a$, $x_s$) that led to the mismatch. Consider aspects such as: i) Are the input values too high/low, too early/late, too sharp/flat, or not sustained enough for each output phase? ii) Is the delay or inverse relationship not properly captured in the input pattern?
3.  Update and return the rules table that better guides the generation of future inputs that will produce outputs more closely aligned with Baseline $Y$.

**Additional Notes:**

* The goal is to iteratively refine the rules so that future input generation produces outputs that are as close as possible to Baseline $Y$.
* There may be one input ($X_a$ or $X_s$) that is more responsible for certain events like containing the peak faster or making resurgence possible, explore that and modify that inputs rule only
* *<Add more comments/constraints at each iteration>*

Make sure to use the table format shown above. Do not use domain-specific terminology; instead, describe time-series patterns using abstract trend labels. Ensure the symbolic rule column uses the logical form:
$$
\Box_{[t1,t2]}(\text{Trend}(X_s) \land \text{Trend}(X_a)) \rightarrow \Diamond_{[t3,t4]}\text{Phase}(Y)
$$

**JSON Schema Representation**
```json
{
  "RuleID": "R1",
  "InputTrendWindow": [5,15],
  "Trends": {"X_s": "Low","X_a": "SlowRise"},
  "OutputPhaseWindow": [20,30],
  "Y_Phase": "PeakFormation",
  "SymbolicRule": "$\\square_{[5,15]}(\\text{Low}(X_s) \\land \\text{SlowRise}(X_a)) \\rightarrow \\lozenge_{[20,30]}\\text{PeakFormation}(Y)$"
}
```
