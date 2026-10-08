# Four investigations to complete on your own

These projects use the existing code and generated data. Each has a concrete
deliverable and an assessment criterion. Create your own files under `projects/`
and save generated outputs under `results/`; do not modify the original listings.
The brief is intentionally less prescriptive than a lesson exercise.

## 1. A forecast you can defend

Use the bakery demand experiment. Compare a mean baseline, a linear model and
an expanded feature model. Separate fitting, selection, interval calibration
and final evaluation. State when each feature is available and what “demand”
means. Fit preprocessing only on training observations.

Deliver a script plus a one-page report containing the split sizes and indices,
validation decisions, final MAE/RMSE, interval coverage and width, and coverage
for promoted weekends with sample counts. Include at least five seeds.

**Success criterion:** another reader can rerun the experiment and tell which
data affected each decision. The strongest model need not win every seed.
Explain how stock-limited sales would change the target and what data would be
needed to learn unconstrained demand.

**Starting points:** `mlfirst/foundations.py`, `mlfirst/classical.py`,
`mlfirst/decisions.py`, `book_code/companion_investigation.py`.

## 2. An algorithm under stress

Choose kNN, a decision tree or PCA. Write down one assumption about scale,
geometry or sampling. Design two synthetic worlds differing in that assumption
while holding other relevant conditions fixed. Examples: rescale one irrelevant
kNN feature by 100; rotate a low-rank cloud; add a distracting identifier to a
tree dataset and change its relationship in the test period.

Deliver the generating function, a baseline comparison and an explanation of
the observed mechanism. Include a hand-computable four-point example and one
test that would fail if your implementation violated the intended property.

**Success criterion:** the investigation distinguishes an implementation bug
from an unsuitable modeling assumption. Report sample sizes and random seeds.

**Starting point:** `mlfirst/classical.py`.

## 3. Representations with a counterexample

Choose contrastive learning, paired multimodal alignment or frozen features.
Train on constructed pairs and evaluate on held-out examples. Then deliberately
break the correspondence or introduce a nuisance cue shared only in training.
Compare matched-pair retrieval with random or shuffled-pair performance.

Deliver a runnable experiment, the unchanged evaluation protocol and an error
analysis. Inspect both loss and a downstream score: a declining optimization
loss alone is insufficient. Show one transformation that should preserve meaning
and one that should change it.

**Success criterion:** you can explain why the training signal rewards a useful
relationship in one world and a shortcut in the other. State that this is a toy
representation study, not evidence about pretrained language or vision systems.

**Starting point:** `mlfirst/neural.py`.

## 4. Learn a policy, then audit the boundary

Use the small robot MDP. Solve it by value iteration, then learn action values
from sampled transitions. Compare a greedy learned policy with the exact one.
Repeat with different exploration rates and seeds. Introduce a rollout time
limit and distinguish it from genuine termination in the learning target.

Deliver curves or tables of value error versus observed transitions, state-action
visit counts and a worked Bellman backup. Explain why the return observed during
exploration differs from evaluation of the greedy learned policy.

**Success criterion:** the exact solution acts as an independent oracle. Show a
case where treating truncation as termination changes the target, and quantify
the resulting bias rather than merely reporting a pass/fail assertion.

**Starting points:** `mlfirst/decisions.py`, `book_code/companion_rl.py`.

## Report template

For each project, write: the question; data-generating assumptions; what was
fixed before evaluation; the competing explanations; the observed result with
uncertainty; the limitation; and the next experiment that could distinguish the
remaining explanations. Separate exploratory findings from preregistered tests.
