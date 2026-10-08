# 15 — Probabilistic graphical models

**Prerequisites:** Bayes' rule (04), matrix products (03), and conditional independence (08). **Objective:** use a hidden Markov model to distinguish filtering, smoothing and the most likely joint sequence.

A graphical model expresses a joint distribution through local relationships. This lesson implements one concrete structure: a hidden Markov model (HMM). A latent state generates each observation, and the next latent state depends on the current state. With `K` states and `M` observation symbols, the initial vector has shape `(K,)`, transition matrix `A` has shape `(K,K)`, and emission matrix `B` has shape `(K,M)`. `A[i,j]` means next-state `j` given current-state `i`; every matrix row sums to one.

Filtering asks for the current state using observations seen so far. Given a posterior row vector, predict with `prior=posterior @ A`, multiply by the likelihood for the new symbol, then normalize. At the first observation, use the initial distribution directly. The forward algorithm performs these operations in roughly `O(TK²)` for dense transitions, avoiding all `K**T` possible state paths.

Consider a machine initially normal with probability .9 and degraded with probability .1. Alarm probability is .1 when normal and .8 when degraded. After one alarm, masses are .09 and .08, giving degraded probability `8/17≈.471`. If normal becomes degraded with probability .05 and degraded remains degraded with probability .9, the next degraded prior is `.45`. A second alarm gives degraded probability `.36/(.055+.36)≈.867`.

```bash
python -m mlfirst --lesson 15
```

`observations=[1,1]` encodes two alarms; `state_names` defines the state order. `filtered_probabilities` is a `(2,2)` nested list matching the hand calculation. `sequence_probability=.07055` is the total probability of these observations, summed over all possible hidden paths. `log_likelihood` is its logarithm. Normalizing each forward step prevents products shrinking to numerical zero; adding the logarithms of the saved normalizers recovers the sequence likelihood.

Smoothing asks about an earlier state after seeing later evidence. A backward recursion sends future likelihood information back through transitions. `smoothed_probabilities` revises the first-time probability using both alarms; its final row equals the final filtering row because no future evidence remains there. Strong evidence tomorrow can change today's retrospective assessment without changing what was knowable today.

Viterbi decoding replaces sums over previous states with maxima and records predecessors. `viterbi_path=[1,1]`, `viterbi_state_names`, and `viterbi_joint_probability=.0576` identify the most probable whole path. This joint path probability is smaller than total sequence probability. Selecting each state's marginal winner separately is a different operation and need not produce the most probable coherent path.

Model failures include incorrect emission assumptions, dependencies between observations after conditioning on state, and unrealistic state durations. The basic HMM implies geometric durations. A discovered latent state is not automatically a physical mechanism. Zero-probability observations cause an explicit error here rather than an invented posterior; an empty sequence has probability one and no inferred rows.

**Practice**

1. Calculate the joint probability of the all-normal path for two alarms.
2. Normalize the four path masses to find the posterior probability of the Viterbi path.
3. Explain why smoothing must not be used for an online prediction evaluated at the first alarm.

<details><summary>Worked solutions</summary>

1. `.9×.1×.95×.1=.00855`.
2. Path masses for normal/normal, normal/degraded, degraded/normal and degraded/degraded are `.00855,.0036,.0008,.0576`. Their sum is `.07055`; the winning path posterior is `.0576/.07055≈.8164`.
3. Smoothing consumes the second alarm, which is future information at that time. Using it would evaluate a retrospective inference task while claiming online performance.

</details>

**Scope and source:** categorical HMM filtering, smoothing and Viterbi are implemented. General Bayesian networks, Markov random fields, variable elimination, Gibbs sampling, and Baum–Welch parameter learning are not. See [module](../mlfirst/classical.py), [original forward algorithm](../book_code/companion_classical.py), and [complete-path oracle tests](../tests/test_classical.py).
