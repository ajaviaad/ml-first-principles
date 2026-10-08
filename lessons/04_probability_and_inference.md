# 4 · Probability and statistical inference

**Prerequisites:** lesson 3, fractions and natural logarithms.
**Goal:** reverse a conditional probability correctly and distinguish fitting a
parameter from quantifying uncertainty about it.

## Conditioning changes the reference population

Suppose a manufactured component is defective with probability .01. A screen
flags 90% of defects and 5% of sound components. In 10,000 hypothetical parts,
there are 100 defects, 90 flagged defects and 495 flagged sound parts. Therefore
`P(defect | flag) = 90/(90+495)`, approximately .154. It is not .9: that number
describes `P(flag | defect)`, a different denominator.

Bayes' rule expresses the same calculation as
`P(A|B) = P(B|A)*P(A) / P(B)`. The prior describes the hypothesis before this
evidence; the likelihood describes evidence conditional on the hypothesis; the
posterior updates the hypothesis after observing evidence. With zero evidence
probability, this conditional probability is undefined and our function raises
an error.

## Fit a Bernoulli probability

For s successes in n independent Bernoulli trials, maximum likelihood chooses
`p_hat=s/n`. Eight successes in ten trials give .8. With a Beta(1,1) prior, the
posterior is Beta(9,3), whose mean is `9/(9+3)=.75`. The posterior mean averages
over parameter uncertainty; it is not the maximum-likelihood estimate.

The bootstrap resamples n observations with replacement and recomputes a
statistic. The percentiles of the resulting means give a simple interval
approximation. This uses an iid assumption and cannot create a population absent
from the sample. Group or serial dependence calls for a different resampling
unit. With only ten Bernoulli outcomes, the sampling distribution is coarse.
Do not interpret this bootstrap interval as the Beta posterior interval.

## Probabilities also define losses

For a discrete distribution p, entropy is `H(p)=-sum(p*log(p))`. Cross-entropy
against q is `-sum(p*log(q))`. Their difference is
`KL(p||q)=sum(p*log(p/q))`. Natural logs measure these quantities in nats.
For a fair coin, entropy is log(2). Assigning probabilities [.9,.1] increases
expected log loss even though that forecast sounds more decisive.

```bash
python -m mlfirst --lesson 4
```

Read `defect_given_flag`, `bernoulli_mle`, `posterior_mean` and
`bootstrap_mean_interval` as different answers to different questions. Verify
that `entropy_nats + kl_nats` equals `cross_entropy_nats`. Only the bootstrap
resampling changes with the seed; the analytic arithmetic is fixed.

## Practice

1. Recalculate the defect posterior when prevalence is .1, keeping the screen
   characteristics unchanged.
2. With four successes in four trials and a Beta(1,1) prior, compare MLE and
   posterior mean.
3. Bootstrap `[3,3,3]`. What interval must any resampling produce, and what does
   it fail to tell you about unobserved values?

<details><summary>Hints and worked solutions</summary>

1. `.09 / (.09 + .045) = 2/3`. The same screen has different positive predictive
   value in a population with a different base rate.
2. MLE is 1; the posterior is Beta(5,1) with mean 5/6. A posterior distribution
   still allows values below one, conditional on the assumed prior and likelihood.
3. Every resampled mean is 3, so the interval is `[3,3]`. This empirical sample
   contains no variability; the result cannot prove the population is constant.

</details>

**Code:** [Bayes, bootstrap and lesson_04](../mlfirst/foundations.py).
