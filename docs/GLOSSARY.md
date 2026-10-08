# Working glossary

| Term | Meaning in these experiments |
| --- | --- |
| Example / observation | One unit the model receives, usually a row. |
| Feature | An input available at the declared prediction time. |
| Target / label | The outcome the model tries to predict; its measurement process matters. |
| Parameter | A fitted quantity such as a weight, split or probability. |
| Hyperparameter | A setting of the learning procedure, selected before or through validation. |
| Loss | Cost assigned to an individual prediction or an explicitly stated aggregate. |
| Objective | The quantity optimized, often loss plus a regularization term. |
| Baseline | A simple, relevant comparison evaluated on the same cases. |
| Training | Fitting model and preprocessing parameters. |
| Validation | Choosing between procedures, settings or checkpoints. |
| Test | Evaluating a procedure after those choices are fixed. |
| Calibration set | Separate observations used to fit a probability mapping or prediction-set rule. |
| Leakage | Using information unavailable at prediction time or across an evaluation boundary. |
| Generalization | Performance beyond fitting observations, relative to a stated population. |
| Gradient | Vector of local partial derivatives of a scalar objective. |
| Learning rate | Scale of a parameter update; its effect depends on curvature. |
| Regularization | A preference or constraint limiting a fitted rule's behavior. |
| Logit | A score before sigmoid or softmax; in binary models, log odds. |
| Likelihood | Probability or density of observed data considered as a function of parameters. |
| Posterior | Distribution after conditioning on observations and a model/prior. |
| Calibration | Agreement between stated probabilities and outcome frequencies. |
| Bootstrap | Resampling observed cases to approximate a statistic's sampling variation. |
| Confidence interval | Interval procedure for a population quantity with a repeated-sampling interpretation. |
| Prediction interval | A range for a future observation, not merely a fitted mean. |
| Exchangeability | Invariance of a joint distribution to permutation; stronger than matching histograms. |
| Embedding | Vector representation used for computation or comparison. |
| Latent variable | Unobserved variable in a model of observations. |
| Attention | Query-dependent weighted combination of value vectors. |
| Contrastive learning | Learning representations by distinguishing declared matches from alternatives. |
| Generative model | A model of an observation distribution, often supporting sampling. |
| Reward | Numerical signal defining a sequential decision objective. |
| Return | Accumulated, often discounted, future rewards. |
| Value | Expected return under a specified decision rule. |
| Policy | A rule or distribution for choosing actions from available information. |
| Advantage | An action's value relative to a state baseline. |
| Termination | A true end of the modeled task, normally with no future value. |
| Truncation | An external rollout boundary; future task value may remain. |
| Covariate shift | A change in input distribution. Pure covariate shift keeps the conditional target mechanism fixed. |
| Concept shift | A change in the relationship between inputs and targets. |
| Permutation importance | Performance change after scrambling a feature under a specified evaluation. |
| Invariance | A quantity remains unchanged by a transformation. |
| Equivariance | Transforming an input transforms an output in a corresponding way. |
| Seed | Initialization of a pseudo-random generator; only one part of reproducibility. |
