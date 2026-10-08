# companion_investigation.py
"""Chapter 32: synthetic demand experiment. Requires Python 3.11+ and NumPy.
Run: python companion_investigation.py
All data are generated locally; no downloads or private data are used.
"""
import json
import numpy as np

def sample(rng, n, shift=False):
    temperature = rng.normal(18.0 + (5.0 if shift else 0.0), 5.0, n)
    promotion = rng.binomial(1, 0.3, n)
    weekend = rng.binomial(1, 2 / 7, n)
    t = (temperature - 18.0) / 5.0
    mean = 80 + 7*t - 2*t*t + 10*promotion + 14*weekend
    mean += (24 if shift else 12)*promotion*weekend
    noise = rng.normal(0.0, 3 + 2*weekend, n)
    return np.column_stack([t, promotion, weekend]), mean + noise

def design(x, kind):
    t, p, w = x.T
    if kind == 'constant':
        return np.ones((len(x), 1))
    base = np.column_stack([np.ones(len(x)), t, p, w])
    if kind == 'linear':
        return base
    if kind == 'expanded':
        return np.column_stack([base, t*t, p*w])
    raise ValueError(kind)

def rmse(y, prediction):
    return float(np.sqrt(np.mean((y-prediction)**2)))

def run():
    rng = np.random.default_rng(20261004)
    train = sample(rng, 1000)
    validation = sample(rng, 250)
    calibration = sample(rng, 250)
    test = sample(rng, 250)
    models, scores = {}, {}
    for name in ('constant','linear','expanded'):
        beta = np.linalg.lstsq(
            design(train[0], name), train[1], rcond=None
        )[0]
        models[name] = beta
        scores[name] = rmse(validation[1], design(validation[0], name) @ beta)
    selected = min(scores, key=scores.get)
    def predict(x):
        return design(x, selected) @ models[selected]
    residuals = np.abs(calibration[1] - predict(calibration[0]))
    rank = int(np.ceil((len(residuals)+1)*0.90))
    radius = (
        float(np.sort(residuals)[rank-1])
        if rank <= len(residuals) else float('inf')
    )
    def evaluate(dataset):
        x, y = dataset
        prediction = predict(x)
        return {'rmse':rmse(y,prediction),
                'mae':float(np.mean(np.abs(y-prediction))),
                'coverage':float(np.mean(np.abs(y-prediction)<=radius)),
                'mean_interval_width':2*radius}
    shifted = sample(rng, 250, shift=True)
    nominal = evaluate(test)
    shifted_metrics = evaluate(shifted)
    base_prediction = design(test[0], 'constant') @ models['constant']
    difference = (test[1]-base_prediction)**2 - (test[1]-predict(test[0]))**2
    boot_rng = np.random.default_rng(314159)
    indices = boot_rng.integers(0,len(difference),(2000,len(difference)))
    boot_means = difference[indices].mean(axis=1)
    interval = np.quantile(boot_means,[0.025,0.975]).tolist()
    result = {'seed':20261004,'validation_rmse':scores,'selected':selected,
              'test':nominal,'shifted':shifted_metrics,'conformal_rank':rank,
              'conformal_radius':radius,
              'test_mse_improvement_over_constant':float(difference.mean()),
              'paired_bootstrap_95_percent_interval':interval,
              'selected_coefficients':models[selected].tolist()}
    assert selected == 'expanded'
    assert np.isfinite(list(scores.values())).all()
    assert rank == 226 and radius > 0
    assert len(models[selected]) == 6
    assert nominal['rmse'] < rmse(test[1],base_prediction)
    return result

if __name__ == '__main__':
    print(json.dumps(run(),indent=2))
# End of companion_investigation.py
