# 27. Multimodal learning

**Prerequisites:** vector similarity, matrix multiplication, variance, and the distinction between training and evaluation data. **Objective:** fit an alignment between paired representations, evaluate retrieval on unseen pairs, and calculate a justified fusion of two measurements.

Different modalities can describe complementary, redundant, or conflicting evidence. Sharing a vector width does not make their coordinates comparable. An image encoder might place one concept along a different direction from a text encoder. Paired observations provide evidence for learning a correspondence, but the meaning and granularity of the pairing must be stated.

Our laboratory uses synthetic numeric features to isolate alignment. Fifty training rows have shape `[50,6]`. A hidden orthogonal matrix rotates the first modality into the second, followed by small independent noise. The names “image” and “text” in the code label the roles only: no actual image or language encoder runs. We fit an orthogonal map `R[6,6]` minimizing

`||X @ R - Y||_F², subject to R.T @ R = I`.

If `X.T @ Y = U @ S @ V.T` is its singular value decomposition, the solution is `R=U @ V.T`. Orthogonality preserves lengths and angles within a modality. Retrieval compares row-normalized mapped features against row-normalized candidate features. The similarity matrix for thirty new pairs has shape `[30,30]`; each row's largest entry chooses a candidate. All thirty evaluation pairs are generated separately from fitting.

The second example concerns numerical evidence fusion. For independent unbiased measurements with known variances, inverse-variance weights minimize the variance of an unbiased linear combination. A camera reading of 1000 with standard deviation 20 and an acoustic reading of 1060 with standard deviation 10 get weights `.2` and `.8`. Their fused value is `1048`; its variance is `1/(1/400+1/100)=80`, so standard deviation is approximately `8.944`. This calculation relies on independent errors, not merely different sensors.

Run:

```bash
python -m mlfirst --lesson 27
```

Inspect `retrieval_before_alignment`, `retrieval_after_alignment`, `heldout_pairs`, `fusion_weights`, `fused_speed`, and `fused_std`. Seed 42 improves retrieval from zero to one on this deliberately simple heldout task. That result validates the alignment mechanism under its assumptions. It does not establish semantic understanding, visual grounding, or robustness to real-world modality gaps.

The output `aligned_audio_time=1.6` adds a `.4`-second recorder offset to an audio event at `1.2`. Synchronization metadata can change an event-order conclusion. Missing audio differs from a silent audio recording; neither justifies inventing a sound. Likewise, high embedding similarity is a compatibility score, not a calibrated probability that every detail of a caption is true.

## Exercises

1. Find fusion weights when both sensors have equal variance.
2. What breaks if the sensor errors share a strong common bias?
3. Why not fit the alignment using evaluation pairs?

<details><summary>Hints and worked solutions</summary>

1. Equal inverse variances normalize to `.5` and `.5`; the estimate is their arithmetic mean.
2. Independence and unbiasedness fail. A shared bias survives averaging, and covariance changes the uncertainty calculation. The reported standard deviation would be too optimistic.
3. That would allow evaluation examples to influence the map, measuring fitted correspondence rather than its transfer to new observations. Use training pairs for fitting and preserve a separate test set.

</details>

Source: [`orthogonal_alignment`, `normalize_rows`, and `lesson_27`](../mlfirst/neural.py). Verification: [`test_neural.py`](../tests/test_neural.py).
