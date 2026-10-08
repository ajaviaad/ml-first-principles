# 17. Learning from images

**Prerequisites:** arrays, weighted sums, and the dense-layer idea from Lesson 16. **Objective:** calculate a spatial filter, predict its output shape, and distinguish shared parameters from computational cost. This is a forward-operation laboratory, not a trained image classifier.

An image can be represented as an array. Adjacent entries usually describe neighboring locations, so a local filter encodes useful structure. The same small set of coefficients is reused across the image. Our `conv2d` function takes a single-channel `image[H,W]` and `kernel[kH,kW]`, then returns one feature map. It implements cross-correlation: the filter is not reversed. That convention is commonly called convolution in neural networks.

At output coordinate `(i,j)`, the function computes

`output[i,j] = sum(image[i*s+a,j*s+b] * kernel[a,b])`.

Here `s` is stride; padding is added before indexing. The output height is `floor((H + 2p - kH)/s) + 1`, with an analogous width expression. Dilation is not implemented in this small helper. For `H=32`, `kH=3`, `p=1`, and `s=2`, the height is `16`.

Calculate one row by hand. Applying filter `[-1,1]` to `[1,1,4,4]` gives `[0,3,0]`. The positive response locates a rise. Repeating the input row four times makes a `4×4` image and a `4×3` feature map. Its global average is `1`. Averaging keeps evidence that a rise exists while removing its explicit location. That distinction matters when the task is locating damage instead of assigning one image label.

Run:

```bash
python -m mlfirst --lesson 17
```

Inspect `edge_response`, `feature_shape`, `global_average`, and `zero_branch_residual_error`. A residual construction adds a transformed branch to the input; a zero branch gives the original image exactly. The lesson also reports `dense_convolution_weights=36864` and `depthwise_separable_weights=4672` for a `3×3`, 64-channel example. These counts exclude biases. The cheaper factorization changes the representable family; it is not guaranteed to produce identical features.

Translation equivariance means a translated input gives a correspondingly translated representation. It does not mean the representation is unchanged. Boundaries, padding, and strides complicate this statement. A one-pixel movement can change which positions a stride-two operation samples. Similarly, an augmentation asserts that a transformation preserves the target: reflecting an ordinary object may be reasonable, but reflecting a printed digit or left/right medical label may change its meaning.

## Exercises

1. Apply the filter to `[4,4,1,1]`. What does the sign communicate?
2. Find the output width for a length-7 signal, width-3 filter, stride 2, and no padding.
3. Compare parameter counts when both channel counts double from 64 to 128.

<details><summary>Hints and worked solutions</summary>

1. Subtract each value from its successor: `[0,-3,0]`. The negative response signals a fall rather than a rise.
2. Use the shape formula: `floor((7-3)/2)+1 = 3`. Starting positions are 0, 2, and 4.
3. Ordinary convolution changes from `9×64²` to `9×128²`, a fourfold increase. The depthwise-separable count becomes `9×128 + 128² = 17536`, because its spatial and channel-mixing parts scale differently.

</details>

Source: [`conv2d` and `lesson_17`](../mlfirst/neural.py). Verification: [`test_neural.py`](../tests/test_neural.py).
