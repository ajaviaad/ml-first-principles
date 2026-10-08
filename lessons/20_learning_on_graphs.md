# 20. Learning on graphs

**Prerequisites:** matrices, weighted sums, and the meaning of an undirected edge. **Objective:** propagate node features, verify consistent relabeling, and observe how repeated averaging reduces distinctions. This is a fixed linear graph layer, not a trained node-classification benchmark.

A graph describes entities and their relationships. Its adjacency matrix `A[N,N]` has entry 1 when two nodes are connected. Each node has a feature row in `X[N,F]`. A simple graph convolution first includes self-connections, then normalizes neighbor contributions:

`A_tilde = A + I`

`S = D^(-1/2) @ A_tilde @ D^(-1/2)`

`H = S @ X @ W`.

Here `D` contains row sums of `A_tilde`, and `W[F,H]` maps feature width to hidden width. The normalization controls how strongly high-degree neighborhoods contribute. Our helper assumes an undirected unweighted graph supplied without self-loops; it adds one loop per node. Other graph types need explicit treatment rather than blindly reusing this convention.

Use a three-node path, with edges 0–1 and 1–2, and scalar features `[1,0,3]`. Adding self-loops gives degrees `[2,3,2]`. With scalar weight 1, node 0 receives half its own feature and `1/sqrt(6)` times node 1's feature, yielding `.5`. Node 1 receives `1/sqrt(6)` times each endpoint, yielding `4/sqrt(6)≈1.633`. Node 2 yields `1.5`. This can be checked without fitting any parameters.

Run:

```bash
python -m mlfirst --lesson 20
```

Inspect `node_outputs`, `permutation_error`, and `degree_adjusted_spread_after_40`. Relabeling nodes must consistently reorder both adjacency axes and feature rows. The output should follow the same permutation. The reported error is zero up to floating-point arithmetic. This property is permutation equivariance; a graph-level sum would instead be invariant to node ordering.

Repeated propagation illustrates oversmoothing. With this symmetric normalization, the limiting values are proportional to the square root of degree, not necessarily identical across nodes. Dividing the final scalar states by `sqrt(degree)` before measuring their spread is therefore essential. The spread after forty steps is tiny because repeated propagation has largely removed the distinguishing components of these initial features.

Graph construction is itself a modeling choice. Connecting test nodes using labels unavailable during deployment leaks information. Aggregating many neighbors may amplify irrelevant relations. More layers expand the reachable neighborhood but can reduce distinctions and increase computation. The toy layer has neither nonlinearities nor learned edge weights, so it only reveals one mechanism behind these effects.

## Exercises

1. Give the result for an isolated node with feature 7 and scalar weight 1.
2. Explain why permuting feature rows alone does not test equivariance.
3. Replace the symmetric normalization with row normalization. What changes about constant features?

<details><summary>Hints and worked solutions</summary>

1. Its added self-loop gives degree 1, so its output remains 7.
2. The features would now belong to different nodes while the edges stay fixed. That changes the problem rather than merely relabeling it.
3. Every row of `D^(-1) @ A_tilde` sums to one. A constant scalar feature is therefore preserved exactly. Symmetric normalization generally preserves degree-weighted structure instead.

</details>

Source: [`lesson_20`](../mlfirst/neural.py), [`gcn_step`](../book_code/companion_deep.py). Verification: [`test_neural.py`](../tests/test_neural.py).
