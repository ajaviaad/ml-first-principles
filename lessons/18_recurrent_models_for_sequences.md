# 18. Recurrent models for sequences

**Prerequisites:** matrix multiplication, tanh, sigmoid, and the chain rule. **Objective:** trace a recurrent state, calculate an LSTM gate update, and understand why carrying information forward differs from learning to retain it. These experiments implement forward recurrence; they do not train a recurrent model with backpropagation through time.

A recurrent network applies the same transition to each sequence element:

`h_t = tanh(x_t @ W_x + h_(t-1) @ W_h + b)`.

In the code, `X[T,D]` contains `T` observations, `W_x[D,H]` transforms their features, `W_h[H,H]` transforms the previous state, and `b[H]` shifts the result. The output is `states[T,H]`. The initial state defaults to zero, but it is an explicit argument so a long stream can be processed in chunks without losing its history.

For a scalar example, feed `[1,0,0,0]` through input weight 1 and recurrent weight `.8`. The first state is `tanh(1) ≈ .762`; the next is `tanh(.8×.762) ≈ .544`. Later values continue to fade. Training must differentiate through every repeated transition. Ignoring activation derivatives for a simple illustration, fifty multipliers of `.8` produce approximately `.0000143`, while fifty multipliers of `1.2` produce approximately `9100`. Actual recurrent derivatives include both weights and activation derivatives.

An LSTM separates a memory cell from its exposed hidden state:

`c_new = f*c_old + i*g; h_new = o*tanh(c_new)`.

The forget, input, and output gates `f,i,o` are sigmoid vectors; the candidate `g` is a tanh vector. Our packed weight matrix has shape `[D+H,4H]`, with gate blocks ordered **forget, input, output, candidate**. Using `c_old=2`, `f=.9`, `i=.2`, `g=-.5`, and `o=.5` gives `c_new=1.7` and `h_new≈.468`. Holding gates fixed, the direct cell derivative is `f`; indirect paths through the gates still matter.

Run:

```bash
python -m mlfirst --lesson 18
```

Compare `rnn_states`, `lstm_cell`, `lstm_hidden`, `contracting_path_50`, `expanding_path_50`, and `forget_path_50`. The latter uses `.99**50≈.605`, illustrating a less contracting direct path, not proving that a trained LSTM remembers indefinitely.

State ownership is a practical pitfall: reusing a patient's or device's state for another stream mixes histories. Padding can also change a final state unless state updates are masked. A bidirectional network can use later observations only when those observations exist at prediction time. Gradient clipping limits large updates; it does not restore a signal already diminished by many small derivatives.

## Exercises

1. Set recurrent weight to zero. Which observations influence each state?
2. Calculate the next cell if the input gate is zero and forget gate is one.
3. Explain how to split a stream into two equivalent forward passes.

<details><summary>Hints and worked solutions</summary>

1. The previous state disappears from the expression, leaving `tanh(x_t @ W_x + b)`. Each output depends only on its current observation.
2. The cell remains exactly `c_old`; the output gate still controls how much of `tanh(c_old)` is exposed.
3. Feed the first pass's final state as `initial` to the second pass. Resetting to zero would change the result. The automated test checks this equality for the plain RNN.

</details>

Source: [`rnn_sequence`, `lstm_step`, and `lesson_18`](../mlfirst/neural.py). Verification: [`test_neural.py`](../tests/test_neural.py).
