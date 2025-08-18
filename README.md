# attention
A self-attention implementation

Consider a self-attention block operating on sequential inputs of shape (seq_len × batch_size × input_dim). Each layer performs scaled dot-product attention with a learned residual connection from inputs to outputs. Using single-head attention with hidden dimension d, the forward propagation for a layer is:

Q = X W_q
K = X W_k
V = X W_v
L = (Q K^T) / sqrt(d)
P = softmax(L)  (row-wise over keys)
C = P V
Y = C W_o + X W_d

- W_q, W_k, W_v map inputs to queries, keys, and values in R^(input_dim × d)
- W_o maps the contextualized representations to outputs in R^(d × output_dim)
- W_d provides a learned residual (feedthrough) in R^(input_dim × output_dim)

For the backward pass with mean-squared error loss and ∂L/∂Y known:

∂L/∂W_o = C^T (∂L/∂Y)
∂L/∂W_d = X^T (∂L/∂Y)
∂L/∂C = (∂L/∂Y) W_o^T
∂L/∂P = (∂L/∂C) V^T
∂L/∂V = P^T (∂L/∂C)

Let softmax be row-wise: P = softmax(L). For each row r the gradient through softmax is:
∂L/∂L_r = (∂L/∂P_r - (∑_j (∂L/∂P_r)_j P_{rj})) ⊙ P_r

Through the score matrix:
∂L/∂Q = (∂L/∂L) K / sqrt(d)
∂L/∂K = (∂L/∂L)^T Q / sqrt(d)

Finally, input-side parameter and data gradients:
∂L/∂W_q = X^T (∂L/∂Q), ∂L/∂W_k = X^T (∂L/∂K), ∂L/∂W_v = X^T (∂L/∂V)
∂L/∂X accumulates from all paths:
(∂L/∂Y) W_d^T + (∂L/∂V) W_v^T + (∂L/∂Q) W_q^T + (∂L/∂K) W_k^T

The AdamW optimizer maintains exponential moving averages of gradients and their squares via β1 and β2, and applies decoupled weight decay λ. With learning rate η, step t, and ε for numerical stability, each parameter W is updated as:

m = β1 m + (1−β1) g
v = β2 v + (1−β2) g^2
W ← (1 − λη) W − η · (m/(1−β1^t)) / (sqrt(v/(1−β2^t)) + ε)

The implementation leverages BLAS (OpenBLAS) for matrix operations and follows the same style and data pipeline used in the mlp and ssm repositories:
- Time-major layout (seq_len × batch_size × feature) for intermediate buffers
- Full-batch training with synthetic sequence data and MSE loss
- AdamW with weight decay
- Save/load routines persisting both parameters and optimizer state

How to run
- CPU
  - sudo apt update
  - sudo apt install clang time libopenblas-dev
  - make run

Notes
- The provided CPU implementation uses single-head attention for clarity and compactness. Extending to multi-head attention is straightforward by splitting the hidden dimension across heads and reshaping Q/K/V accordingly.
- Synthetic data shares the same generator style as in ssm, including temporal lags to encourage modeling sequential dependencies.