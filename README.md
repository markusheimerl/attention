# attention
A self-attention implementation

Consider a multi-head self-attention mechanism operating on batched sequence inputs of shape `(batch_size × seq_len × d_model)`. The architecture consists of query, key, value transformations followed by scaled dot-product attention and output projection, where the forward propagation follows:

```math
\begin{align*}
Q &= XW_q \\
K &= XW_k \\
V &= XW_v \\
\text{Attention}(Q,K,V) &= \text{softmax}\left(\frac{QK^\top}{\sqrt{d_{model}}}\right)V \\
Y &= \text{Attention}(Q,K,V)W_o
\end{align*}
```

The query transformation matrix $W_q$ maps input sequences to query representations, the key transformation matrix $W_k$ generates attention keys, and the value transformation matrix $W_v$ produces value vectors for weighted aggregation. The output projection matrix $W_o$ transforms the attention output to the final representation. The scaled dot-product attention mechanism computes similarity scores between queries and keys, applies softmax normalization to obtain attention weights, and produces a weighted combination of values.

### Backward Pass

The backward pass through the attention mechanism involves computing gradients for each component through the chain rule, where $\odot$ denotes elementwise multiplication and the softmax gradient follows:

```math
\begin{align*}
\frac{\partial L}{\partial Y} &= Y - Y_{\text{true}} \\
\frac{\partial L}{\partial W_o} &= \text{AttnOutput}^\top\left(\frac{\partial L}{\partial Y}\right) \\
\frac{\partial L}{\partial \text{AttnOutput}} &= \left(\frac{\partial L}{\partial Y}\right)W_o^\top \\
\frac{\partial L}{\partial V} &= \text{AttnWeights}^\top\left(\frac{\partial L}{\partial \text{AttnOutput}}\right) \\
\frac{\partial L}{\partial \text{AttnWeights}} &= \left(\frac{\partial L}{\partial \text{AttnOutput}}\right)V^\top \\
\frac{\partial L}{\partial \text{AttnScores}_i} &= \text{AttnWeights}_i \odot \left(\frac{\partial L}{\partial \text{AttnWeights}_i} - \sum_j \frac{\partial L}{\partial \text{AttnWeights}_{i,j}} \text{AttnWeights}_{i,j}\right) \\
\frac{\partial L}{\partial Q} &= \frac{\partial L}{\partial \text{AttnScores}} K / \sqrt{d_{model}} \\
\frac{\partial L}{\partial K} &= \left(\frac{\partial L}{\partial \text{AttnScores}}\right)^\top Q / \sqrt{d_{model}} \\
\frac{\partial L}{\partial W_q} &= X^\top\left(\frac{\partial L}{\partial Q}\right) \\
\frac{\partial L}{\partial W_k} &= X^\top\left(\frac{\partial L}{\partial K}\right) \\
\frac{\partial L}{\partial W_v} &= X^\top\left(\frac{\partial L}{\partial V}\right)
\end{align*}
```

### AdamW Optimizer

The AdamW optimizer maintains exponential moving averages of gradients and their squares through $\beta_1$ and $\beta_2$, while simultaneously applying L2 regularization through weight decay $\lambda$. The learning rate is denoted by $\eta$, $t$ is the current training iteration, and $\epsilon$ is a small constant for numerical stability. For each weight matrix $W$, the update rule is:

```math
\begin{align*}
m &= \beta_1m + (1-\beta_1)(\frac{\partial L}{\partial W}) \\
v &= \beta_2v + (1-\beta_2)(\frac{\partial L}{\partial W})^2 \\
W &= (1-\lambda\eta)W - \eta\cdot\frac{m}{1-\beta_1^t}/\sqrt{\frac{v}{1-\beta_2^t} + \epsilon}
\end{align*}
```

The implementation leverages BLAS for matrix operations, enabling efficient computation on modern hardware. The attention mechanism is trained on a synthetic task where the model must identify the sequence position with the maximum value in the first feature dimension and propagate that entire row to all output positions.

## How to run

```sh
sudo apt update
sudo apt install clang time libopenblas-dev
make run
```
