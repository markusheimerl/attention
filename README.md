# attention
A self-attention implementation

Consider a self-attention mechanism operating on batched inputs of shape (batch_size × seq_len × d_model). The architecture consists of query, key, value, and output projection matrices that enable the model to selectively attend to different positions in the input sequence. For each attention head, the forward propagation follows:

$$
\begin{align*}
Q &= XW_q \\
K &= XW_k \\
V &= XW_v \\
S &= \frac{QK^\top}{\sqrt{d_{\text{model}}}} \\
A_{ij} &= \frac{\exp(S_{ij})}{\sum_k \exp(S_{ik})} \\
Z &= AV \\
Y &= ZW_o
\end{align*}
$$

The query transformation matrix $W_q$ maps inputs to query representations, the key matrix $W_k$ produces keys for matching, the value matrix $W_v$ generates values to be aggregated, and the output projection matrix $W_o$ transforms the attended values to final outputs. The scaled attention scores $S$ measure compatibility between queries and keys, normalization produces attention weights $A$, and $Z$ represents the weighted combination of values.

The backward pass through the attention mechanism involves careful propagation through the softmax operation and matrix multiplications. For the softmax gradient, where $A_{ij}$ represents the attention weights and $\frac{\partial L}{\partial A_{ij}}$ is the incoming gradient:

$$
\begin{align*}
\frac{\partial L}{\partial Y} &= Y - Y_{\text{true}} \\
\frac{\partial L}{\partial W_o} &= Z^\top(\frac{\partial L}{\partial Y}) \\
\frac{\partial L}{\partial Z} &= (\frac{\partial L}{\partial Y})(W_o)^\top \\
\frac{\partial L}{\partial V} &= A^\top(\frac{\partial L}{\partial Z}) \\
\frac{\partial L}{\partial A} &= (\frac{\partial L}{\partial Z})V^\top \\
\frac{\partial L}{\partial S_{ij}} &= A_{ij}\left(\frac{\partial L}{\partial A_{ij}} - \sum_k \frac{\partial L}{\partial A_{ik}}A_{ik}\right) \\
\frac{\partial L}{\partial Q} &= \frac{(\frac{\partial L}{\partial S})K}{\sqrt{d_{\text{model}}}} \\
\frac{\partial L}{\partial K} &= \frac{(\frac{\partial L}{\partial S})^\top Q}{\sqrt{d_{\text{model}}}} \\
\frac{\partial L}{\partial W_q} &= X^\top(\frac{\partial L}{\partial Q}) \\
\frac{\partial L}{\partial W_k} &= X^\top(\frac{\partial L}{\partial K}) \\
\frac{\partial L}{\partial W_v} &= X^\top(\frac{\partial L}{\partial V})
\end{align*}
$$

The AdamW optimizer maintains exponential moving averages of gradients and their squares through $\beta_1$ and $\beta_2$, while simultaneously applying L2 regularization through weight decay $\lambda$. The learning rate is denoted by $\eta$, $t$ is the current training iteration, and $\epsilon$ is a small constant for numerical stability. For each weight matrix $W$, the update rule is:

$$
\begin{align*}
m &= \beta_1m + (1-\beta_1)(\frac{\partial L}{\partial W}) \\
v &= \beta_2v + (1-\beta_2)(\frac{\partial L}{\partial W})^2 \\
W &= (1-\lambda\eta)W - \eta\cdot\frac{m}{1-\beta_1^t}/\sqrt{\frac{v}{1-\beta_2^t} + \epsilon}
\end{align*}
$$

The implementation leverages BLAS for matrix operations, enabling efficient computation on modern hardware.

## How to run
```
sudo apt update
sudo apt install clang time libopenblas-dev
make run
```