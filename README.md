# Build Deep Learning From Scratch — My Implementation

A personal implementation of [roiamiel1's Build-Deep-Learning-From-Scratch curriculum](https://github.com/roiamiel1/Build-Deep-Learning-From-Scratch), with my own solutions, notes, and extensions as I work through it. I'm following the original 35-stage structure and plan to add new stages and experiments beyond it as I progress.

The philosophy stays the same: **you are the autodiff library.** Nothing is imported that I haven't already built by hand. Every gradient is code I wrote and understand.

---

## What This Is

This repo contains my solutions and notes for each stage of the curriculum. The original ships skeleton files and tests; I fill in the implementations. As I get deeper into the material I'll start diverging — adding extra stages, experiments, or projects that aren't in the original.

The scalar autodiff engine in stages 01–05 is a from-scratch reimplementation of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd): `Value(data)` (stage 01) → computational graph (stage 02) → per-op `_backward` closures (stage 03) → the `backward()` reverse pass (stage 04) → remaining ops `tanh`/`exp`/`relu` (stage 05). Everything afterward generalizes it to tensors.

---

## Tool Restriction

The only permitted packages are NumPy (forward array math only — never to compute a derivative), Matplotlib (visualization), and pytest (tests). No `torch`, `tensorflow`, `jax`, `autograd`, `tinygrad`, `micrograd`, or any library that does autodiff/backprop — until I've built that concept by hand in an earlier stage.

---

## How a Stage Works

Each `stage_xx/` directory contains:

- `README.md` — context, background, key equations, and the exercise.
- `code.py` — my implementation (skeleton filled in).
- `test.py` — pytest tests, including numerical gradient checks via central differences.

Run the tests for a stage:

```bash
pytest stage_xx/test.py
```

Each stage builds on prior ones. Later stages import symbols from the most recent stage that extended them — e.g. `Tensor` is created in stage 08 but extended in stages 11 and 12, so stage 13+ imports from stage 12.

---

## The 35 Stages (+ My Extensions)

| # | Stage | What I Build |
|---|---|---|
| 01 | Scalar Values | `Value`: wrap one number; forward arithmetic via operator overloading. |
| 02 | Computational Graph | Record each result's parents (`_prev`), op (`_op`), and a no-op `_backward` hook. |
| 03 | Local Derivatives | Install per-op `_backward` closures (the local-derivative push). |
| 04 | Chain Rule | `backward()`: topological sort + seed `grad=1` + reverse-walk the closures. |
| 05 | Backprop Engine | Add `tanh`, `exp`, `relu` on the scalar `Value`; the complete micrograd engine. |
| 06 | Vector Operations | A `Vec` container of `Value`s: elementwise ops, dot, sum. |
| 07 | Matrix Operations | A `Mat` of `Value`s: `matmul`/`@`, transpose, reshape, sum, mean. |
| 08 | Tensor Engine | Collapse scalar graphs onto one N-dim NumPy-backed autodiff `Tensor`. |
| 09 | Neuron | A single learnable neuron `y = phi(x @ w + b)` on the `Tensor`. |
| 10 | Dense Layer | Vectorized fully-connected layer `Z = X @ W + b`. |
| 11 | MLP | Stack `Dense` layers with activations between them. |
| 12 | Loss Functions | Single-example MSE/MAE/cross-entropy (+ stable softmax) and sum/mean reductions. |
| 13 | Loss Functions (Batched) | Lift softmax/cross-entropy to a `(B, C)` batch; mean over the batch. |
| 14 | SGD Optimizer | The `Optimizer`/`SGD` update step abstraction. |
| 15 | First Training Loop | Wire MLP + loss + SGD into the canonical learn loop. |
| 16 | Weight Initialization | Xavier/Glorot and He/Kaiming init, and why scale matters. |
| 17 | Momentum | SGD with momentum. |
| 18 | Adam | RMSProp/Adam with bias correction and weight decay. |
| 19 | Batch Training | Minibatching, epochs, shuffling; gradient-variance intuition. |
| 20 | DataLoader | `Dataset`/`DataLoader` batching abstraction. |
| 21 | Regularization | L2 / weight decay; train vs eval mode. |
| 22 | Dropout | Dropout forward + backward; inverted scaling. |
| 23 | BatchNorm | Batch normalization forward + backward by hand. |
| 24 | Conv2D Math | Convolution arithmetic and gradients via im2col. |
| 25 | Conv2D Implementation | `Conv2D`/pooling/flatten as `Tensor` layers. |
| 26 | CNN Project | Stack conv/pool/linear; train on image data. |
| 27 | Attention Math | Scaled dot-product attention forward + backward (pure NumPy). |
| 28 | Self-Attention | Self-attention on the `Tensor` autodiff engine. |
| 29 | Multi-Head Attention | Split/concat heads; the full MHA module. |
| 30 | Transformer | Residuals + LayerNorm + FFN; a full Transformer block. |
| 31 | Vision Transformer | Patch embeddings + Transformer for image classification. |
| 32 | Framework Refactor | Package it all into a clean PyTorch-like `Module`/`Parameter` API. |
| 33 | Capstone: MNIST | End-to-end MNIST classifier. |
| 34 | Capstone: CIFAR-10 | End-to-end CIFAR-10 classifier. |
| 35 | Capstone: Transformer | End-to-end Transformer language model. |
| 36+ | My Extensions | Stages I add beyond the original curriculum. |

---

## Getting Started

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

---

## Progress

| Stage | Status |
|---|---|
| 01 – Scalar Values | ✅ |
| 02 – Computational Graph | ✅ |
| 03 – Local Derivatives | 🚧 |
| 04 – Chain Rule | ⬜ |
| 05 – Backprop Engine | ⬜ |
| 06 – Vector Operations | ⬜ |
| 07 – Matrix Operations | ⬜ |
| 08 – Tensor Engine | ⬜ |
| 09 – Neuron | ⬜ |
| 10 – Dense Layer | ⬜ |
| … | … |

> Updated as I work through stages. ✅ = tests passing, 🚧 = in progress, ⬜ = not started.

---

## Credit

This curriculum was designed by [Roi Amiel](https://github.com/roiamiel1). The scalar autodiff engine is based on [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd). I'm building on their work — all original structure and test suites belong to them.
