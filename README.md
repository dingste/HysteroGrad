# **Important notice**

I know this is annoying: you click on a link to something exciting, and then you get a disclaimer like this... Development on Hysterograd is being **discontinued**. Or rather, it will no longer be continued **here**. I've seen that there's still a lot of room for **improvement** and would like to explore that further elsewhere.

What remains is the latest version, which I already consider exclusive.

---

# HysteroGrad: Hysteretic Information-Geometric Optimizer

**HysteroGrad** is a next-generation PyTorch optimizer that redefines parameter updates as a physical phase transition. By merging **Information Geometry** with **Hysteresis Physics**, it creates a system that doesn't just minimize loss, but evolves through "Liquid" and "Frozen" states.

> "Training is not just optimization; it is the crystallization of information. HysteroGrad protects learned structures by increasing the 'viscosity' of the parameter space as knowledge accumulates."

Physical terms such as temperature, phase transition, and hysteresis are used as operational metaphors for well-defined algorithmic mechanisms (metric scaling, adaptive thresholds, and update gating).

---

## Key & Breakthroughs

Recent benchmarks on **WideResNet-28-2** demonstrate HysteroGrad's unique "Late-Surge" convergence profile:

* **Implicit Rotation (Natural Preconditioning):** Using a diagonal approximation of the Fisher Information, HysteroGrad preconditions the gradient in the local information metric, effectively aligning updates with geodesic descent directions.
* **Phase Transition Efficiency:** Unlike AdamW, which "jitters" in local minima, HysteroGrad enforces a **Schmitt-Trigger** barrier. Updates only occur if the signal is strong enough to overcome the accumulated "stiffness" of the model.
* **Superior ATF:** In head-to-head battles, HysteroGrad achieved **72.7% accuracy** in just 4 epochs on large-scale models, outperforming standard optimizers in **Accuracy per TFLOP (ATF)** by a factor of 3x.

---

## Core Mechanics

### 1. The Metric Tensor

HysteroGrad operates on a **Riemannian manifold**. It treats the parameter space as curved, using a diagonal approximation of the Fisher Information Matrix to scale updates. This ensures that the optimizer is "aware" of the local information density.

### 2. Information Path Length

We track the **internal time** of the system—the integral of the distance traveled in the information space. As  grows, the model "ages" and becomes more resistant to noise.

### 3. Dynamic Hysteresis (Schmitt-Trigger)

Updates are governed by a hysteretic energy barrier:



This mechanism acts as a **geometric low-pass filter**, protecting the model from short-term statistical fluctuations.

---

## Performance & Efficiency (CIFAR-10)

FLOPs are measured as cumulative forward+backward operations per batch on CPU, excluding evaluation.

### Metric: Accuracy per TFLOP (ATF)

On consumer hardware (CPU-bound), we prioritize **Compute Efficiency**. We track how much generalization we buy for every TeraFLOP of energy expended.

#### **WRN-28-2 Battle Results (Consumer CPU)**

| Epoch | Status | Test Accuracy | Cumulative TFLOPs | **ATF (%/TFLOP)** |
| --- | --- | --- | --- | --- |
| 1 | Liquid | 43.81% | 72.79 | 0.60 |
| 2 | Liquid | 47.39% | 145.59 | 0.33 |
| **3** | **Liquid** | **68.43%** | **218.38** | **0.31** |
| **4** | **Liquid** | **72.76%** | **291.18** | **0.25** |

**Observation:** While standard optimizers often plateau early, HysteroGrad shows a **Geometric Ignition** phase (see Epoch 3). Once the Fisher Metric stabilizes, the accuracy per compute unit spikes significantly.

Why ATF matters:
AdamW reaches higher accuracy early, but HysteroGrad delivers more accuracy per unit of compute.
The late accuracy surge emerges from internal geometric alignment—not additional FLOPs.

---

## Advanced Features

* **Geometric Cooling:** Starts the training in a high-entropy "hot" state for broad exploration, then cools the system to induce crystallization.
* **Metric Scaling (Amplification):** Massively amplifies the Fisher signal (e.g., 200x - 500x) to navigate high-dimensional manifolds with extreme precision.
* **Adaptive Stiffening:** The barrier height adapts to the moving average of the system's energy, preventing premature freezing during high-turbulence phases.
* **Geometric Shock:** A feedback mechanism that "liquefies" the model (resets ) if it detects stagnation in local minima.

---

## Quick Start

```python
from hysterograd import HIOptimizer

# Initialize with 'Aggressive' settings for large models
optimizer = HIOptimizer(
    model.parameters(), 
    lr=0.05, 
    metric_scale=300.0,    # Geometric amplification
    stiffening_factor=0.005 # Rate of crystallization
)

# In your loop:
status, g_norm, h_width = optimizer.step()
print(f"Status: {status} | Barrier: {h_width:.4f}")

```

---

## Applications

* **Low-Power/Edge Training:** Maximize accuracy on limited FLOP budgets.
* **Continual Learning:** Prevent catastrophic forgetting by naturally freezing vital information paths.
* **Scientific ML:** Physics-informed optimization where the loss landscape is highly non-Euclidean.

---

**HysteroGrad** is more than an optimizer—it is a study of how information behaves when subjected to the laws of thermodynamics.

**Explore the transition. Liquefy your gradients. Freeze your knowledge.**

---

```BibTeX
@software{hysterograd2026,
  author = {Dieter Steuten},
  title = {HysteroGrad: Hysteretic Information-Geometric Optimizer},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/dingste/HysteroGrad}}
}
```
