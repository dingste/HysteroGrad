# Mathematical Theory of HysteroGrad

## 1. Introduction
HysteroGrad is an optimization framework that bridges **Information Geometry** and **Physical Systems Theory**. Unlike standard optimizers (SGD, Adam) that operate purely on local gradients, HysteroGrad treats the neural network's parameter space as a physical system with memory and material properties.

The core philosophy is based on the analogy between **information accumulation** in neural networks and **path functionals** in quantum mechanics. As the model "travels" through the parameter space, it accumulates "internal time" (\(\tau\)), causing the system to "stiffen" structurally. This mimics physical materials that undergo work hardening.

## 2. Core Components

### 2.1 Fisher Information Matrix (The Metric)
At the heart of HysteroGrad is the **Fisher Information Matrix (G)**, which acts as the metric tensor of the parameter space (Riemannian Geometry). It measures the local curvature of the loss landscape.
- **Role:** It scales the gradient steps, ensuring that updates represent equal changes in the *probability distribution* rather than just changes in parameter values (Natural Gradient Descent).

### 2.2 The Path Functional (\(\tau\)) - "Internal Time"
We define a scalar value \(\tau\), representing the **accumulated information** or the "geodesic distance" traveled by the model during training.
$$ d\tau = \sqrt{d\theta^T G d\theta} $$
- **\(\tau\) starts at 0** (Liquid state).
- **\(\tau\) increases** as the model learns significant features.
- **Role:** It acts as the "memory" of the system. The more the model learns, the larger \(\tau\) becomes.

### 2.3 Dynamic Hysteresis & Stiffening
The system implements a dynamic energy barrier that grows as a function of \(\tau\). This is the **Stiffening Functional**, $f(\tau)$.
- **Liquid Phase (Low \(\tau\)):** The barrier is low. The model is plastic and adapts rapidly to gradients.
- **Stiffening Phase (Rising \(\tau\)):** As information is acquired, the barrier rises. The system requires stronger evidence (larger gradients) to justify further changes.
- **Frozen Phase (High \(\tau\)):** The barrier exceeds the gradient force. The system locks into a stable configuration.

### 2.4 The IGD Schmitt-Trigger
HysteroGrad employs a mechanism inspired by the **Schmitt Trigger** in electronics. It acts as a non-linear switch that prevents the system from oscillating in flat minima or reacting to noise.
- **Condition:** An update is only performed if the "force" of the Natural Gradient exceeds the current hysteresis barrier.

## 3. The Update Rule
The mathematical governing equation for the parameter update $\Delta \theta$ is:

$$ \Delta \theta = \begin{cases} -\eta G^{-1} \nabla \mathcal{L} & \text{if } \|G^{-1} \nabla \mathcal{L}\| > f(\tau) \\ 0 & \text{otherwise (Frozen State)} \end{cases} $$

Where:
- $\eta$ is the learning rate.
- $G$ is the Fisher Information Matrix (or an approximation thereof).
- $\nabla \mathcal{L}$ is the gradient of the loss function.
- $f(\tau)$ is the stiffening function, typically modeled as $k \cdot \tau^\alpha$.

## 4. Phases of Optimization

### Phase 1: The Liquid State (Approx. Epoch 0-5)
* **Status:** High Plasticity.
* **Behavior:** The accumulated information \(\tau\) is small, so the hysteresis barrier is negligible. The optimizer behaves like a pure Natural Gradient Descent, converging extremely fast by following the curvature of the manifold.

### Phase 2: The Stiffening State
* **Status:** Work Hardening.
* **Behavior:** \(\tau\) grows. The barrier $f(\tau)$ begins to filter out small gradients. The model stops "chasing" minor fluctuations and focuses only on dominant structural updates.

### Phase 3: The Frozen State (Approx. Epoch 10+)
* **Status:** Stability / Convergence.
* **Behavior:** The barrier $f(\tau)$ is higher than the current gradient norm. The system enters the **Frozen** state ($\Delta \theta = 0$).
* **Benefit:** This provides immunity to noise and prevents "catastrophic forgetting" or oscillations at the bottom of a minimum. It effectively implements **automated early stopping**.

## 5. Theoretical Implications

### 5.1 Information Geometry
By using the Fisher Information Matrix, HysteroGrad respects the geometry of the probability space. The "Schmitt-Trigger" addresses a common instability in Natural Gradient Descent by freezing the system when the curvature becomes too flat or noisy, saving computational resources.

### 5.2 Neuromorphic Computing
The concept of **Hysteresis** is fundamental to memristive hardware. HysteroGrad is algorithmically compatible with neuromorphic chips (e.g., Intel Loihi), where physical devices naturally exhibit hysteresis. Instead of fighting this property, HysteroGrad leverages it for optimization.

### 5.3 Continual Learning
The path functional \(\tau\) offers a solution to **Catastrophic Forgetting**. In a continual learning setup, parameters with high accumulated information (high \(\tau\)) can be "frozen" to protect old knowledge, while new, plastic parameters remain "liquid" to absorb new tasks (similar to Elastic Weight Consolidation).