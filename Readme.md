# Privacy-Preserving Federated Learning Pipeline for Pneumonia Detection

An end-to-end, privacy-preserving distributed deep learning system designed to diagnose Pneumonia from chest X-ray images. This framework utilizes **Flower (flwr)** for federated orchestration and **Opacus** for client-side Differential Privacy (DP), simulating a realistic, highly secure multi-hospital collaborative training network.

---

## 🏥 Clinical Problem Context & Non-IID Simulation

In medical AI, centralizing patient data from multiple hospitals is often impossible due to data privacy laws (HIPAA, GDPR) and proprietary constraints. However, training a model on data from only a single clinic results in poor generalization due to local biases.

This project addresses this by simulating a highly skewed, **Non-IID (Independent and Identically Distributed)** multi-hospital infrastructure using the PneumoniaMNIST dataset:

* **Hospital A (Pneumonia Hotspot):** Holds 80% of all positive Pneumonia cases, but only 10% of normal cases.
* **Hospital B (Community Screening Site):** Holds a balanced, smaller subset of leftovers.
* **Hospital C (Healthy Clinic Shadow):** Holds 80% of all normal, healthy cases, and only 10% of Pneumonia cases.

---

## 🛡️ Technical Architecture

The architecture relies on three foundational pillars to achieve distributed convergence without compromising data sovereignty:
┌─────────────────┐
       │  Flower Server  │◄────────────────────────┐
       └────────┬────────┘                         │
                │ Global Weights                   │ Local Weights
                ▼                                  │ & RDP Costs
┌─────────────────────────────────┐                 │
│      Hospital Client Nodes      │                 │
├─────────────────────────────────┴─────────────────┤
│  [Node 0]          [Node 1]          [Node 2]     │
│  (Hotspot)        (Balanced)         (Healthy)    │
│       │                │                │         │
│       ▼                ▼                ▼         │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐    │
│  │ Opacus  │      │ Opacus  │      │ Opacus  │    │
│  │ DP-SGD  │      │ Opacus  │      │ DP-SGD  │    │
│  └─────────┘      └─────────┘      └─────────┘    │
└───────────────────────────────────────────────────┘


1.  **Federated Orchestration (Flower):** Manages communication, weight aggregation (`FedAvg`), and client synchronization across the isolated nodes over secure gRPC channels.
2.  **Client-Side Differential Privacy (Opacus):** Implements DP-SGD by tracing per-sample gradients, clipping them to a flat ceiling (`max_grad_norm=1.0`), and injecting calibrated Gaussian noise before optimization steps occur.
3.  **Privacy Accounting (RDP):** Tracks the strict mathematical cumulative privacy spent ($\epsilon$) across local training epochs utilizing a memory-isolated Rényi Differential Privacy accountant.

---

## 📈 System Evolution & Current Status

### The Classifier Collapse Challenge (The Mode Collapse Trap)
During initial system tests, the extreme interaction between the Differential Privacy noise floor and the heavy Non-IID class skew caused **Classifier Collapse**. 

Because standard adaptive optimizers (like Adam) track historic gradient moments, the constant injection of Gaussian noise permanently corrupted their internal momentum buffers. The model's weights froze completely, forcing it to lazily guess the majority class ("Pneumonia") for 100% of inputs—yielding a deceptive 90.01% localized training accuracy but a flatline **0.00% Specificity** on an independent test set.

### Engineering Breakthroughs
We successfully broke out of the convergence trap by implementing three architectural corrections:
* **Swapped Adam for DP-Stable SGD with Momentum:** Replaced Adam with linear Stochastic Gradient Descent ($lr=0.01, \text{momentum}=0.9$), which naturally tolerates clipped, noisy gradients without buffer corruption.
* **Isolated Opacus Initialization:** Moved the `make_private()` mechanism strictly into the client's `__init__` constructor, ensuring gradient hooks are attached exactly once and eliminating multi-round re-wrapping recursion bugs.
* **Normalized Input Verification:** Verified and matched a hard image scaling constraint strictly between `[0.0, 1.0]` across both distributed training and standalone test matrices.

### Current Operational Metrics
Following optimization over 20+ global rounds, the model successfully began feature extraction, climbing out of the collapse trap:

```text
=============================================
🏥 CLINICALLY ALIGNED EVALUATION REPORT
=============================================
Overall Accuracy          : 75.80%
Sensitivity (Recall)      : 99.49%  <-- (Excellent safety margin for catching Pneumonia)
Specificity               : 36.32%  <-- (Successfully identifying Healthy cases)
---------------------------------------------
False Negatives (Misses!) : 2        <-- Clinically Low Risk
False Positives (Alarms)  : 149      <-- Boundary variance from DP Noise Floor
=============================================