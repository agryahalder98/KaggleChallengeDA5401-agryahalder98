# DA5401 End-Semester Data Challenge: Metric Learning for Conversational AI Evaluation
### Name: AGRYA HALDER

### Roll: ED25D900

## Project Goal

The primary objective of this project is to develop a robust **Metric Learning** model capable of predicting the fitness score (relevance) of a conversational AI agent's response against a specific evaluation metric definition. This is framed as a high-stakes classification problem, crucial for automating the testing and curation of high-quality AI evaluation datasets.

The final prediction is a score in the range **0–10**, and performance is evaluated using the **Root Mean Squared Error (RMSE)**.


## Methodology: TaskMet (Task-Driven Metric Learning)

To address the severe class imbalance and the semantic gap between metric definitions and text pairs, we implemented the **TaskMet** framework. This approach treats the prediction task as a **classification problem** guided by a bilevel optimization strategy that learns dynamic importance weights for each training sample.

### 1\. Feature Representation

The model input is a unified feature vector $x_i$ formed by concatenating the metric embedding and the text context embedding:

$$
x_i = [e_m ; e_c]
$$Where:

* $e_m$: Provided **Metric Definition Embedding** (768-dim, pre-computed using `embeddinggemma-300m`).
* $e_c$: **Text Context Embedding** (generated at runtime) for the concatenated string:
$$

```
$$T\_{context} = \\text{User Prompt} \\oplus \\text{System Prompt} \\oplus \\text{Response}
$$
$$
```

### 2\. Architecture

The system comprises two components:

1.  **Primary Classifier ($F_\theta$):** A multi-layer perceptron (MLP) trained to map $x_i$ to a probability distribution over 11 classes (scores 0-10).
2.  **MetricNet ($M_\phi$ or $\Lambda_\phi$):** An auxiliary MLP that learns to output positive **dynamic importance weights** ($w_{\text{metric}}$) for each training example.

### 3\. Metricized Balanced Loss

The training loss is a modified Cross-Entropy function that incorporates both static class balancing and dynamic metric weights:

$$
\mathcal{L}(\theta, \phi) = \text{CrossEntropy}(\hat{y}, y, w = \Lambda_\phi(x) \cdot w_{balance})
$$  * **$w_{\text{balance}}$ (Fixed Weights):** Inverse frequency weights (e.g., Effective Number formula) to statically counteract the heavy skew towards scores 9 and 10 identified in the EDA.
* **$\Lambda_\phi(x)$ (Dynamic Weights):** Learned by the MetricNet to dynamically up-weight "hard" or informative minority examples, thereby maximizing generalization.

### 4\. Bilevel Optimization

The training follows a meta-learning loop that optimizes the metric weights ($\phi$) based on the classifier's performance on the validation set:

$$\\phi^\* = \\arg \\min\_\\phi \\mathcal{L}*{val}(\\theta^*(\\phi))
$$$$
\\text{s.t. } \\theta^*(\\phi) = \\arg \\min*\\theta \\mathcal{L}\_{train}(\\theta, \\phi)
$$This unrolled optimization forces the MetricNet to learn a weighting scheme that directly improves the final generalization performance, rather than just minimizing the training error.


## 📁 Data Overview and Analysis

| Component | Files | Dimensions/Range |
| :--- | :--- | :--- |
| **Training Data** | `train_data.json` | 5,000 samples |
| **Test Data** | `test\_data.json` | 3,638 samples |
| **Metric Embeddings** | `metric\_name\_embeddings.npy` | $(145, 768)$ |
| **Target Score ($y$)** | `score` column (in training data) | Discrete $\in [0, 10]$ |

### Key EDA Insights

1.  **Score Skew:** The target variable is severely imbalanced, with $\sim 91\%$ of samples clustered at scores **9.0 and 10.0**. This dictated the use of classification and balanced weighting.

[Image of Target Score Distribution Histogram]

2.  **Linguistic Diversity:** The dataset is multilingual, dominated by **Hindi** ($n=2,496$) and **English** ($n=1,472$). The model must rely on the strong cross-lingual semantics of the `embeddinggemma` model.
3.  **Metric Skew:** Evaluation is biased towards safety-critical categories (**toxicity\_level**, **misuse**). Sparse, high-variance categories (e.g., **anonymization\_techniques**) pose a challenge to robustness.
4.  **Missing Data:** The **system\_prompt** is missing in $\sim 31\%$ (1,549) of training samples, necessitating robust input concatenation that handles missing fields.


## Setup and Usage

### Prerequisites

  * Python 3.8+
  * `pip`

### Installation

```bash
pip install torch, numpy, pandas, sentence-transformers, scikit-learn
```

### Data Structure

Place the downloaded data files (`metric_names.json`, `metric_name_embeddings.npy`, `train_data.json`, `test_data.json`) into a local directory, e.g., `./data/`.

### Execution

The primary execution script implements the two modes: `baseline` and `taskmet`.

**1. Training and Evaluating TaskMet (Bilevel Optimization)**

```bash
python train.py --mode taskmet --data_path ./data/ --output_dir ./submissions/taskmet/
```

**2. Training and Evaluating Baseline (Static Balanced Cross-Entropy)**

```bash
python train.py --mode baseline --data_path ./data/ --output_dir ./submissions/baseline/
```

The generated predictions to the specified output directory (e.g., `submissions_taskmet.csv`).
