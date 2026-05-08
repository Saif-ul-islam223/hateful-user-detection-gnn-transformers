This project builds an end‑to‑end machine learning pipeline to detect hateful or suspended users on a social network.
It compares classical ML models, transformer‑based embeddings (BERT & RoBERTa), and Graph Neural Networks (GCN & GraphSAGE) on the same dataset.

The goal is to understand how text‑based embeddings and graph structure can be combined to improve harmful‑user detection.

The dataset contains around 2,400 Twitter users, each represented as a node with profile text and a binary label indicating whether the user is hateful/suspended (1) or non‑hateful/active (0). User text is used to generate transformer embeddings, while edges represent social connections between users. The dataset is imbalanced, with 387 hateful users and 2,013 non‑hateful users, which makes F1‑score sensitive to class distribution and makes ROC‑AUC a more reliable metric for evaluating model performance.

🧠 Project Pipeline
1. Text Embedding Generation
User text is encoded using:

BERT‑base‑uncased

RoBERTa‑base

RoBERTa embeddings were stronger due to:
more training data
dynamic masking
no NSP
byte‑level BPE tokeniser

2. Cosine Similarity Graph Construction
A new graph is built using:
cosine similarity between embeddings
edges added above a similarity threshold
This captures semantic similarity between users.

3. Classical ML Models
Trained on transformer embeddings:
Logistic Regression
Support Vector Machine
Random Forest

4. Graph Neural Networks
Trained on the constructed graph:
GCN
GraphSAGE

5. Evaluation Metrics
ROC‑AUC → ranking ability across all thresholds
F1‑score → final classification quality
Both are needed due to dataset imbalance.

🏆 Results
⭐ 1. RoBERTa Outperformed BERT
RoBERTa produced better embeddings because:
trained on 10× more data
dynamic masking
no Next Sentence Prediction
better tokenisation for noisy text
This improved both ROC‑AUC and F1.

⭐ 2. Classical ML Model Results
Logistic Regression performed the best among the classical models, achieving high ROC‑AUC and stable F1‑scores because it works well with dense transformer embeddings and is less prone to overfitting. In contrast, SVM struggled due to the dataset’s class imbalance and required more tuning to achieve competitive performance.

Lower F1

Random Forest
Inconsistent

Trees don’t exploit embedding geometry well

⭐ 3. GNN Results
GraphSAGE (Best Overall Model)
GraphSAGE outperformed GCN because:
better neighbourhood aggregation
handles sparse graphs well
avoids oversmoothing
inductive learning

GCN
Sensitive to graph density
Oversmoothing on deeper layers
Less robust to noisy edges

