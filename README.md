# Fake News Detection – NLP Pipeline

*Group 4 – Jan & Joachim*

## Overview

This project explores fake news detection using Natural Language Processing (NLP) techniques, combining classic and modern vectorization methods, and benchmarking multiple classifiers. The notebook details every step, from raw text preprocessing to model evaluation, in a reproducible and scalable pipeline.

Our approach: **Test, compare, and learn from the process, not just the results.** We don’t aim for textbook examples — we challenge and iterate.

---

## 1. Pre-Processing Pipeline

The pipeline is designed to be modular and robust, ready for adaptation or scale-up. Steps include:

* **Data Exploration:**
  Inspecting the dataset (`head()`, columns, basic stats).

* **Pre-tokenization Cleanup:**
  Removing noise, handling missing values, and cleaning text.

* **Tokenization:**
  Splitting text into words.

* **Lemmatization with WordNet (NLTK):**
  Leveraging part-of-speech (POS) tagging to improve lemma accuracy.

* **Vectorization:**

  * **TF-IDF:** Classical term frequency–inverse document frequency.
  * **Word2Vec Embedding:**
    Generating word embeddings; then, applying IDF weights to aggregate them for each document (hybrid approach).
  * **Standard Scaling:**
    Features are scaled independently for **TRAIN** and **TEST** sets (no leakage!).

---

## 2. Modelling

**Three models are compared:**

* **Logistic Regression:**

  * Tuned for `max_iter=1000` and `random_state=42`
  * **Accuracy:** 96.22%

* **Support Vector Machine (SVM):**

  * **Accuracy:** 90.51%

* **Naive Bayes:**

  * **Accuracy:** 50.90%

We use cross-validation and robust metrics throughout. All models are trained and evaluated with consistent pre-processing.

---

## 3. Results & Insights

* The **hybrid TF-IDF x Word2Vec** vectorization improves model performance.
* The pipeline shows the impact of each modelling choice, highlighting the gap between theory and actual project execution.
* **Project Management:**
  Collaboration, versioning, and iterative testing were key.
* **Pain Points:**
  We want bigger datasets and even more modelling challenges!

---

## 4. Usage

To run this notebook:

1. Install requirements (`pip install requirements.txt` if provided, or use standard NLP libraries: `nltk`, `scikit-learn`, `gensim`, `pandas`, etc.).
2. Download or provide the dataset in the expected location.
3. Execute the notebook sequentially; all steps are documented.
4. Adapt the pipeline to your own text classification challenge (just plug in your dataset!).

---

## 5. Forward Thinking

* **Scalability:**
  The pipeline is modular — swap models, vectorizers, or pre-processing as needed.
* **Explainability:**
  Next steps: integrate SHAP, LIME, or similar to better understand feature importance.
* **More Data:**
  The real world isn’t small; test on large, multilingual, or messy datasets.

---

## 6. Credits

* **Jan & Joachim**