# Visual Emotion Analysis: Predicting Emotions in Pictures

This repository contains the codebase and presentation materials for the final project in the **Machine Learning for Neuroscience** course. The project evaluates two machine learning modeling pipelines—a sparse tabular model and a dense embedding model—to predict discrete human emotional categories from image data.

## Authors
* **Tzlil Tabib** - [GitHub Profile](https://github.com/TzlilTabib)
* **Gaia Negev** - [GitHub Profile](https://github.com/GaiaNegev13)


---

## Technical Stack & Tooling

The core focus of this project was the practical application of diverse data engineering, NLP, and machine learning toolkits:

* **Feature Extraction & Data Representation:**
  * **OpenAI `gpt-4o-mini`:** Leveraged to generate highly descriptive textual annotations of image content, visible objects, settings, and synthetic subjective viewer feelings.
  * **OpenAI `text-embedding-3-small`:** Used to extract dense, high-dimensional text embeddings from both objective image descriptions and subjective feelings strings.
  * **Open-CLIP (CLIP Architecture):** Deployed to extract high-dimensional visual feature vectors directly from the raw images.

* **Machine Learning & Pipeline Engineering:**
  * **Scikit-Learn (`sklearn`):** The primary engine utilized for preprocessing (median data imputation, `LabelEncoder`, `StandardScaler`), feature selection, and model training.
  * **TF-IDF Vectorization:** Custom text vectorization engineered to unify separate textual descriptions, feelings, and objects into a single sparse numerical matrix, maintaining prefixed semantic markers (e.g., `obj_tree`, `feel_awe`) for downstream tracking.
  * **Dimensionality Reduction:** Automated optimization using **Principal Component Analysis (PCA)** and **t-SNE** to project high-dimensional text and image embeddings into structured, lower-dimensional manifolds.
  * **Unsupervised Clustering:** Implementation of **K-Means Clustering** accompanied by automated evaluation via the **Elbow Method** and **Silhouette Analysis** to map structural separations in the embedding space.

* **Model Diagnostics & Explanations:**
  * **SHAP (SHapley Additive exPlanations):** Applied to the selected $L_2$-regularized Logistic Regression classifier to calculate average SHAP values by emotion class, providing full feature-level transparency for the tabular pipeline.

---

## Project Architecture & Pipeline Workflow

### 1. Exploratory Data Analysis & Preprocessing
* **Data Cleaning:** Conducted missingness profiling (null analysis) to drop low-quality variables missing substantial sample volumes (`facial_expression`, `human_action`, `scene`). Continuous numeric variations (`colorfulness`, `brightness`) were handled via median imputation.
* **Unsupervised Structural Check:** Used the Elbow Method and Silhouette scoring to assess the natural clustering properties of the high-dimensional embedded space against the 8 ground-truth categorical emotion labels.

### 2. Tabular Pipeline vs. Embedding Pipeline
The dataset was processed through an 80/20 train/test split, with all hyperparameter thresholds optimized strictly via cross-validation:

* **The Tabular Model:** Unified the text sequences via an engineered TF-IDF grid. An Elbow Method threshold based on marginal cross-validated AUC gains ($<0.005$ per step) determined the optimal inclusion threshold at the top 90 text features. We benchmarked multiple model families (including baseline dummies and KNNs), ultimately selecting an **$L_2$ Regularized Logistic Regression** model.
  * *Best CV AUC:* **0.920**
* **The Embedding Model:** Bypassed traditional text parsing by directly ingesting Open-CLIP image vectors alongside OpenAI semantic text embeddings. Principal components ($N$ PCs) per embedding type were dynamically isolated using cross-validated logistic regression performance to explain variance efficiently without introducing overfitting.

---

## Key Engineering Performance Insights

* **Embedding Superiority:** Multi-modal embedding models consistently demonstrated higher cross-validated AUC metrics than sparse tabular text models.
* **Feature Drivers:** SHAP value calculations proved that low-level perceptual inputs (`colorfulness` and `brightness`) combined with GPT-generated subjective viewer descriptions carried the highest predictive weight in the tabular architecture.
* **Classification Topography:** Model diagnostic tools (confusion matrices) highlighted that while both modeling approaches easily mastered macro-level valence discrimination (separating positive vs. negative emotions), finer-grained 8-class boundaries remained highly mixed—fully mirroring the data overlapping observed during the initial t-SNE and K-Means EDA steps.

---

## References
1. Yang, J. et al. (2023). "EmoSet: A large-scale visual emotion dataset with rich attributes." *Proceedings of the IEEE/CVF ICCV*.
2. Mikels, J. A. et al. (2005). "Emotional category data on images from the International Affective Picture System." *Behavior Research Methods*.
3. Open-CLIP GitHub Repository: `https://github.com/mifoundations/open_clip`.
