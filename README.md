# Google Cloud Certification Prep Playbook

![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Skill%20Boost-blue?logo=googlecloud)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)
![Status](https://img.shields.io/badge/Status-Active%20Learning-success)

> A curated collection of Google Cloud Skill Boost labs, hands-on notebooks, and supporting assets used while preparing for Google Cloud certification exams.

## Table of Contents
- [Overview](#overview)
- [Learning Objectives](#learning-objectives)
- [Repository Map](#repository-map)
- [Getting Started](#getting-started)
  - [Set Up Your Environment](#set-up-your-environment)
  - [Authenticate With Google Cloud](#authenticate-with-google-cloud)
- [Recommended Learning Paths](#recommended-learning-paths)
- [Practice Checklist](#practice-checklist)
- [Additional Resources](#additional-resources)
- [Contributing & Feedback](#contributing--feedback)
- [License & Attribution](#license--attribution)

## Overview
- **Audience:** Cloud practitioners preparing for Google Cloud professional-level exams who prefer hands-on labs.
- **Scope:** Vertex AI, BigQuery ML, TensorFlow/Keras, TFDV, computer vision, embeddings, recommendation systems, and MLOps workflows.
- **Format:** Jupyter notebooks sourced from Google Cloud Skill Boost labs, plus auxiliary datasets and scripts to reproduce the exercises locally or in Google-managed notebooks.
- **Outcome:** Build confidence with real exam scenarios by practicing end-to-end workflows—data engineering, model training, deployment, and monitoring on Google Cloud.

## Learning Objectives
- Reinforce core Google Cloud ML concepts: Vertex AI Pipelines, training, tuning, deployment.
- Master BigQuery ML modeling patterns (classification, recommendation, WALS/ALS hybrids).
- Practice TensorFlow and Keras feature engineering for structured, image, and text workloads.
- Validate and monitor data quality with TensorFlow Data Validation (TFDV).
- Accelerate experiments on TPUs and managed notebooks.
- Document lab takeaways and personal notes as you iterate through the exam blueprint.

## Repository Map
**Vertex AI & MLOps**
- [`1_training_at_scale_vertex.ipynb`](./1_training_at_scale_vertex.ipynb)
- [`classifying_images_with_pre-built_tf_container_on_vertex_ai.ipynb`](./classifying_images_with_pre-built_tf_container_on_vertex_ai.ipynb)
- [`train_deploy.ipynb`](./train_deploy.ipynb)
- [`tpu_speed_data_pipelines.ipynb`](./tpu_speed_data_pipelines.ipynb)

**Computer Vision Labs**
- [`classifying_images_with_transfer_learning.ipynb`](./classifying_images_with_transfer_learning.ipynb)
- [`classifying_images_using_data_augmentation-Copy1.ipynb`](./classifying_images_using_data_augmentation-Copy1.ipynb)
- [`classifying_images_with_a_linear_model.ipynb`](./classifying_images_with_a_linear_model.ipynb)
- [`classifying_images_using_dropout_and_batchnorm_layer.ipynb`](./classifying_images_using_dropout_and_batchnorm_layer.ipynb)
- [`classifying_images_with_a_nn_and_dnn_model.ipynb`](./classifying_images_with_a_nn_and_dnn_model.ipynb)

**Feature Engineering & Data Pipelines**
- [`1_bqml_basic_feat_eng.ipynb`](./1_bqml_basic_feat_eng.ipynb)
- [`3_keras_basic_feat_eng.ipynb`](./3_keras_basic_feat_eng.ipynb)
- [`content_based_preproc.ipynb`](./content_based_preproc.ipynb)
- [`feature_store_streaming_ingestion_sdk.ipynb`](./feature_store_streaming_ingestion_sdk.ipynb)

**Data Validation & Governance**
- [`tfdv_basic_spending.ipynb`](./tfdv_basic_spending.ipynb)
- [`tfdv_advanced_taxi.ipynb`](./tfdv_advanced_taxi.ipynb)

**Natural Language & Embeddings**
- [`keras_for_text_classification.ipynb`](./keras_for_text_classification.ipynb)
- [`reusable_embeddings.ipynb`](./reusable_embeddings.ipynb)

**Recommendation Systems**
- [`als_bqml_hybrid.ipynb`](./als_bqml_hybrid.ipynb)
- [`wals.ipynb`](./wals.ipynb)

**Sequence Models & RNNs**
- [`rnn_encoder_decoder.ipynb`](./rnn_encoder_decoder.ipynb)

**Multi-Framework & General Labs**
- [`multiple_frameworks_lab.ipynb`](./multiple_frameworks_lab.ipynb)
- [`keras.ipynb`](./keras.ipynb)
- `lab-01.ipynb` → `lab-04.ipynb` for staged practice sets.

**Supporting Assets**
- `perform fundation/` — supplemental datasets, shell scripts, and reference images used across the labs.

> Tip: Each notebook is self-contained. Start with the ones matching your current study objective, then revisit and annotate with personal insights aligned to exam topics.

## Getting Started

### Set Up Your Environment
1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-account>/<your-repo>.git
   cd <your-repo>
   ```
2. **Create an isolated Python environment** (conda or venv)
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```
3. **Install core dependencies** (extend as needed per lab)
   ```bash
   pip install jupyterlab google-cloud-bigquery google-cloud-aiplatform \
       google-cloud-storage tensorflow tensorflow-datasets scikit-learn pandas matplotlib
   ```
4. **Launch JupyterLab**
   ```bash
   jupyter lab
   ```

> Prefer working in Google Cloud? Open the repo in Cloud Shell or Vertex AI Workbench where most dependencies are preconfigured.

### Authenticate With Google Cloud
- Set your active project: `gcloud config set project <PROJECT_ID>`
- Authenticate: `gcloud auth login` (local) or `gcloud auth application-default login`
- Enable required services: BigQuery, Vertex AI, AI Platform Notebooks, Cloud Storage
- Assign IAM roles aligned with the labs (Viewer, BigQuery Admin, Vertex AI User, Storage Admin)

## Recommended Learning Paths
- **Foundation (Week 1):** `lab-01.ipynb` → `lab-04.ipynb`, `keras.ipynb`, `content_based_preproc.ipynb`
- **Data Warehouse & BQML (Week 2):** `1_bqml_basic_feat_eng.ipynb`, `als_bqml_hybrid.ipynb`, `wals.ipynb`
- **Model Development (Week 3):** Computer vision series, `keras_for_text_classification.ipynb`, `reusable_embeddings.ipynb`
- **MLOps & Production (Week 4):** Vertex AI labs, `feature_store_streaming_ingestion_sdk.ipynb`, `tpu_speed_data_pipelines.ipynb`
- **Quality & Monitoring (Week 5):** TFDV notebooks, revisit Vertex AI deployment playbooks

Adapt the cadence to your exam schedule—capture notes, questions, and command snippets directly inside each notebook markdown cell.

## Practice Checklist
- [ ] Review the official exam guide and map topics to notebooks
- [ ] Enable audit logs for services touched in labs
- [ ] Practice deploying a model to Vertex AI endpoints (rest/SDK)
- [ ] Create custom training jobs with GPUs and TPUs
- [ ] Build at least one BigQuery ML pipeline with scheduled queries
- [ ] Run TFDV against drift scenarios and export validation summaries
- [ ] Benchmark data ingestion patterns into Vertex AI Feature Store
- [ ] Conduct a mock review of model compliance (explainability, fairness)

Use GitHub Projects, Issues, or your favorite tracker to log observations and questions for follow-up study sessions.

## Additional Resources
- [Google Cloud Certification Hub](https://cloud.google.com/certification)
- [Skill Boost Catalog](https://www.cloudskillsboost.google/catalog)
- [Google Cloud Architecture Framework](https://cloud.google.com/architecture/framework)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [BigQuery ML Guides](https://cloud.google.com/bigquery-ml/docs)
- [MLOps on Google Cloud](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

## Contributing & Feedback
- This repository reflects personal study notes. Feel free to adapt the notebooks, but keep original lab attributions intact.
- Suggestions or pull requests that enhance clarity, add commentary, or include practice questions are welcome.
- Open an issue for broken links, dependency updates, or requests for additional lab coverage.

## License & Attribution
- The notebooks reference content from Google Cloud Skill Boost labs. Respect the [Skill Boost Terms of Service](https://www.cloudskillsboost.google/terms-of-service) when reusing.
- Unless otherwise noted, materials are shared for personal educational use. Add a `LICENSE` file if you plan to distribute or collaborate publicly.
- Trademark notice: Google Cloud, Vertex AI, and related product names are trademarks of Google LLC.

