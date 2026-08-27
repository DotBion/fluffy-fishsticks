# 🚀 **FinPulse: Stock Sentiment-Aware Price Forecasting + MLStack**

A production-grade hybrid ML system that combines market data with real-time financial sentiment (from news, Twitter, etc.) to predict short-term stock price movements. Designed and deployed with full **MLOps** automation on Chameleon Cloud, **FinPulse** targets **quantitative hedge funds** and **alt-data research teams** seeking operational efficiency and faster signal-to-trade pipelines.

---

## 🔍 Value Proposition

**🎯 Target Customer:**  
Quantitative research teams at hedge funds like Two Sigma, Citadel, or Bloomberg and personal investors.

**📊 Status Quo:**  
These teams rely on fragmented workflows: NLP-based sentiment research, price modeling, and deployment are siloed. Retraining is ad hoc, and monitoring is minimal.

---

### 🛠️ Our Solution:

**FinPulse** automates the full ML lifecycle:

- 📰 Real-time ingestion of financial data (Twitter + Vantage)
- 💬 Sentiment extraction using **FinBERT**
- 📉 Price prediction via **LSTM**
- ⚙️ Serving via **FastAPI**
- 📈 Monitoring via **Prometheus + Grafana**
- 🔁 Retraining pipeline with **Ray + MLflow**

---

### 📐 Business Metric Impact:

- ✅ **Accuracy:** MAE, MAPE for classification metrics for sentiment  
- ⚡ **Inference Latency:** Tracked for CPU  
- 🧠 **Model Freshness:** Ensured via scheduled retraining and drift detection


## Contributors

| Name              | Responsibilities                                                | Commits                                                                 |
| ----------------- | --------------------------------------------------- | ------------------------------------------------------------------------|
| Aviraj Dongare    | CI/CD, Infra-as-Code                                | [View Commits](https://github.com/DotBion/fluffy-fishstick/commits?author=avirajdongare) |
| Ronit Gehani      | Data pipeline, Airflow, MinIO                       | [View Commits] |
| Deeptanshu Lnu    | LSTM training, Model Serving and optimizations, frontend, Endpoint Metrics              | [View Commits](https://github.com/DotBion/fluffy-fishstick/commits?author=deeptanshu20371), [unverified commits 1](https://github.com/DotBion/fluffy-fishstick/commit/3ed630d92dbbd24503c91fa22cddca49fc72ceee),[2](https://github.com/DotBion/fluffy-fishstick/commit/935f6cfff10cbb27de879ae70454cc87f85a1e39) |
| Nobodit Choudhury | Model Training, MLFlow                              | [View Commits](https://github.com/DotBion/fluffy-fishstick/commits/main/?author=DotBion) |


## 📏 Scale

Our project meets medium-scale criteria across all three axes—**data**, **model**, and **deployment**:

---

### 📊 Data Scale

- 🐦 **Tweet Dataset**  
  - [Company_Tweet.csv](https://www.kaggle.com/datasets/omermetinn/tweets-about-the-top-companies-from-2015-to-2020/data?select=Company+Tweet.csv)  
  - [Tweet.csv](https://www.kaggle.com/datasets/omermetinn/tweets-about-the-top-companies-from-2015-to-2020/data?select=Tweet.csv)

- 📈 **Market Data:**  
  Daily OHLC (open-high-low-close) and volume data from **Vantage**, typically spanning 10+ years, covering ~2500 data points per stock.

---

### 🧠 Model Scale

- 🤖 **[FinBERT](https://huggingface.co/ProsusAI/finbert):**  
  A large pre-trained transformer model (~110M parameters) from **ProsusAI** used for sentiment inference trained for tweets.

- 🔮 **Prediction Layer:**  
  LSTM over engineered time-series and sentiment features.


## Components Overview

| Component      | Technology Stack                |
| -------------- | ------------------------------- |
| Text Ingestion | Kaggle                          |
| Market Data    | Vantage                         |
| Model Training | LSTM, FinBERT                   |
| Serving        | Flask, Docker                   |
| Monitoring     | Prometheus, Grafana             |
| CI/CD          | GitHub Actions, Terraform, Helm |
| Persistence    | MinIO, Chameleon Volumes        |
| Infra-as-Code  | Terraform, Ansible, ArgoCD      |

## 1. 🚀 Continuous X & DevOps
 
### CI/CD Workflow

Our aim is to deploy an end-to-end **MLOps** infrastructure and workflow automation system on the Chameleon Cloud testbed. The goal was to support continuous training, containerization, and deployment of a custom LSTM model for stock price prediction. The implementation combined Infrastructure as Code (**IaC**) practices with Continuous X (training, building, deployment, and promotion) workflows using **Terraform**, **Ansible**, **Argo Workflows**, **ArgoCD**, **MLflow**, and **MinIO**.

---

## Section 1: Infrastructure as Code (**IaC**)

### 🛠 Work Executed

We provisioned a full-featured **MLOps** infrastructure on Chameleon Cloud using Infrastructure as Code (**IaC**) principles.

This included:

- A 3-node Kubernetes cluster using **Terraform**
- Automated configuration and deployment using **Ansible**
- Installation of core services such as **ArgoCD**, **Argo Workflows**, **MLflow**, **MinIO**, and **PostgreSQL**

### Implementation Approach and Methodology -

#### Keypair and Project Setup via [Trovi JupyterLab](https://jupyter.chameleoncloud.org) (KVM@TACC)

```python
from chi import server, context

context.version = "1.0"
context.choose_project()
context.choose_site(default="KVM@TACC")

server.update_keypair()
```
### 1. Terraform for Provisioning

- Defined cloud resources in `.tf` files (`main.tf`, `provider.tf`, `variables.tf`)
- Set variables such as instance count, keypair, network, and image
- Used `terraform init`, `terraform validate`, and `terraform apply` to spin up 3 VM instances on Chameleon Cloud
- Outputted static private IPs and floating public IPs

#### Provisioning with Terraform

Terraform configuration files were organized under `tf/kvm/`. They defined:

- 3 compute instances on Chameleon Cloud (node1, node2, node3)
- Networking setup using `sharednet1` and private subnet `192.168.1.0/24`
- SSH key management
- Floating IP allocation for node1 (entry point)

**Deployment steps:**

```bash
cd tf/kvm
terraform init
terraform apply -auto-approve
```
### 2. Ansible for Configuration

- Used Ansible playbooks to:
  - Install Kubernetes via **Kubespray**
  - Disable host firewalls and configure Docker
  - Set up networking, RBAC, and Helm

- Configured Ansible inventory with the provisioned IPs
- Verified node connectivity using Ansible’s ping module
- Executed playbooks like `pre_k8s_configure.yml`, `kubespray/cluster.yml`, and `post_k8s_configure.yml`

Ansible automation was organized in `ansible/` with subdirectories:

- `pre_k8s/`, `k8s/`, `post_k8s/` for system configuration and Kubernetes installation
- `argocd/` for managing **ArgoCD** apps and workflow orchestration

**Steps:**

```bash
cd ansible
ansible-playbook -i inventory.yml pre_k8s/pre_k8s_configure.yml
ansible-playbook -i inventory.yml k8s/kubespray/cluster.yml
ansible-playbook -i inventory.yml post_k8s/post_k8s_configure.yml
```
This fully bootstrapped a self-managed Kubernetes cluster with:

- `kube-apiserver` on node1
- `CoreDNS`, `kubelet`, and container runtime on all nodes

---

### 3. Platform Services with **ArgoCD + Helm**

- Used `argocd_add_platform.yml` to deploy **MLflow**, **MinIO**, and Postgres via Helm
- Injected runtime IP addresses (`externalIPs`) dynamically using Ansible facts
- Created secure secrets (**MinIO** access key and secret key) dynamically
- Confirmed deployment via the **ArgoCD** dashboard and service endpoints

Platform services were deployed using Helm charts located under `k8s/platform/`. These included:

- **MLflow** (`mlflow.yaml`)
- **MinIO** (`minio.yaml`)
- **PostgreSQL** (backend for **MLflow**)
- Kubernetes namespace definitions (`namespace.yaml`)
- Parameterized via `values.yaml`

Deployment was triggered via:

```bash
ansible-playbook -i inventory.yml argocd/argocd_add_platform.yml
```
The `argocd_add_platform.yml` playbook:

- Logged into **ArgoCD** via port-forwarding
- Dynamically detected the public external IP of node1
- Set up secrets for **MinIO** access
- Registered the Helm chart and set **externalIP** values via `--helm-set-string`

### Outcomes and System **Behavior** -

- Fully operational 3-node K8s cluster running on Chameleon Cloud
- **MLflow** accessible via [http://129.114.26.118:8000](http://129.114.26.118:8000)
- **MinIO** via [http://129.114.26.118:9001](http://129.114.26.118:9001)
- **IaC** templates and playbooks stored in version control for repeatable deployment

---

### 4. Argo Workflows: Continuous Training and Image Build

Workflow templates were defined in `workflows/`:

- `train-model.yml`: triggers model training via HTTP
- `build-container-image.yml`: clones repo, pulls model from **MLflow**, builds Docker image with **Kaniko**
- `deploy-container-image.yml`: deploys image to a specific environment
- `promote-model.yml`: uses **Skopeo** to retag and promote models

Workflow execution was triggered with:

```bash
ansible-playbook -i inventory.yml argocd/workflow_build_init.yml
```
This used the `build-initial.yml` template to:

- Clone the model repo from GitHub  
- Copy the latest model version into the image  
- Build and push version-tagged images (**staging-1.0.0**, **canary-1.0.0**, etc.)

### 5. Multi-Environment Deployment (Staging → Canary → Production)

Each deployment stage was configured using its own Helm chart under:

```bash
k8s/staging/
k8s/canary/
k8s/production/
```

Each contains:

- `Chart.yaml`, `values.yaml` 
- `templates/lstm-app.yaml`

Deployment commands:

```bash
ansible-playbook -i inventory.yml argocd/argocd_add_staging.yml
ansible-playbook -i inventory.yml argocd/argocd_add_canary.yml
ansible-playbook -i inventory.yml argocd/argocd_add_prod.yml
```

### Observations and Outcomes

| **Area**                | **Outcome**                                                                 |
|-------------------------|------------------------------------------------------------------------------|
| **IaC Provisioning**     | Cluster created in ~3 minutes via **Terraform**                              |
| **System Configuration** | Zero manual SSH config; fully automated with **Ansible**                     |
| **Platform Services**    | **MLflow** + **MinIO** accessible at floating IP; secrets securely injected  |
| **Model Training**       | LSTM models trained and logged to **MLflow**, with version metadata          |
| **Container Build**      | **Kaniko** used to build model-serving image inside cluster                  |
| **Argo Workflows**       | Chained workflows allowed model → build → deploy in one click                |
| **Deployment**           | Images deployed and promoted using **Helm** + **ArgoCD** with version tags   |

- All model lifecycles are now tracked and repeatable  
- No manual image builds or k8s edits needed post-push  
- Training + deployment can now be triggered via a single UI workflow in **Argo**


## 2. Data Pipeline

### Offline Pipeline

### Sources:

* Kaggle datasets (financial news, historical prices)
* Alpha Vantage API (daily stock price data for AAPL, MSFT)
* CSVs or JSON files from `/mnt/block/raw/` (manual uploads or external sources)

### Processing:

* ETL pipelines defined in Airflow DAGs:

  * Fetch raw data from APIs or Kaggle
  * Clean, deduplicate, and reformat data using Python (pandas)
  * Add derived features (moving averages, percent changes)
* DAGs scheduled or manually triggered via Airflow UI

### Storage:

* MinIO (S3-compatible) used as the main data repository for:

  * Cleaned and versioned datasets (e.g., `processed_data/aapl_2025-05-11.csv`)
  * Raw Kaggle/API dumps (e.g., `raw_data/raw_kaggle_dump.csv`)
* Persistent block storage on Chameleon (`/mnt/block`) for:

  * Training artifacts (models, checkpoints)
  * DAG logs, requirements, container images if necessary
  * Backup or sync of raw/processed MinIO data



### Offline Pipeline Overview & Flow:

### Node Setup:

* Conducted on node1 of Chameleon Cloud with floating IP for external access

### Dockerized Orchestration:

* Custom `docker-compose.yml` launching:

  * Airflow-webserver, Airflow-scheduler, Airflow-init containers
  * MinIO object store for persistent storage
* All services interconnected and managed via Docker

### Persistent Storage:

* 20GB persistent block storage volume mounted at `/mnt/block`
* Used to store intermediate and processed data
* Accessible by Airflow and Python containers

### Data Sources:

* Kaggle datasets stored locally
* Airflow DAG fetching updated stock data via Alpha Vantage API

### ETL via Airflow:

* Airflow DAGs fetch data (Kaggle/API), process it with Python operators, and upload cleaned data to MinIO
* Modular and reproducible DAGs scheduled via Airflow UI

### Data Repository:

* MinIO serves as central object store for raw and processed datasets

### EDA Dashboard:

* Dashboard developed for Exploratory Data Analysis
* Reads data from `/mnt/block/` or MinIO
* Provides visualizations, filters, and data quality summaries

## 3. Model Training

### Setup:

* Cleaned tweet dataset for FinBERT model
* Labeled tweets with sentiment scores using VADER
* Financial dataset appended with sentiment scores
* Target variables: next-day closing price, next-day sentiment prediction

### Models Used:

* FinBERT for sentiment prediction (`train_finbert.py`)
* LSTM for stock price prediction (`lstm_train.ipynb`, `lstm_train_pytorch.py`)
* VADER for dataset labeling (`tweets_2018_limited.csv`, `data_2018.csv`)

### Experiment Tracking:

* Hyperparameters tracked via MLflow hosted on CHI\@UC (`192.5.87.29`)
* Logged metrics: MAE, accuracy, latency
* Logged artifacts: model checkpoints, tokenizer, configuration (`lstm_model.pth`)

## 4. Model Serving

### API Endpoint:

* Hosted via Flask API
* LSTM Endpoint: `http://129.114.27.146:9090/predict`
* FinBERT Endpoint: `http://129.114.27.146:8080/predict`
* Prometheus:`http://129.114.27.146:9000/`
* Grafana: `http://129.114.27.146:3001/`
* To host the endpoints manually, run `python -m serving.app` (LSTM — `BACKEND=torch` or `BACKEND=onnx`), or `/train/inference_server_finbert.py` (FinBERT).

### LSTM:

* Input: List of prices in JSON format
* Output: Predicted next-day price (Python List)

### FinBERT:

* Input: JSON with recent tweets and market indicators
* Output: Sentiment score and confidence as Python dictionary

### Model Optimizations:

* FP-16 and Int8 Quantizations via ONNX Runtime with full graph optimizations
* Option to switch to Triton Inferencing and 4 Uvicorn workers for production ready concurrent deployement.
* To manually launch optimised onnyx LSTM endpoint using triton inference, run /App/src/optimised_LSTM.py. 


## 5. Evaluation & Monitoring

**Offline Evaluation:**

* Metrics: MAE, MAPE, BLEU, accuracy
* Dataset split: 70/15/15
* MLflow tracking

**Load Testing:**

* Artillery simulations (1K+ concurrent requests)
* Stability tested up to 150 RPS

**Drift Monitoring:**

* Embedding-based input drift alerts
* Grafana visualizations

## 6. Chameleon Cloud Resources

| Resource     | Usage Purpose               |
| ------------ | --------------------------- |
| gpu\_a100    | FinBERT fine-tuning         |
| m1.large     | MLflow, Ray, ETL            |
| m1.medium    | FastAPI serving, monitoring |
| Floating IP  | Public API access           |
| 100GB Volume | Models, data, logs          |





