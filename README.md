# domino-training-lending-club
Domino Training v6.2.0

# Domino User Training Workshop: Predicting Loan Default Risk

#### In this workshop you will work through an end-to-end ML workflow using LendingClub loan data to build a credit risk scoring system. The labs are broken into six sections:

* Upload raw training data to a Domino Dataset
* Explore and prepare the data in JupyterLab
* Train multiple models across different frameworks and compare performance in MLflow
* Deploy a real-time credit scoring API endpoint
* Build and share a loan officer dashboard with SHAP explainability
* Monitor your model for drift and trigger retraining
* Orchestrate the full pipeline with Domino Flows
* Build an agentic GenAI system that explains credit decisions using an LLM

---

# Section 1 — Project Setup

## Lab 1.1 — Fork the Project

***Documentation: [Fork Projects](https://docs.dominodatalab.com/en/latest/user_guide/ef261b/fork-projects/)***

Once you have access to the Domino training environment, navigate to the top Search menu and begin typing *LendingClubProject*. Select the project from the dropdown.

Read the readme to understand the use case and project structure.

In the top right corner, click the **Fork** icon. Name your project *LendingClub-Training-YourName*.

**You've created your first project!**

---

## Lab 1.2 — Update Project Settings

***Documentation: [Project Settings](https://docs.dominodatalab.com/en/latest/user_guide/dba65c/set-project-settings/)***

In your new project, click **Settings** in the bottom left.

Verify the default hardware tier is set to **Small** and the compute environment is set to **LendingClubProject-TrainingEnvironment**.

Navigate to the **Access and Sharing** tab. Set project visibility to **Private** and add your instructor or a fellow attendee as a collaborator.

**You've updated your project settings and added collaborators!**

---

## Lab 1.3 — Define Project Stages & Tasks

***Documentation: [Add Project Tasks](https://docs.dominodatalab.com/en/latest/user_guide/1f1064/set-project-stages/)***

Navigate to **Settings > Tasks** in the left pane. Keep the task stage **Ideation** since we are in the ideation stage of our project. 

Remove the remaining task stages by clicking the red bin icon next to each. 

Under **Add task stage** create the following three task stages:

| Task Stage Name
|---|
| Data Acquisition and Exploration |
| Model Training and Evaluation |
| Model Deployment |

***Documentation: [Add Project Tasks](https://docs.dominodatalab.com/en/latest/user_guide/4e60ef/add-project-tasks/)***

Navigate to **Govern > Tasks** in the left pane. Click **Add Tasks** and create the following three tasks:

| Task Name | Stage |
|---|---|
| Explore Loan Data | Data Acquisition and Exploration |
| Train Credit Risk Model | Model Training and Evaluation |
| Deploy Scoring API | Model Deployment |

Assign yourself as owner on each. Click **Save**.

**You've defined your project stages & tasks!**

---

# Section 2 — Execute Code

## Lab 2.1 — Add raw data to a Domino Dataset

***Documentation: [Add a Data Source to a Project](https://docs.dominodatalab.com/en/latest/user_guide/ba5bad/work-with-domino-datasets/)***

Navigate to **Data > Datasets** in the left pane. Click on your project dataset. 

Upload **lending_raw.csv**. You will see a new CSV file appear in your Dataset overview.

**You've added training data to your Domino project!**

---

## Lab 2.2 — Launch a Workspace

***Documentation: [Launch a Workspace](https://docs.dominodatalab.com/en/latest/user_guide/e6e601/launch-a-workspace/)***

Navigate to **Workspaces** in the left pane and click **Create New Workspace**.

- **Workspace IDE**: JupyterLab
- **Environment**: LendingClubProject (default)
- **Hardware Tier**: Small (default)

Click **Launch**.

**You've launched your workspace!**

---

## Lab 2.3 — Exploratory Data Analysis

Open `notebooks/EDA_lending.ipynb` from the file browser.

In the **Data** panel on the left, navigate to **Data Sources** and find the LendingClubWorkshop source. Copy the Python connection snippet into the first cell of the notebook and run it.

Work through the notebook cell by cell. The notebook will:

1. Load raw loan data from S3
2. Inspect missing values and data quality
3. Explore the target variable — default rate by loan grade, purpose, and home ownership
4. Plot feature distributions split by defaulted vs fully paid loans
5. Run a correlation heatmap to identify key predictors
6. Run the preprocessing pipeline (`scripts/preprocess.py`)
7. Verify engineered features and write the cleaned dataset to your Domino Dataset

Key insights to note as you run through the notebook:

- The default rate is approximately 20–25% across resolved loans — a moderately imbalanced classification problem
- **Interest rate** and **grade** are the strongest predictors — Grade E/F/G loans default significantly more
- **DTI** shows clear separation between defaulted and fully paid loans
- Engineered features (`loan_to_income`, `credit_utilization`) add meaningful signal

Rename the notebook `EDA_lending.ipynb` when complete.

**You've successfully explored the loan dataset!**

---

## Lab 2.4 — Sync Files

***Documentation: [Sync changes in a Workspace](https://docs.dominodatalab.com/en/latest/user_guide/262fef/sync-changes-in-a-workspace/)***

Click the **File Changes** tab in the top left corner of your workspace. Enter a commit message such as *"EDA notebook complete"* and click **Sync All Changes**.

Click **Stop** to stop the workspace instance.

**You've synced your changes!**

---

## Lab 2.5 — Complete Project Tasks

Navigate to **Artifacts** in your project. Click on `results` folder and select **Link to Task** on the notebook outputs. Choose the *Explore Loan Data* task.

Navigate back to **Tasks**, open *Explore Loan Data*, and mark it as complete.

**You've completed your first project task!**

---

## Lab 2.6 — Train Models with Domino Jobs

***Documentation: [Run a Job](https://docs.dominodatalab.com/en/latest/user_guide/af97b7/create-and-run-jobs/)***

Navigate to **Jobs** in your project and click **Run**.

Inspect `scripts/multitrain.py` in the Code browser first — it orchestrates three parallel training jobs:

- `scripts/train_sklearn.py` — Random Forest classifier with balanced class weights
- `scripts/train_xgboost.py` — XGBoost with `scale_pos_weight` for class imbalance
- `scripts/train_h2o.py` — H2O AutoML with up to 15 models

In the **File Name or Command** field enter:

```
scripts/multitrain.py
```

Click **Start**. Watch as three job runs appear. Click into the `train_xgboost.py` job run and inspect the **Results** tab once complete — you'll see confusion matrices and feature importance charts logged automatically.

**You've trained three credit risk models!**

---

## Lab 2.7 — Compare Experiments with MLflow

***Documentation: [Track and monitor experiments](https://docs.dominodatalab.com/en/latest/user_guide/da707d/track-and-monitor-experiments/)***

Navigate to **Experiments** in your project. Click on the **LendingClub-CreditRisk** experiment.

You'll see three runs — one per framework — each tagged with the framework name and logging four metrics: **AUC**, **F1**, **Precision**, and **Recall**.

Click **Edit** on the visualisation and add all four metrics to the chart so you can compare across frameworks at a glance.

Select all three runs using the checkboxes and click **Compare**. Review:

- Which framework achieves the highest AUC?
- How do precision and recall trade off across models?
- What are the top feature importances in each run?

The Sklearn Random Forest identified the most actual high-risk borrowers while maintaining competitive precision. It was also the fastest to train (14 seconds).
 
This is the recommended model from the comparison. We'll use this for deployment.  

Find the Train Credit Risk Model task and mark it as complete!

**You've compared experiments and selected your best model!**

---

# Section 3 — Deploy to Production

## Lab 3.1 — Deploy a Domino Endpoint

***Documentation: [Deploy Domino endpoints](https://docs.dominodatalab.com/en/latest/user_guide/8dbc91/deploy-domino-endpoints/)***

Navigate to **Deployments > Endpoints** and click **Create Domino endpoint (ML endpoint)**.

- **Name**: `lending-credit-risk-yourname`
- **Description**:
```
Credit risk scoring endpoint for LendingClub loan default prediction.

Sample scoring request:
{
  "data": {
    "loan_amnt": 15000,
    "int_rate": 13.5,
    "grade": "C",
    "annual_inc": 65000,
    "dti": 18.5,
    "home_ownership": "RENT",
    "purpose": "debt_consolidation",
    "term": "36"
  }
}
```
- **File**: `scripts/predict.py`
- **Function**: `predict`
- **Environment**: LendingClubProject
- Check **Log HTTP requests and responses to model instance logs**

Click **Create Endpoint**. The status will progress: Preparing → Building → Starting → Running (2–5 minutes).

**You've created a Domino scoring endpoint!**

---

## Lab 3.2 — Test the Endpoint

***Documentation: [Request Predictions](https://docs.dominodatalab.com/en/latest/user_guide/8dbc91/deploy-domino-endpoints/#Request-a-prediction)***

Once the endpoint is **Running**, navigate to the **Overview** tab and paste the sample request into the **Tester**:

```json
{
  "data": {
    "loan_amnt": 15000,
    "int_rate": 13.5,
    "grade": "C",
    "annual_inc": 65000,
    "dti": 18.5,
    "home_ownership": "RENT",
    "purpose": "debt_consolidation",
    "term": "36"
  }
}
```

The response will include:

| Field | Description |
|---|---|
| `default_probability` | Model's predicted probability of default (0–1) |
| `risk_score` | Inverted score (100 = safest, 0 = riskiest) |
| `risk_tier` | Low / Medium / High |
| `recommendation` | Approve / Review / Decline |
| `shap_values` | Top 5 features driving the decision (if shap was used) |

Try modifying the values and observe how the probability and recommendation change.

**You've tested your scoring endpoint!**

---

## Lab 3.3 — Deploy the Loan Officer Dashboard

***Documentation: [Publish Apps](https://docs.dominodatalab.com/en/latest/user_guide/71635d/publish-apps/)***

Before publishing the app, set the endpoint environment variables so the app can call your scoring API:

1. Navigate to your endpoint's **Overview** tab and copy the **Endpoint URL** and **API Key**
2. In your project, go to **Settings > Environment Variables** and add:
   - `DOMINO_MODEL_API_URL` — your endpoint URL
   - `DOMINO_MODEL_API_KEY` — your API key in Account Settings

Navigate to **Deployments > App**. Enter the title *YourName Loan Officer Dashboard* and click **Publish**.

Once active (1–3 minutes), click **View App**. You'll see:

- A **sidebar** containing the full loan application form — all inputs for loan details, applicant profile, and credit history
- Four metric cards across the top showing probability, risk score, risk tier, and recommendation
- A gauge chart visualising the default probability with green/amber/red bands
- A SHAP waterfall chart showing the top decision drivers — red bars increase default risk, green bars decrease it
- A **Raw API Response** expander at the bottom for inspecting the full JSON payload

Enter some test applications and observe how the SHAP chart changes as you adjust features like `dti`, `grade`, and `annual_inc`.

**You've deployed a loan officer dashboard!**

---

## Lab 3.4 — Share the App

***Documentation: [App Security and Identity](https://docs.dominodatalab.com/en/latest/user_guide/cb9195/app-security-and-identity/)***

Navigate to the **Access & sharing** tab in your app. Update permissions to **Anyone in Domino**.

Copy the app link from the **Settings** tab and ask your team mates to open it in new window. Confirm they can access the app provided they are logged into Domino.

Mark the *Deploy Scoring API* task as complete. You've completed 3 tasks in Domino!

**You've shared your loan officer dashboard!**

---

# Section 4 — Model Monitoring

## Lab 4.1 — Set Up a Model Monitor

***Documentation: [Domino Model Monitor](https://docs.dominodatalab.com/en/latest/user_guide/model-monitor/)***

Navigate to **Deployments > Model Monitor** and click **Set Up Monitor** on your endpoint.

Configure the monitor using the settings in `monitoring/drift_config.yaml`:

- **Prediction type**: Classification
- **Target column**: `is_default`
- **Positive class**: 1

Upload `monitoring/baseline_stats.json` as the baseline dataset (generated in the next lab).

---

## Lab 4.2 — Generate a Baseline and Define Drift Metrics

Run the baseline script as a Domino Job:

```
scripts/monitoring_baseline.py --mode baseline
```

Once complete, inspect the **Results** tab — you'll see the baseline feature distribution plots. These represent the "normal" state of incoming loan applications.

Back in Model Monitor, configure drift alerts for the following features using PSI thresholds:

| Feature | Priority | Alert at PSI | Retrain at PSI |
|---|---|---|---|
| `dti` | High | 0.10 | 0.20 |
| `int_rate` | High | 0.10 | 0.20 |
| `annual_inc` | High | 0.10 | 0.20 |
| `loan_to_income` | High | 0.10 | 0.20 |
| `revol_util` | Medium | 0.10 | 0.20 |

**You've configured drift monitoring!**

---

## Lab 4.3 — Simulate Drift

Run the drift simulation job to see alerts in action:

```
scripts/monitoring_baseline.py --mode simulate --shift-severity medium
```

Inspect the **Results** tab — you'll see a PSI bar chart colour-coded green/amber/red. Features above 0.2 PSI are flagged for retraining.

Navigate back to **Model Monitor** and observe the drift alerts that have been triggered. Discuss with your instructor: what real-world events might cause this kind of shift in loan application data?

**You've observed model drift in action!**

---

## Lab 4.4 — Trigger Retraining

From the Model Monitor alert view, click **Trigger Retraining** and select `scripts/multitrain.py` as the retraining job.

Once the job completes, navigate to **Experiments** and compare the new run against the previous baseline. Has the model performance changed?

**You've triggered a model retraining from a drift alert!**

---

# Section 5 — Domino Flows

## Lab 5.1 — Introduction to Domino Flows

***Documentation: [Domino Flows](https://docs.dominodatalab.com/en/latest/user_guide/flows/)***

Navigate to **Flows** in the left pane. Flows lets you define DAG-based pipelines that orchestrate multiple jobs with dependencies, conditional branching, and scheduling.

Review the pipeline DAG for this workshop:

```
[Ingest from S3] → [Preprocess] → [Train sklearn ]
                                → [Train XGBoost ] → [Evaluate AUC]
                                → [Train H2O     ]       ↓
                                              AUC > 0.80 → [Promote Endpoint]
                                              AUC ≤ 0.80 → [Alert & Hold]
```

Note that the three training steps run **in parallel** after preprocessing — Domino handles this automatically based on the `depends_on` configuration.

---

## Lab 5.2 — Build the Retraining Flow

Click **New Flow** and select **Import YAML**. Upload `flows/retraining_flow.yaml`.

Review each step in the Flow editor:
- **ingest** → **preprocess**: sequential dependency
- **train_sklearn**, **train_xgboost**, **train_h2o**: all depend on preprocess, run in parallel
- **evaluate**: depends on all three training steps
- **promote** / **alert_hold**: conditional branches based on `outputs.evaluate.auc`

Update the `project` field in the YAML to match your project name before importing.

**You've defined a retraining pipeline as code!**

---

## Lab 5.3 — Run and Monitor the Flow

Click **Run Flow**. Observe the DAG view as each step executes — you'll see the parallel training steps run simultaneously and the conditional branch fire once evaluate completes.

Click into individual steps to inspect their logs and outputs in real time.

**You've run a full ML retraining pipeline!**

---

## Lab 5.4 — Schedule the Flow

In the Flow editor, click **Schedule**. Set a monthly schedule (1st of each month at 06:00 UTC).

This represents production-ready continuous retraining — the pipeline will automatically ingest fresh data, retrain, evaluate, and only promote if quality thresholds are met.

**You've scheduled a recurring retraining pipeline!**

---

# Section 6 — GenAI & Agentic Systems

## Lab 6.1 — Introduction to GenAI in Domino 6.2

***Documentation: [GenAI in Domino](https://docs.dominodatalab.com/en/latest/user_guide/genai/)***

Domino 6.2 introduces native support for LLM integration, RAG pipelines, and agentic patterns. In this section you'll build a system that explains the credit model's decisions to loan officers in plain English — combining the ML model, SHAP values, and an LLM with access to company lending policy documents.

Review `genai/loan_explainer_agent.py`. The agent has two tools:

| Tool | Description |
|---|---|
| `score_loan` | Calls your Domino scoring endpoint to get default probability and SHAP values |
| `retrieve_policy` | Retrieves relevant policy documents based on the top SHAP risk factors |

---

## Lab 6.2 — Run the Loan Decision Explainer

Launch a workspace and open a terminal. Set your API key:

```bash
export ANTHROPIC_API_KEY=your-key-here
export DOMINO_MODEL_API_URL=your-endpoint-url
export DOMINO_MODEL_API_KEY=your-endpoint-key
```

Run the explainer with the default sample application:

```bash
python genai/loan_explainer_agent.py
```

Observe the agent's step-by-step reasoning in the logs:
1. It calls `score_loan` to get the model prediction and SHAP values
2. It calls `retrieve_policy` with the top risk factor feature names
3. It generates a structured plain-English explanation citing policy

Try the interactive mode to test with custom inputs:

```bash
python genai/loan_explainer_agent.py --interactive
```

**You've built and run an agentic credit decision explainer!**

---

## Lab 6.3 — Explore RAG with Policy Documents

Navigate to `genai/policy_docs/`. These text files simulate the company's internal lending policy manual.

Open `debt_to_income_policy.txt` — note the tiered DTI thresholds and compensating factors. Now run the agent with a high-DTI application and observe how it retrieves and cites this document in its explanation.

Discuss with your instructor:
- How would you extend this to a proper vector database (e.g. Pinecone, pgvector)?
- What happens if the policy changes — how do you keep the agent in sync?

---

## Lab 6.4 — Integrate the Explainer into the Dashboard

The loan officer dashboard (`app/app.py`) is designed to accept the SHAP values returned by `predict.py`. The agent in `loan_explainer_agent.py` can be called directly from the app to add a natural-language explanation panel beneath the SHAP waterfall chart.

As a stretch exercise, modify `scripts/app.py` to add:

1. An **"Explain Decision"** button below the SHAP chart
2. A call to `explain_loan_decision()` with the current loan inputs (triggered via `st.button`)
3. A text panel that renders the agent's explanation using `st.markdown`

This wires together the full stack: ML model → SHAP explainability → LLM explanation → loan officer UI.

**You've built a full GenAI-augmented credit decisioning system!**

---

### *** End of Labs ***

---

## Repository Structure

```
domino-training-lending/
├── data/                              # Sample dataset (50k rows)
├── notebooks/
│   └── EDA_lending.ipynb              # Exploratory data analysis
├── scripts/
│   ├── ingest.py                      # Step 1: Pull data from S3
│   ├── preprocess.py                  # Step 2: Feature engineering & cleaning
│   ├── multitrain.py                  # Orchestrates all model training
│   ├── train_sklearn.py               # Random Forest classifier
│   ├── train_xgboost.py               # XGBoost classifier
│   ├── train_h2o.py                   # H2O AutoML
│   ├── predict.py                     # Scoring function for endpoint
│   ├── evaluate.py                    # Model selection & AUC gate
│   ├── promote.py                     # Endpoint promotion via Domino API
│   ├── alert.py                       # Alert & hold on gate failure
│   ├── monitoring_baseline.py         # Baseline stats & drift simulation
│   ├── app.py                         # Streamlit loan officer dashboard
│   └── app.sh                         # App launch script
├── models/                            # Serialised model artifacts
├── monitoring/
│   ├── baseline_stats.json            # Generated baseline statistics
│   └── drift_config.yaml              # Model Monitor configuration
├── flows/
│   └── retraining_flow.yaml           # Domino Flows pipeline definition
├── genai/
│   ├── loan_explainer_agent.py        # Agentic LLM decision explainer
│   └── policy_docs/                   # Lending policy documents for RAG
├── results/                           # Job outputs and charts
├── Dockerfile                         # Domino Compute Environment definition
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## Sample Scoring Request

```json
{
  "data": {
    "loan_amnt": 15000,
    "int_rate": 13.5,
    "grade": "C",
    "annual_inc": 65000,
    "dti": 18.5,
    "home_ownership": "RENT",
    "purpose": "debt_consolidation",
    "term": "36",
    "installment": 350.0,
    "revol_bal": 12000,
    "revol_util": 55.0,
    "open_acc": 7,
    "total_acc": 18,
    "pub_rec": 0,
    "delinq_2yrs": 0,
    "verification_status": "Verified"
  }
}
```

## About

Repository of Domino Training Materials — LendingClub Credit Risk Workshop (Domino 6.2)
