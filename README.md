# Trust and Retention Intelligence

A reproducible customer attrition benchmark for the public BankChurners dataset. The Jupyter notebook is the single source for data checks, cleaning, EDA, preprocessing, model training, evaluation, and the model artifact loaded by Streamlit.

**Live app:** [Open the live app](https://trust-and-retention-intelligence.streamlit.app/)

**Status:** research benchmark and demonstration app. The public data contain no observation date, churn-event date, or future prediction horizon. Results do not establish future churn prediction or bank-specific performance.

## Research reviewed

- Brito et al. (2024), [A framework to improve churn prediction performance in retail banking](https://link.springer.com/article/10.1186/s40854-023-00558-3). The study uses 36 months of data from more than 3 million customers and about 170 million transactions at one bank. It motivates time-aware behavior features and minority-class evaluation; the bank data are private.
- [Calibrated, cost-sensitive, explainable ensemble learning for customer churn prediction in retail banking](https://pmc.ncbi.nlm.nih.gov/articles/PMC13627369/) (2026). This public-data study demonstrates leakage-safe preprocessing, model comparison, calibration, and threshold selection. Its static dataset also limits temporal claims.

These papers inform the workflow, but their data and results are not transferable to this project.

## Data and cleaning

The included `Data/BankChurners.csv` file comes from Sakshi Goyal's [Credit Card customers dataset on Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers). The Kaggle dataset page lists the data as CC0: Public Domain. It has 10,127 rows, 8,500 existing customers, and 1,627 attrited customers (16.1%). In this copy, the data contain no missing cells or duplicate rows. The notebook validates the schema and target, removes exact duplicate rows and rows without a target, converts numeric fields safely, and leaves any missing predictors to fold-fitted imputers.

Only 13 account and activity fields enter the model. `CLIENTNUM` and the two Naive Bayes output columns are excluded. Age and gender are reserved for subgroup audits. The other available public datasets do not provide a compatible dated churn outcome, so they are not used as external validation. **The German credit dataset is not used:** it describes credit risk rather than customer attrition and has no churn outcome. It was removed from the project; the only current input is `BankChurners.csv`.

### Current benchmark model inputs

The model predicts the supplied `Attrition_Flag` label using these fields:

| Field | Meaning |
|---|---|
| `Months_on_book` | Customer tenure on the account |
| `Total_Relationship_Count` | Number of products/relationships |
| `Months_Inactive_12_mon` | Inactive months in the last 12 months |
| `Contacts_Count_12_mon` | Customer-service contacts in the last 12 months |
| `Credit_Limit` | Credit limit |
| `Total_Revolving_Bal` | Revolving balance |
| `Avg_Open_To_Buy` | Available credit amount |
| `Total_Amt_Chng_Q4_Q1` | Transaction amount change, Q4 relative to Q1 |
| `Total_Trans_Amt` | Total transaction amount |
| `Total_Trans_Ct` | Transaction count |
| `Total_Ct_Chng_Q4_Q1` | Transaction-count change, Q4 relative to Q1 |
| `Avg_Utilization_Ratio` | Credit utilization ratio |
| `Card_Category` | Card product category |

The app derives available credit as credit limit minus revolving balance and rounds utilization to the three decimal places used in the source data. `Attrition_Flag` is the label (Existing Customer = 0; Attrited Customer = 1), not a model input. Age and gender are audit-only. The model does not use customer ID or the two target-derived Naive Bayes output columns.

The notebook records numeric input ranges and categorical values from the training partition in `outputs/data_audit.json`. Streamlit uses these to set the scoring form's limits and defaults. A calculated open-to-buy or utilization value outside the training range is not scored.

No included public file is sufficient to validate future churn performance for a real bank. A production study needs institution-approved longitudinal snapshots with a defined future label such as `churn_90d`, account and product status, dated transactions, engagement and service history, and closure events. Customer IDs must remain out of predictors, and repeated customers must not cross time-based train and test partitions. Campaign assignment, offer response, intervention cost, and customer contribution margin are needed to evaluate business impact; a randomized retention pilot is needed to establish causal benefit.

### Fields required for a real-world dataset

Create one row per eligible customer per scoring date. Every predictor must be available on that date; label the row only from outcomes in a later fixed horizon.

| Field group | Fields to collect | Purpose |
|---|---|---|
| Keys and dates | Pseudonymous customer ID, snapshot date, account open date, product closure date/status | Join snapshots and define time splits and outcomes; never score on the ID |
| Product relationship | Active products, product type, account status, tenure, balances | Describe relationship state at score time |
| Transactions | Date, amount, product, channel, transaction type; derived recency, frequency, and monetary aggregates over trailing 30/90/180-day windows | Capture prior behavior without future information |
| Engagement and service | Digital active days, last login, contacts, complaints, case outcomes and dates | Measure activity and service friction before scoring |
| Retention activity | Contact/offer dates, randomized treatment or control assignment, offer type, response, campaign cost | Measure intervention outcomes and cost |
| Customer value | Contribution margin or approved customer value and offer cost | Estimate net value rather than gross revenue |
| Audit-only attributes | Approved age band, gender, and other protected-group fields | Review errors and outcomes by group; keep separate from model inputs unless approved |
| Target | Churn event date and reproducible `churn_90d` label | Define the future outcome; never use it as an input |

## Benchmark design

- **Primary selection metric:** five-fold stratified cross-validated average precision (AP), appropriate for the 16.1% minority class.
- **Models:** Logistic Regression, Random Forest, and Histogram Gradient Boosting, compared with a prevalence-only Dummy baseline.
- **Partitions:** stratified 60% training, 20% validation, and 20% locked holdout. Tuning uses training folds only; validation selects the F1 threshold; the holdout is reserved for final reporting.
- **Holdout discipline:** compare candidates using training cross-validation only. Evaluate the calibrated selected model once on the locked holdout; do not use holdout scores to rank or revise candidates.
- **Leakage control:** imputing, scaling, and one-hot encoding are inside each model pipeline and fitted within each training fold.
- **Evaluation:** report AP, ROC-AUC, Brier score, F1, precision, recall, balanced accuracy, confusion counts, and a bootstrap 95% interval for holdout AP. Also report subgroup diagnostics and validation-set permutation importance.
- **Static benchmark screen:** the selected model must beat the Dummy baseline on CV AP, the lower bound of the holdout AP bootstrap interval must exceed holdout prevalence, and holdout Brier score must beat a constant training-prevalence prediction. This screen applies only to this static dataset and split; it is not evidence of temporal or cross-bank performance.

## Benchmark results

The current benchmark selected Histogram Gradient Boosting by mean cross-validated AP. Candidate models are compared on training cross-validation only. The locked holdout metrics below are from the selected calibrated model, evaluated once at a threshold chosen on validation data.

| Model | 5-fold CV AP mean ± SD | Selected |
|---|---:|:---:|
| Dummy baseline | 0.161 ± 0.000 | No |
| Logistic Regression | 0.737 ± 0.025 | No |
| Random Forest | 0.933 ± 0.019 | No |
| Histogram Gradient Boosting | **0.969 ± 0.007** | **Yes** |

| Selected calibrated model metric | Locked holdout |
|---|---:|
| Average precision | 0.959 (95% bootstrap interval 0.942–0.973) |
| ROC-AUC | 0.991 |
| Brier score | 0.025 |
| Training-prevalence baseline Brier score | 0.135 |
| F1 | 0.892 |
| Precision | 0.871 |
| Recall | 0.914 |
| Balanced accuracy | 0.944 |
| Validation-selected F1 threshold | 0.337 |
| Confusion counts | TN 1,657; FP 44; FN 28; TP 297 |

The static benchmark screen was met on this public static-label split. These scores are not proof of future prediction: the data label existing attrition status and have no timestamps or forecast horizon. The threshold is not a bank-approved outreach decision rule. The project does not claim revenue saved or causal retention impact. The reported bootstrap interval describes uncertainty on this holdout only, not across time, banks, or populations.

The notebook displays the benchmark table and evaluation metrics when run and writes `outputs/benchmark.csv` and `outputs/test_metrics.json`. Re-run it after changing the input data or methodology; update the table here from that run before reporting new results.

## Run the project

Use Python 3.11 with the pinned application and notebook packages:

    python -m pip install -r requirements.txt
    jupyter lab

From the project folder, open `Model.ipynb` in JupyterLab and run all cells from top to bottom. This fits the benchmark models and saves `models/churn_pipeline.joblib` plus the evaluation reports. Then start the app in a terminal:

    streamlit run app.py

Streamlit loads the model and feature schema saved by the notebook. If the artifact is missing, run the notebook first.

## Project files

| Path | Purpose |
|---|---|
| `Model.ipynb` | Complete data-science workflow and benchmark; source of the fitted model |
| `app.py` | Streamlit profile scorer and benchmark results view |
| `Data/BankChurners.csv` | Public static-label benchmark data |
| `models/churn_pipeline.joblib` | Notebook-produced calibrated model, feature list, threshold, and metrics |
| `outputs/` | EDA, benchmark, holdout, subgroup, and feature-importance reports |
| `requirements.txt` | Pinned application and model dependencies |
| `CONTRIBUTING.md` | Contribution workflow and review expectations |

The notebook creates `outputs/data_audit.json`, EDA tables and figures, `outputs/benchmark.csv`, `outputs/test_metrics.json`, `outputs/subgroup_audit.csv`, `outputs/feature_importance.csv`, and `outputs/holdout_precision_recall.png`.

## Requirements before a real-world pilot

Obtain approved dated bank data and define churn consistently; create time-based and customer-grouped validation; evaluate calibration, outreach-capacity precision/recall, subgroup behavior, and drift; run a randomized offer/control pilot; and set decisions using customer value, offer cost, capacity, and measured treatment effects. Privacy, security, compliance, and model-governance owners must approve the intended use.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and checks expected for changes. Never commit private customer data, credentials, environment folders, or notebook checkpoints. Model changes should preserve the training-only selection and locked holdout method, rerun the notebook, and update the benchmark results here.

There is not yet a project-level `LICENSE` for the source code. The dataset's CC0 status applies to the dataset, not automatically to this repository's code; the project owner should choose a code license before encouraging broad reuse.
