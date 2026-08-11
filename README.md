# Decline prediction and renewal pricing support tool

MSc Data Science final project, UWE Bristol  


## What this project is about

This project looks at renewal pricing for advertiser groups.

The issue I wanted to test was whether clients that are likely to see a meaningful drop in commission can be identified before renewal, and whether that risk can then be used to make a better decision on the mix of fixed and variable fees.

The target used in the modelling is a 10% or greater fall in commission over the following 12 months compared with the previous 12 months.

The work ended up covering three questions:

1. Can I predict which advertiser groups are likely to decline?
2. Can I also predict how large that decline is likely to be?
3. Can the predicted risk be converted into a pricing rule that performs better when it is applied to the commission that actually happened afterwards?

The final artefact is the set of five notebooks below. The pricing model is the end result, but the earlier notebooks are important because they show how I got there and which ideas were rejected along the way.

## Notebook order

The notebooks are numbered in analytical order and can be read straight through:

**01 → 02 → 03 → 04 → 05**

| Notebook | What it does |
|---|---|
| `01_Dataset_Creation_Final.ipynb` | Builds the monthly observation panel and the January 2025 modelling snapshot from transaction, advertiser, fixed-fee and publisher data. It also creates the forward 12-month target and the momentum, commercial and publisher features used later. |
| `02_Classification_Model_Final.ipynb` | Tests whether future commission decline can be predicted. It compares the existing 12-month YoY rule with Logistic Regression, Random Forest and XGBoost, then compares the full feature set with a much simpler three-month momentum model. It also includes GP-weighted metrics, SHAP, permutation importance and sensitivity checks. |
| `03_Severity_Model_Final.ipynb` | Tests whether the size of the future commission movement can be predicted. It compares direct regression with a two-stage classify-then-size approach. The main finding is that individual severity predictions are too noisy to use confidently in pricing, although the average outcome becomes much clearer when clients are grouped by classification risk. |
| `04_Pricing_Curve_Comparison_Final.ipynb` | Compares five ways of converting risk into a fixed-fee share: severity-based isotonic, risk-only linear, logistic, quantile step and hard threshold. The comparison is fitted on training data using out-of-fold risk scores and evaluated on the hold-out sample. |
| `05_Pricing_Model_Final.ipynb` | Applies the selected logistic pricing curve. It changes the fixed/variable mix while keeping GP the same at baseline commission, then applies the new terms to realised forward commission and compares the result with historical terms and a fully-variable alternative. |

## Main findings

The classification work changed the direction of the project quite a bit.

The existing 12-month YoY rule gives a GP-weighted F1 of **0.416**. A Random Forest using only fitted three-month momentum reaches **0.625**, while the full Random Forest reaches **0.626**. In other words, almost all of the improvement in the headline classification measure comes from using more recent momentum rather than from adding a large number of extra features.

The full model still changes the error mix in a useful way. At the selected thresholds it produces 117 fewer false-positive clients than the three-month model, representing about **€2.56m less false-positive GP**, although it also misses about **€1.10m more declining GP**. I therefore treat the value of the wider model as a commercial trade-off rather than claiming that 40 features materially improve the headline F1 score.

XGBoost has the highest single GP-F1 point estimate for the full feature set (**0.634**), but Random Forest is the model carried forward because the selection is based on GP-weighted AUC/ranking. Random Forest has the stronger GP-AUC (**0.638** versus **0.596** for XGBoost) and is also more stable in the sensitivity checks.

The severity work is mainly a negative result at individual-client level. The regressors show some relationship with future outcomes, but not enough to support an exact client-level severity forecast. The relationship is much clearer after grouping clients into risk bands. I use that as supporting evidence, but **individual severity prediction is not an input to the final pricing formula**.

Notebook 04 then tests the pricing shape itself. The unrestricted logistic optimisation keeps becoming steeper and starts to behave like a hard threshold. The hard-threshold method gives the highest hold-out GP point estimate at **+4.26%**, but it is an all-or-nothing rule around one cutoff. I therefore use the constrained logistic curve as the final method. Its fitted parameters are:

- midpoint: **0.51**
- steepness: **30**
- clients within two percentage points of either fee extreme: **39.9%**
- hold-out GP uplift: **+3.82%**

The hold-out sample is used to compare the pricing methods, so I treat this as a method-selection result rather than a completely untouched final test.

The final backtest in Notebook 05 produces:

- historical-terms GP: **€39.18m**
- risk-based terms GP: **€40.67m**
- risk-based uplift: **€1.49m / +3.82%**
- fully-variable GP: **€39.69m**
- fully-variable uplift: **€0.51m / +1.31%**

The mirror check is also important. The risk-based structure improves GP by around **16.9% for clients that subsequently decline**, while reducing GP by around **2.1% for clients that subsequently grow**. That trade-off is the point of the pricing model rather than an accidental side effect.

These are backtest results, not a claim that the same uplift would automatically be realised in live pricing. Client acceptance, negotiation and behaviour are outside the backtest and would need to be tested in a live pilot.

## How the notebooks connect

Notebook 01 creates:

`processed/modelling_dataset_jan25.csv`

Notebooks 02 to 05 all read this same modelling dataset.

They do **not** pass fitted model objects or saved prediction files from one notebook to another. Each notebook rebuilds the model it needs using the same feature definitions and random state. I did this so the modelling notebooks can be rerun independently once the processed dataset exists.

Notebook 01 also creates the rolling observation panel:

`processed/observation_panel.csv`

Notebook 02 reads this separately from the modelling dataset, specifically for the out-of-time check.

## Data

The source data is commercial Awin data and is not intended to be committed to the repository.

Notebook 01 expects the following files under the local `Data` folder:

- `2023TransactionsTable.csv`
- `2024TransactionsTable.csv`
- `2025TransactionsTable.csv`
- `Advertiser.csv`
- `FixedFees.csv`
- `Publisher.csv`

The publisher extract covers January 2024 to January 2025, so publisher features do not have the same historical depth as the transaction data. This is one of the limitations discussed in the project.

## Running the notebooks

The notebooks currently use my local project path:

`C:\Users\ian.forbes\FinalProject`

If they are being run somewhere else, the `BASE_PATH`, `DATA_PATH` and `OUT_DIR` definitions near the top of the notebooks need to be changed first.

To rebuild everything from the raw extracts:

1. Run `01_Dataset_Creation_Final.ipynb` to create the processed modelling data.
2. Run `02_Classification_Model_Final.ipynb` and `03_Severity_Model_Final.ipynb` for the modelling analysis.
3. Run `04_Pricing_Curve_Comparison_Final.ipynb` to reproduce the pricing-method comparison.
4. Run `05_Pricing_Model_Final.ipynb` for the final pricing backtest.

Once `processed/modelling_dataset_jan25.csv` exists, Notebooks 02 to 05 can be run separately.

Where randomness is involved, `RANDOM_STATE = 42` is used so that train/test splits and model results can be reproduced consistently.

## Main Python packages

The notebooks use standard Python data-science libraries:

- pandas
- numpy
- scikit-learn
- XGBoost
- SHAP
- SciPy
- matplotlib
- python-dateutil

## Outputs

The notebooks write their generated tables and charts into the project folders rather than requiring them as inputs for later modelling:

- `processed/` — datasets created by Notebook 01
- `outputs/classification_full/` — classification diagnostics and model comparisons
- `outputs/severity/` — severity-model results and charts
- `outputs/pricing_curve_comparison/` — pricing-method comparison
- `outputs/pricing/` — final pricing backtest outputs

The main artefact is the executed notebook set itself. The generated CSVs and charts are supporting outputs and can be recreated by rerunning the relevant notebook.
