# Decline prediction and renewal pricing support tool

MSc Data Science final project, UWE Bristol

## What this project is about

This project looks at renewal pricing for advertiser groups.

The main question was whether clients that are likely to see a meaningful decline in commission can be identified before renewal, and whether that risk can then be used to make a better decision on the mix of fixed and variable fees.

The modelling target is a **10% or greater fall in commission over the following 12 months compared with the previous 12 months**.

The project ended up covering three questions:

1. Can advertiser groups that are likely to decline be identified in advance?
2. Can the likely size of that decline also be predicted?
3. Can the predicted risk be turned into a pricing rule that performs better when applied to the commission that actually happened afterwards?

The final artefact is the five notebooks below (01-05). The pricing model is the end result, but the earlier notebooks are important because they show how the analysis developed, which ideas worked, and which ones were rejected. `EDA_Combined.ipynb` sits alongside them as supporting exploratory work, described below.

## Notebook order

The core modelling notebooks are numbered in analytical order and can be read straight through:

**01 → 02 → 03 → 04 → 05**

| Notebook | What it does |
| --- | --- |
| `01_Dataset_Creation_Final.ipynb` | Builds the monthly observation panel and the January 2025 modelling snapshot from transaction, advertiser, fixed-fee and publisher data. It creates the forward 12-month decline target and the momentum, commercial and publisher features used later. |
| `02_Classification_Model_Final.ipynb` | Tests whether future commission decline can be predicted. It compares the existing 12-month YoY rule with Logistic Regression, Random Forest and XGBoost, and then compares the full feature set with a much simpler model based mainly on three-month momentum. It also includes GP-weighted metrics, SHAP, permutation importance and sensitivity checks. |
| `03_Severity_Model_Final.ipynb` | Tests whether the size of the future commission movement can also be predicted. It compares direct regression with a two-stage classify-then-size approach. Individual severity prediction is too noisy to use confidently in pricing, although the average outcome becomes much clearer when clients are grouped by classification risk. |
| `04_Pricing_Curve_Comparison_Final.ipynb` | Compares five ways of converting predicted risk into a fixed-fee share: severity-based isotonic, risk-only linear, constrained logistic, quintile step and hard threshold. The comparison is fitted using training data and out-of-fold risk scores, then evaluated on the hold-out population. |
| `05_Pricing_Model_Final.ipynb` | Applies the selected logistic pricing curve. It changes the fixed/variable fee mix while keeping GP neutral at baseline commission, then applies the new terms to realised forward commission and compares the result with historical terms and a fully-variable alternative. |

### Supporting EDA

`EDA_Combined.ipynb` is not one of the five core files above, but the exploratory work that sits behind them and justifies several modelling decisions (e.g. the choice of a 10% decline threshold, the case for GP-weighting, and the publisher lower-funnel-reliance hypothesis referenced in Notebook 01). It combines what used to be two separate scripts and runs the same EDA across **two different populations**:

- the full rolling observation panel (`processed/observation_panel.csv`), covering all observation months, and
- the single January 2025 modelling snapshot (`processed/modelling_dataset_jan25.csv`), the actual population the models train and price on.

It can be read independently of the 01-05 sequence, though it depends on Notebook 01 having already produced both processed datasets above. Its charts and summary tables are written to `eda_outputs/`.

## Main findings

The classification work changed the direction of the project quite a bit.

The existing 12-month YoY rule gives a GP-weighted F1 of **0.416**. A fitted model using only three-month momentum reaches around **0.625**, while the full Random Forest reaches **0.626**.

So almost all of the improvement in the headline classification measure comes from using more recent momentum rather than from adding a large number of extra features.

The full model still changes the error mix in a useful way. At the selected thresholds it produces **117 fewer false-positive clients** than the reduced three-month model, representing around **€2.56m less false-positive GP**, although it also misses around **€1.10m more declining GP**.

I therefore treat the value of the full model as a commercial trade-off rather than claiming that the wider feature set materially improves headline F1.

XGBoost produces the highest single GP-weighted F1 point estimate for the full feature set at **0.634**, but Random Forest is carried forward because model selection is based on GP-weighted AUC and the quality of the continuous risk ranking.

Random Forest has the stronger GP-weighted AUC at **0.638**, compared with **0.596** for XGBoost, and is also more stable in the sensitivity checks.

### Severity

The severity work is mainly a negative result at individual-client level.

The regression models show some relationship with future outcomes, but not enough to support an exact client-level severity forecast. All models slightly beat a naive zero-change prediction on whole-population MAE, but the margin is small.

The relationship becomes much clearer after grouping clients into classification risk bands. I use this as supporting evidence that the classification score contains some information about magnitude as well as direction, but:

**individual severity prediction is not used as an input to the final pricing formula.**

### Pricing curve selection

Notebook 04 tests the pricing shape itself.

The unrestricted logistic optimisation repeatedly becomes steeper until it starts to behave almost like a hard threshold. The hard-threshold method produces the highest hold-out GP point estimate at **+4.26%**, but creates an all-or-nothing change around a single cutoff.

That was tested explicitly rather than left hidden inside a very steep logistic curve.

The final method is therefore a constrained logistic curve. Its fitted parameters are:

- midpoint: **0.51**
- steepness: **30**
- clients within two percentage points of either fee extreme: **39.9%**
- hold-out GP uplift: **+3.82%**

The hold-out population is also used to compare the pricing methods, so I treat this as a method-selection result rather than a completely untouched final test.

### Final pricing backtest

The final backtest in Notebook 05 produces:

- historical-terms GP: **€39.18m**
- risk-based terms GP: **€40.67m**
- risk-based uplift: **€1.49m / +3.82%**
- fully-variable GP: **€39.69m**
- fully-variable uplift: **€0.51m / +1.31%**

The split by realised outcome is also important.

The risk-based structure improves GP by around **16.9% for advertisers that subsequently decline**, while reducing GP by around **2.1% for advertisers that subsequently grow**.

That trade-off is the purpose of the pricing model rather than an accidental side effect. The model gives up some growth-side upside in exchange for greater protection when commission declines.

These are backtest results rather than a claim that the same uplift would automatically be realised in live pricing. Client acceptance, negotiation and behavioural response are outside the backtest and would need to be tested through a live pilot.

## How the notebooks connect

Notebook 01 creates:

`processed/modelling_dataset_jan25.csv`

Notebooks 02 to 05 all read this processed modelling dataset.

They do not pass fitted model objects or saved model predictions from one notebook to another. Each notebook rebuilds the model it needs using the same feature definitions and random state.

This means the modelling notebooks can be rerun independently once the processed modelling dataset exists.

Notebook 01 also creates the rolling observation panel:

`processed/observation_panel.csv`

Notebook 02 reads this separately for the out-of-time validation.

## Data

The source data is commercially sensitive Awin data and is not included in the repository.

This means the notebooks cannot be rerun against the real source data from the repository alone. The executed notebook outputs, charts and tables show the results of the completed pipeline, and the full pipeline can also be run locally against the original extracts.

Notebook 01 expects the following files under the local `Data` folder:

- `2023TransactionsTable.csv`
- `2024TransactionsTable.csv`
- `2025TransactionsTable.csv`
- `2026TransactionsTable.csv`
- `Advertiser.csv`
- `FixedFees.csv`
- `Publisher.csv`

The publisher extract covers January 2024 to January 2025, so the publisher features do not have the same historical depth as the transaction data. This is one of the limitations discussed in the project.

## Running the notebooks

The notebooks currently use my local project path:

`C:\Users\ian.forbes\FinalProject`

If they are being run somewhere else, the `BASE_PATH`, `DATA_PATH` and `OUT_DIR` definitions near the top of the notebooks will need to be changed first.

To rebuild the project from the raw extracts:

1. Run `01_Dataset_Creation_Final.ipynb` to create the processed modelling data.
2. Run `02_Classification_Model_Final.ipynb` for the classification analysis.
3. Run `03_Severity_Model_Final.ipynb` for the severity analysis.
4. Run `04_Pricing_Curve_Comparison_Final.ipynb` to reproduce the pricing-method comparison.
5. Run `05_Pricing_Model_Final.ipynb` for the final pricing backtest.

Once `processed/modelling_dataset_jan25.csv` exists, Notebooks 02 to 05 can be run separately.

Where randomness is involved, `RANDOM_STATE = 42` is used so that splits and model results can be reproduced consistently.

## Main Python packages

The notebooks use standard Python data-science libraries:

- pandas
- numpy
- scikit-learn
- XGBoost
- SHAP
- SciPy
- matplotlib

All the packages are in `requirements.txt` via an extract of the virtual environment. As noted above, the source data is not included in the repo due to data confidentiality,  so the notebooks cannot actually be re-executed without data access.

## Outputs

The notebooks write generated tables and charts into the project folders rather than relying on them as inputs for later modelling.

- `processed/`: datasets created by Notebook 01
- `outputs/classification_full/`: classification diagnostics and model comparisons
- `outputs/severity/`: severity results and charts
- `outputs/pricing_curve_comparison/`: pricing-method comparison outputs
- `outputs/pricing/`: final pricing backtest outputs
- `eda_outputs/`: charts and summary tables from `EDA_Combined.ipynb`

The main artefact is the executed notebook set itself. The generated CSVs and charts are supporting outputs and can be recreated by rerunning the relevant notebook.

## Important limitations

The main limitations to keep in mind are:

- the model predicts patterns associated with decline but cannot explain the underlying reason for that decline
- three relationship features are only available as current snapshots rather than historically dated values
- the fallback population is small and has not been through the same stability testing as the main model
- the same hold-out population is used for model and pricing-method comparison rather than having a separate final untouched test set
- the pricing result is a historical backtest and does not capture how clients may negotiate or change behaviour when presented with new terms
- the analysis is based on one affiliate network and one main modelling period, so wider generalisation has not been demonstrated

The intended use is therefore **decision support for renewal pricing**, not automated repricing.

A live pilot would be the next step to test whether the pricing recommendation produces the same type of benefit when clients actually receive and respond to the new terms.