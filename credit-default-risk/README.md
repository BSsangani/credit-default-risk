# Credit Card Default Risk Model (Probability of Default)

A probability-of-default (PD) model for **30,000 credit card customers**, built to rank customers by the risk of missing next month's payment and to support collections and credit-limit decisions.

**Tools:** Python (Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn), SQL (SQLite, NTILE window function)

## Key results
| Model (test set, 7,500 customers) | ROC-AUC | Gini | KS |
|---|---|---|---|
| Logistic Regression (scorecard-style) | 0.732 | 0.465 | 0.380 |
| **XGBoost (tuned, 5-fold CV)** | **0.781** | **0.561** | **0.428** |

- The **riskiest 20% of customers contain 50.9% of all defaulters** (top 10%: 31.6%, with a 69.9% default rate vs 3.6% in the safest 10%).
- Predicted PDs are **well calibrated**: e.g. top decile predicted 72.1% vs actual 69.9%.
- **Portfolio insights (SQL):** default rate rises from 13.8% for customers paying on time to 69.1% when 2 months late; customers with the lowest credit limits default 2.4× more than the highest (33.0% vs 13.7%); over-limit customers default at 30.1%.
- **Top risk drivers:** latest repayment status, worst arrears status, months in arrears, payment-to-bill ratio.

## Responsible modelling
- Sex, marital status and age are protected characteristics (Equality Act 2010), so they were **excluded from model inputs**.
- Outcome check by sex: mean predicted PD 22.0% (female) vs 22.9% (male), closely matching actual default rates (21.2% vs 23.5%).

## Data validation
- 30,000 records, no duplicates or missing values; default rate 22.1%.
- Merged **399 undocumented category codes** (345 EDUCATION, 54 MARRIAGE) into 'other'.
- Identified 590 customers with credit balances (negative bills) and handled them in ratio features.

## Files
- `Credit_Default_Risk.ipynb` — full analysis with outputs
- `sql/portfolio_queries.sql` — portfolio analysis queries
- `charts/` — risk deciles, ROC curve, feature importance

## Data
Yeh, I-C. (2009). Default of Credit Card Clients. UCI Machine Learning Repository, CC BY 4.0: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
