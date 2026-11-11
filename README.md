# Baseline Predictive Pipeline -- ETAI
20231631 Beatriz Partidário

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Weekly results

### Week 2
Overall, logistic regression performs better than the decision tree. The decision tree achieves 82.9% accuracy on the training set but only 62.9% on the test set, which suggests overfitting. Logistic regression achieves about 67.9% accuracy on both sets, showing better generalization.
It also performs better for class 1, with higher precision, recall, and F1-score. Although the decision tree shows a smaller difference in false positive rates between the two main racial groups, logistic regression is more reliable overall because of its stronger and more consistent performance on unseen data.

### Week 3 


### Week 4

Compared with the previous week, both models now use 5-fold cross-validation, making the evaluation more reliable than a single train/test split. Logistic regression remained stable at about 0.67 accuracy and showed almost no overfitting, with the gap decreasing from 0.008 to 0.003. The decision tree remained around 0.61 accuracy, but its overfitting decreased substantially, with the gap falling from 0.181 to 0.085. Overall, logistic regression performed better than the decision tree this week, with higher validation accuracy (0.672 vs. 0.610) and a much smaller train-validation gap (0.003 vs. 0.085).

### Week 6

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic Regression | 0.672 ± 0.013 (+0.003) | 0.674 ± 0.016 (+0.002) | 0.675 | `C=0.315` |
| Decision Tree | 0.610 ± 0.018 (+0.085) | 0.669 ± 0.015 (+0.013) | 0.671 | `max_depth=8`, `min_samples_leaf=149`, `criterion=gini` |
| Random Forest | 0.650 ± 0.019 (+0.082) | 0.679 ± 0.013 (+0.013) | 0.679 | `max_depth=6`, `min_samples_leaf=1`, `max_features=sqrt` |
| Oblique Random Forest | 0.647 ± 0.011 (+0.114) | **0.681 ± 0.012 (+0.005)** | 0.681 | `max_depth=12`, `min_samples_leaf=149`, `n_components=8` |
| AdaBoost | 0.676 ± 0.012 (+0.006) | 0.680 ± 0.013 (+0.007) | 0.682 | `max_depth=3`, `n_estimators=349`, `learning_rate=0.291` |

  
This week, I compared five models using 5-fold cross-validation and hyperparameter tuning. Oblique Random Forest achieved the highest accuracy (0.681 ± 0.012), closely followed by AdaBoost (0.680 ± 0.013) and Random Forest (0.679 ± 0.013). However, these differences are smaller than the standard deviations, so there is no clear winner.

I would choose AdaBoost because it performs almost as well as the best model and has a small train-validation gap. I would report its nested CV accuracy of 0.680 ± 0.013, as this is more reliable than the tuning score.

Tuning improved the tree-based models the most, especially Decision Tree (from 0.610 to 0.669). The chosen hyperparameters helped control overfitting by limiting tree complexity. Overall, more complex models did not necessarily perform better.


### Week 7

| Model | Default CV mean ± std (gap) | Tuned nested CV mean ± std (gap) | Tuning score | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic Regression | 0.719 ± 0.014 (+0.002) | 0.719 ± 0.013 (+0.001) | 0.719 | `C=0.315` |
| Gaussian Naive Bayes | 0.685 ± 0.011 (+0.002) | 0.701 ± 0.011 (+0.001) | 0.701 | `var_smoothing=0.029` |
| Decision Tree | 0.604 ± 0.015 (+0.110) | 0.714 ± 0.013 (+0.016) | 0.716 | `max_depth=8`, `min_samples_leaf=149`, `criterion=gini` |
| Random Forest | 0.683 ± 0.018 (+0.121) | 0.726 ± 0.015 (+0.020) | 0.726 | `max_depth=6`, `min_samples_leaf=1`, `max_features=sqrt` |
| **Oblique Random Forest** | 0.685 ± 0.016 (+0.150) | **0.728 ± 0.012 (+0.011)** | 0.728 | `max_depth=12`, `min_samples_leaf=149`, `n_components=8` |
| AdaBoost | 0.723 ± 0.016 (+0.005) | 0.727 ± 0.014 (+0.011) | 0.727 | `max_depth=3`, `n_estimators=349`, `learning_rate=0.291` |
| AdaBoost + Logistic Regression | 0.696 ± 0.012 (+0.002) | 0.713 ± 0.013 (+0.000) | 0.713 | `n_estimators=61`, `learning_rate=0.300` |
| Gradient Boosting | 0.727 ± 0.015 (+0.022) | 0.722 ± 0.015 (+0.027) | 0.722 | `n_estimators=39`, `learning_rate=0.013`, `max_depth=6`, `min_samples_leaf=18`, `subsample=0.854` |
| HistGradientBoosting | 0.717 ± 0.020 (+0.058) | 0.721 ± 0.013 (+0.021) | 0.721 | `max_iter=39`, `learning_rate=0.013`, `max_leaf_nodes=38`, `min_samples_leaf=44`, `l2_regularization=0.090` |

*All models were evaluated using 5-fold stratified cross-validation and nested CV with 3 Optuna trials. The selection metric was ROC-AUC, with target encoding and robust scaling. Engineered features were disabled and the locked test set was not evaluated.*

This week, I compared nine models using ROC-AUC instead of only accuracy. Oblique Random Forest achieved the highest nested CV score (0.728 ± 0.012), followed closely by AdaBoost (0.727 ± 0.014) and Random Forest (0.726 ± 0.015). However, the differences are smaller than the standard deviations, so there is no clear winner.

Tuning improved the tree-based models the most, especially Decision Tree, which increased from 0.604 to 0.714. The selected hyperparameters helped reduce overfitting by controlling tree complexity. Interestingly, Gradient Boosting performed slightly better without tuning (0.727 compared with 0.722), showing that tuning does not always improve results.

For now, I would choose Oblique Random Forest because it achieved the highest nested CV ROC-AUC, although AdaBoost and Random Forest are also good alternatives. I would report 0.728 ± 0.012 rather than the tuning score alone, since nested CV provides a more reliable performance estimate. These results are still preliminary because I only used 3 Optuna trials.


I also checked the false positive rates across racial groups. For Oblique Random Forest, the FPR was 0.30 for African-American defendants and 0.16 for Caucasian defendants, showing that differences between groups still exist.

This highlights why ROC-AUC alone is not enough to evaluate a model, especially in a sensitive prediction problem such as recidivism.

### Week 8

| Model | Week 7: Without FS (Nested ROC-AUC) | Week 8: With FS (Nested ROC-AUC) | Difference |
|---|---|---|---|
| Logistic Regression | 0.719 ± 0.013 | 0.716 ± 0.011 | -0.003 |
| Gaussian Naive Bayes | 0.701 ± 0.011 | 0.706 ± 0.012 | +0.005 |
| Decision Tree | 0.714 ± 0.013 | 0.714 ± 0.013 | 0.000 |
| Random Forest | 0.726 ± 0.015 | 0.721 ± 0.011 | -0.005 |
| Oblique Random Forest | 0.728 ± 0.012 | 0.724 ± 0.012 | -0.004 |
| AdaBoost | 0.727 ± 0.014 | 0.724 ± 0.014 | -0.003 |
| AdaBoost + Logistic Regression | 0.713 ± 0.013 | 0.710 ± 0.010 | -0.003 |
| Gradient Boosting | 0.722 ± 0.015 | 0.725 ± 0.012 | +0.003 |
| HistGradientBoosting | 0.721 ± 0.013 | 0.725 ± 0.013 | +0.004 |
| **XGBoost** | Not tested | **0.725 ± 0.013** | Pending |

*FS = Feature Selection. Results use 5-fold nested cross-validation, ROC-AUC and 3 Optuna trials. Engineered features were disabled and the test set remained untouched.*

This week, I tested how feature selection affects model performance. The selector reduced the number of features from 10 to just 2: `age` and `priors_count`.

The results showed that feature selection did not always improve performance. Gaussian Naive Bayes, Gradient Boosting and HistGradientBoosting improved slightly, while Random Forest, Oblique Random Forest and AdaBoost performed slightly worse. However, all these differences are smaller than the standard deviations, so I cannot conclude that feature selection significantly improves or reduces performance.

XGBoost achieved a nested CV ROC-AUC of 0.725 ± 0.013 with feature selection. This is close to the best results from previous weeks, but I still need to test XGBoost without feature selection to make a direct comparison.

The selected features were `age` and `priors_count`, suggesting that age and previous criminal history are important predictors of recidivism. I found it interesting that the models could achieve similar results using only two features instead of ten. So, feature selection can help simplify the pipeline without losing much predictive performance.


### Week 9 — Option A

This week, I adapted my pipeline to work with a new dataset, Adult Income, instead of only COMPAS. I challenge A.

I created separate configuration files for Adult Income, adapted the data cleaning process to handle missing values represented by `?`, and updated the pipeline to generate predictions for new observations.

I used XGBoost because it had already been added to the pipeline in Week 8. I also compared its performance with and without feature selection. The selected features were: `age`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`, `v01`, `marital_status_Married-civ-spouse`, `marital_status_Never-married`, `occupation_Exec-managerial`, `relationship_Wife` (VoteSelector reduced the number of transformed features from 149 to 10).

| Metric | Required | Without Feature Selection | With Feature Selection |
|---|---|---|---|
| ROC-AUC | >= 0.900 | **0.904 ± 0.016** | 0.904 ± 0.011 |
| Accuracy | >= 0.855 | **0.856 ± 0.011** | 0.853 ± 0.009 |
| F1 | >= 0.660 | **0.664 ± 0.030** | 0.655 ± 0.034 |
| Balanced Accuracy | — | 0.766 ± 0.019 | 0.761 ± 0.024 |
| Precision | — | 0.756 ± 0.033 | 0.745 ± 0.015 |
| Recall | — | 0.594 ± 0.041 | 0.586 ± 0.052 |
| Average Precision | — | 0.784 ± 0.028 | 0.784 ± 0.019 |
| Train-validation ROC-AUC gap | — | +0.065 | +0.043 |
| Features used | — | 149 | 10 |
| All three targets achieved? | — | **Yes** | No |


XGBoost without feature selection achieved all three targets: ROC-AUC of 0.904, Accuracy of 0.856 and F1 of 0.664. With feature selection, ROC-AUC remained at 0.904, but Accuracy dropped to 0.853 and F1 to 0.655, so feature selection version did not meet all the required targets.

I would therefore keep XGBoost without feature selection as my final configuration for Option A.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |

## Environment setup

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
