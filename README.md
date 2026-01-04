# Heart Disease Classification Using Machine Learning

Predicting heart disease presence using clinical features with Decision Trees and Random Forest models.

## Why This Project?

Cardiovascular disease is the leading cause of death globally, responsible for ~18 million deaths annually. Early detection using readily available clinical measures could enable timely intervention. This project explores which clinical features best predict heart disease and compares two tree-based classification methods.

## The Data

**Source:** UCI Machine Learning Repository - Cleveland Heart Disease Database  
**Sample:** 297 patients (after removing missing values)  
**Prevalence:** ~54% with heart disease

**Clinical Features (13 total):**
- Demographics: Age, Sex
- Symptoms: Chest pain type, exercise-induced angina
- Measurements: Resting blood pressure, cholesterol, max heart rate
- Tests: ECG results, ST depression, vessel fluoroscopy, thalassemia

The target variable indicates presence/absence of heart disease (originally 0-4 scale, simplified to binary).

## Methods

### Data Preparation
- Removed 6 observations with missing values (kept 297/303 = 98%)
- Converted multi-class outcome (0-4) to binary (disease vs. no disease)
- Created train/test split (75/25) with stratified sampling

### Models Tested

**1. Decision Tree**
- Single tree using CART algorithm
- Advantages: Highly interpretable, shows decision paths
- Used to understand which features drive predictions

**2. Random Forest**
- Ensemble of 500 trees
- Advantages: Usually more accurate, reduces overfitting
- Used for better prediction performance

## Key Findings

### Model Performance

| Model | Accuracy | Sensitivity | Specificity |
|-------|----------|-------------|-------------|
| Decision Tree | ~73% | ~85% | ~58.8% |
| Random Forest | ~87.8% | ~90% | ~85.3% |

Random Forest performed better, as expected for ensemble methods.

### Most Important Features

The top predictors of heart disease were:
1. **Thalassemia (thal)** - Blood disorder indicator
2. **Chest pain type (cp)** - Symptom presentation
3. **Number of vessels (ca)** - From fluoroscopy
4. **Maximum heart rate (thalach)**
5. **ST depression (oldpeak)** - Exercise ECG result

**Interesting:** Some commonly discussed risk factors like cholesterol and blood pressure were less predictive in this model. This might be because:
- The dataset is relatively small
- These factors may have indirect effects captured by other variables
- Or these are important for disease *development* but not *diagnosis* when other clinical signs are present

## Files

```
├── heart_disease_analysis.R       # Main analysis script
├── figures/
│   ├── age_distribution_heart.png
│   ├── cholesterol_heart.png
│   ├── gender_heart.png
│   ├── decision_tree_plot.png
│   └── rf_importance.png
└── model_comparison.csv
```

## Running the Analysis

**Required packages:**
```r
install.packages(c("tidyverse", "caret", "rpart", "rpart.plot", "randomForest"))
```

**Run:**
```r
source("heart_disease_analysis.R")
```

The script will:
1. Download data directly from UCI repository
2. Clean and prepare the dataset
3. Create visualizations
4. Train both models
5. Compare performance
6. Save all outputs

**Runtime:** ~1 minute

## What I Learned

**Technical Skills:**
- First time working with tree-based models (vs. regression in my diabetes project)
- Understanding feature importance vs. regression coefficients
- Comparing interpretability (Decision Tree) vs. accuracy (Random Forest)

**Conceptual Insights:**
- Tree models ask "yes/no" questions vs. regression's continuous relationships
- Random forests trade interpretability for better predictions
- Small datasets can still provide useful models but need external validation

**Practical Considerations:**
- Missing only 6 observations (2%) is manageable with complete case analysis
- With 297 observations and 13 features, risk of overfitting is moderate
- Would need much larger sample for clinical deployment

## Limitations

1. **Small dataset (n=297)** - Limits generalizability
2. **No external validation** - Tested on held-out data from same study, not independent population
3. **Missing data handling** - Just deleted incomplete cases rather than imputing
4. **No hyperparameter tuning** - Used default settings for both models
5. **Medical variables** - Some features (thalassemia, vessel count) require clinical expertise to fully interpret

## If I Were to Extend This

- **Get more data:** Combine multiple hospital datasets
- **Try other algorithms:** Gradient boosting (XGBoost), neural networks
- **Tune hyperparameters:** Grid search for optimal settings
- **Add cost-sensitive learning:** Penalize false negatives more (missing disease is worse than false alarm)
- **External validation:** Test on different population/hospital
- **Feature engineering:** Create interaction terms (age × max heart rate?)
- **Explainability:** Use SHAP values to explain individual predictions


## References

- **Data Source:** Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine, School of Information and Computer Sciences.
- **Original Publication:** Detrano, R. et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. American Journal of Cardiology.
