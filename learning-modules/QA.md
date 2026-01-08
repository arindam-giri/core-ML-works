## Question 1: Feature Engineering & Selection
**"You have a dataset with 1000 features but only 500 samples. How would you approach feature selection, and what problems might arise?"**

**Expected Answer:**
- **Curse of dimensionality**: More features than samples can lead to overfitting
- **Feature selection techniques**:
  - Filter methods: Correlation analysis, chi-square test, mutual information
  - Wrapper methods: Recursive Feature Elimination (RFE)
  - Embedded methods: LASSO regularization, tree-based feature importance
- **Dimensionality reduction**: PCA, t-SNE, or autoencoders
- **Cross-validation strategy**: Use stratified k-fold to ensure robust evaluation
- **Domain knowledge**: Leverage business understanding to prioritize relevant features

## Question 2: Model Evaluation & Bias
**"Your binary classification model shows 95% accuracy, but stakeholders are concerned about model bias. How would you investigate and address this?"**

**Expected Answer:**
- **Look beyond accuracy**: Check precision, recall, F1-score, AUC-ROC
- **Class imbalance**: 95% accuracy might mean predicting majority class always
- **Confusion matrix analysis**: Examine false positives/negatives
- **Bias investigation**:
  - Demographic parity across different groups
  - Equalized odds and opportunity
  - Fairness metrics (statistical parity, individual fairness)
- **Solutions**:
  - Resampling techniques (SMOTE, undersampling)
  - Cost-sensitive learning
  - Threshold optimization per group
  - Bias-aware algorithms

## Question 3: Real-world ML Pipeline Challenge
**"You've deployed a recommendation model that performed well in testing, but click-through rates are dropping in production. What could be wrong and how would you debug?"**

**Expected Answer:**
- **Data drift**: Input feature distributions may have changed
- **Concept drift**: User behavior patterns evolved
- **Training-serving skew**: Differences in data preprocessing pipelines
- **Debugging approach**:
  - Monitor feature distributions over time
  - A/B test against baseline model
  - Analyze model predictions vs. actual outcomes
  - Check for seasonal patterns or external factors
- **Solutions**:
  - Implement model retraining pipeline
  - Online learning approaches
  - Ensemble methods with recent data
  - Feature store for consistent preprocessing

## Question 4: Algorithm Selection & Trade-offs
**"Compare Random Forest and Gradient Boosting for a tabular dataset. When would you choose one over the other?"**

**Expected Answer:**
- **Random Forest**:
  - Pros: Faster training, less prone to overfitting, handles missing values well, parallel training
  - Cons: Can plateau in performance, less interpretable individual trees
  - Use when: Need quick baseline, have noisy data, want stability

- **Gradient Boosting (XGBoost/LightGBM)**:
  - Pros: Often higher accuracy, better handling of imbalanced data, built-in regularization
  - Cons: Longer training time, more hyperparameter tuning needed, prone to overfitting
  - Use when: Need maximum performance, have clean data, can invest in tuning

- **Considerations**: Dataset size, interpretability requirements, training time constraints, maintenance overhead

## Question 5: Deep Learning & Optimization
**"Explain the vanishing gradient problem and how modern architectures address it."**

**Expected Answer:**
- **Vanishing gradient problem**: 
  - Gradients become exponentially smaller in earlier layers during backpropagation
  - Caused by repeated multiplication of small derivatives (especially with sigmoid/tanh)
  - Results in slow/no learning in early layers

- **Solutions**:
  - **Activation functions**: ReLU family (ReLU, Leaky ReLU, ELU) instead of sigmoid/tanh
  - **Weight initialization**: Xavier/He initialization to maintain gradient scale
  - **Architecture innovations**:
    - ResNet: Skip connections allow gradients to flow directly
    - LSTM/GRU: Gating mechanisms control information flow
    - Batch Normalization: Normalizes inputs to each layer
  - **Optimization**: Adam optimizer adapts learning rates per parameter
  - **Gradient clipping**: Prevents exploding gradients

**Follow-up questions to gauge depth:**
- "How would you detect overfitting during training?"
- "Explain the bias-variance tradeoff with a practical example"
- "How do you handle missing data in different scenarios?"

These questions test both theoretical knowledge and practical problem-solving skills expected from a 4-year experienced data scientist.
