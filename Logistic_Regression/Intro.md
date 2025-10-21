# LOGISTIC REGRESSION - INTERVIEW GUIDE

## Table of Contents
1. [What is Logistic Regression?](#1-what-is-logistic-regression)
2. [Mathematical Foundation](#2-mathematical-foundation)
3. [Types of Logistic Regression](#3-types-of-logistic-regression)
4. [Real-World Examples](#4-real-world-worked-example)
5. [Assumptions](#5-assumptions-of-logistic-regression)
6. [Implementation (Python)](#6-implementation-python)
7. [Common Interview Questions](#7-common-interview-questions--answers)
8. [Quick Cheat Sheet](#8-quick-cheat-sheet)

---

## 1. What is Logistic Regression?

Logistic Regression is a **supervised machine learning algorithm** used for **classification problems**. Despite its name containing "regression," it's primarily used to predict categorical outcomes.

### Key Characteristics
- **Type:** Classification algorithm
- **Output:** Probability between 0 and 1
- **Decision Boundary:** Linear (separates classes with a straight line/hyperplane)
- **Use Case:** Binary and multi-class classification

### Goal
Predict the probability that an instance belongs to a particular class. For binary classification:
- P(y=1|X) - Probability of positive class
- P(y=0|X) = 1 - P(y=1|X) - Probability of negative class

### Intuition
Imagine you want to predict whether a student will pass (1) or fail (0) based on study hours. Instead of predicting the exact score (regression), you predict the **probability of passing**, then use a threshold (typically 0.5) to make the final decision.

---

## 2. Mathematical Foundation

### The Sigmoid Function (Logistic Function)

The core of logistic regression is the **sigmoid function** that maps any real number to a value between 0 and 1:

```
σ(z) = 1 / (1 + e^(-z))
```

Where:
- `σ(z)` = Output probability (between 0 and 1)
- `e` = Euler's number (≈ 2.718)
- `z` = Linear combination of features

### Model Equation

**For Binary Logistic Regression:**

```
z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

Or in matrix form:
```
z = β^T X
```

**Probability Prediction:**
```
P(y=1|X) = σ(z) = 1 / (1 + e^(-z))
```

**Final Classification:**
```
ŷ = 1  if P(y=1|X) ≥ 0.5
ŷ = 0  if P(y=1|X) < 0.5
```

### Log-Odds (Logit)

The logit function is the inverse of sigmoid:

```
log(p/(1-p)) = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

Where:
- `p/(1-p)` = Odds ratio
- `log(p/(1-p))` = Log-odds (logit)

### Cost Function (Log Loss / Binary Cross-Entropy)

Unlike linear regression (which uses MSE), logistic regression uses **log loss**:

```
Cost(h(x), y) = -y·log(h(x)) - (1-y)·log(1-h(x))
```

For all training examples:
```
J(β) = -(1/m) Σ[y⁽ⁱ⁾·log(h(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)·log(1-h(x⁽ⁱ⁾))]
```

Where:
- `m` = Number of training examples
- `h(x)` = Predicted probability
- `y` = Actual label (0 or 1)

### Gradient Descent Update Rule

```
β := β - α · ∂J(β)/∂β
```

The gradient is:
```
∂J(β)/∂β = (1/m) X^T (h(X) - y)
```

Where `α` is the learning rate.

---

## 3. Types of Logistic Regression

### A. Binary Logistic Regression

**Definition:** Classifies data into one of **two classes** (0 or 1, Yes or No, True or False)

**Output:** Single probability value P(y=1|X)

#### Example 1: Email Spam Detection
- **Features:** Email text, sender, subject line, attachments
- **Target:** Spam (1) or Not Spam (0)
- **Use Case:** Filter unwanted emails

#### Example 2: Disease Diagnosis
- **Features:** Age, blood pressure, cholesterol, glucose level
- **Target:** Disease Present (1) or Absent (0)
- **Use Case:** Medical screening

#### Example 3: Customer Churn Prediction
- **Features:** Usage frequency, customer support calls, account age, plan type
- **Target:** Will Churn (1) or Stay (0)
- **Use Case:** Retention strategies

---

### B. Multinomial Logistic Regression

**Definition:** Classifies data into **three or more unordered classes**

**Output:** Probability distribution across all classes

**Method:** Uses softmax function instead of sigmoid

#### Example 1: Handwritten Digit Recognition
- **Features:** Pixel values of digit image
- **Target:** Digits 0-9 (10 classes)
- **Use Case:** OCR systems, check processing

#### Example 2: News Article Categorization
- **Features:** Article text, keywords, author
- **Target:** Politics, Sports, Technology, Entertainment, Business
- **Use Case:** Content recommendation systems

#### Example 3: Customer Segmentation
- **Features:** Demographics, purchase history, engagement
- **Target:** Premium, Regular, Occasional, At-Risk
- **Use Case:** Targeted marketing campaigns

---

### C. Ordinal Logistic Regression

**Definition:** Classifies data into **three or more ordered classes** where order matters

**Output:** Probability for each ordered category

#### Example 1: Movie Ratings
- **Features:** Genre, director, cast, budget
- **Target:** 1 star, 2 stars, 3 stars, 4 stars, 5 stars
- **Use Case:** Recommendation systems

#### Example 2: Education Level Prediction
- **Features:** Income, location, parents' education
- **Target:** High School, Bachelor's, Master's, PhD
- **Use Case:** Educational planning

#### Example 3: Customer Satisfaction Survey
- **Features:** Service quality metrics, response time
- **Target:** Very Dissatisfied, Dissatisfied, Neutral, Satisfied, Very Satisfied
- **Use Case:** Service improvement

---

## 4. Real-World Worked Example

### Problem: Predict if a student will pass an exam based on study hours

#### Data

| Student | Study Hours (X) | Pass (y) |
|---------|----------------|----------|
| 1       | 0.5            | 0        |
| 2       | 1.0            | 0        |
| 3       | 1.5            | 0        |
| 4       | 2.0            | 0        |
| 5       | 2.5            | 1        |
| 6       | 3.0            | 1        |
| 7       | 3.5            | 1        |
| 8       | 4.0            | 1        |

#### Step 1: Initialize Parameters
```
β₀ = 0 (intercept)
β₁ = 0 (coefficient for study hours)
```

#### Step 2: Compute Linear Combination
For student 1 (X = 0.5):
```
z = β₀ + β₁·X = 0 + 0·(0.5) = 0
```

#### Step 3: Apply Sigmoid Function
```
P(pass=1|X=0.5) = σ(z) = 1/(1+e⁻⁰) = 1/(1+1) = 0.5
```

#### Step 4: After Training (Example Results)
Suppose after gradient descent, we get:
```
β₀ = -4.0
β₁ = 1.5
```

#### Step 5: Make Predictions

**For X = 1 hour:**
```
z = -4.0 + 1.5(1) = -2.5
P(pass=1) = 1/(1+e^2.5) = 1/(1+12.18) = 0.076 ≈ 7.6%
Prediction: FAIL (< 0.5)
```

**For X = 3 hours:**
```
z = -4.0 + 1.5(3) = 0.5
P(pass=1) = 1/(1+e^(-0.5)) = 1/(1+0.606) = 0.622 ≈ 62.2%
Prediction: PASS (≥ 0.5)
```

**For X = 5 hours:**
```
z = -4.0 + 1.5(5) = 3.5
P(pass=1) = 1/(1+e^(-3.5)) = 1/(1+0.030) = 0.970 ≈ 97%
Prediction: PASS (≥ 0.5)
```

### Interpretation

- **β₀ = -4.0:** When study hours = 0, log-odds of passing = -4.0 (very unlikely to pass)
- **β₁ = 1.5:** For each additional study hour, log-odds increase by 1.5
- **Decision boundary:** When P = 0.5, z = 0 → X = 4.0/1.5 ≈ 2.67 hours

Students who study **less than 2.67 hours** are predicted to fail, and those who study **more than 2.67 hours** are predicted to pass.

---

## 5. Assumptions of Logistic Regression

### 1. Binary or Ordinal Outcome
- Dependent variable must be binary (0/1) or ordinal
- Cannot use for continuous outcomes

### 2. Independence of Observations
- Each observation should be independent
- No repeated measurements from same subject (unless using specialized methods)

### 3. No Multicollinearity
- Independent variables should not be highly correlated
- **Check:** VIF < 10, correlation matrix

### 4. Linear Relationship (Log-Odds)
- Linear relationship between independent variables and **log-odds**
- Not between X and y directly
- **Check:** Box-Tidwell test

### 5. Large Sample Size
- Needs sufficient data for reliable estimates
- **Rule of thumb:** At least 10-15 observations per predictor
- More data needed if classes are imbalanced

### 6. No Outliers
- Extreme values can influence the model significantly
- **Check:** Standardized residuals, Cook's distance

---

## 6. Implementation (Python)

### Method 1: Using scikit-learn (Recommended)

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample data: Hours studied vs Pass/Fail
X = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)  # Get probabilities

# Evaluate
print(f"Intercept (β₀): {model.intercept_[0]:.4f}")
print(f"Coefficient (β₁): {model.coef_[0][0]:.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# Predict for new values
new_hours = np.array([[2.0], [3.0], [4.0]])
predictions = model.predict(new_hours)
probabilities = model.predict_proba(new_hours)[:, 1]

for hours, pred, prob in zip(new_hours, predictions, probabilities):
    print(f"Study Hours: {hours[0]}, Prediction: {'Pass' if pred == 1 else 'Fail'}, Probability: {prob:.2%}")
```

---

### Method 2: From Scratch (Using Gradient Descent)

```python
import numpy as np

class LogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """Train the model using gradient descent"""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for epoch in range(self.epochs):
            # Linear combination
            z = np.dot(X, self.weights) + self.bias
            
            # Apply sigmoid
            predictions = self.sigmoid(z)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Optional: Print loss every 100 epochs
            if epoch % 100 == 0:
                loss = self.compute_loss(predictions, y)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def compute_loss(self, predictions, y):
        """Compute binary cross-entropy loss"""
        epsilon = 1e-15  # To avoid log(0)
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        return loss
    
    def predict_proba(self, X):
        """Predict probabilities"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Predict class labels"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

# Usage
X = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

model = LogisticRegressionFromScratch(learning_rate=0.1, epochs=1000)
model.fit(X, y)

print(f"\nTrained weights: {model.weights}")
print(f"Trained bias: {model.bias}")

# Make predictions
test_X = np.array([[2.0], [3.0], [4.0]])
predictions = model.predict(test_X)
probabilities = model.predict_proba(test_X)

for x, pred, prob in zip(test_X, predictions, probabilities):
    print(f"X={x[0]}, Prediction={pred}, Probability={prob:.2%}")
```

---

### Method 3: Multinomial Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load iris dataset (3 classes)
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train multinomial logistic regression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# Show probabilities for first 5 predictions
print("\nProbabilities for each class:")
for i in range(5):
    print(f"Sample {i}: {y_pred_proba[i]}")
    print(f"Predicted class: {y_pred[i]}, Actual class: {y_test[i]}\n")
```

---

## 7. Common Interview Questions & Answers

### Q1: What is the difference between Linear Regression and Logistic Regression?

**A:** 

| Aspect | Linear Regression | Logistic Regression |
|--------|-------------------|---------------------|
| **Purpose** | Predicts continuous values | Predicts probabilities/classes |
| **Output** | Any real number | 0 to 1 (probability) |
| **Function** | Linear function | Sigmoid function |
| **Loss** | Mean Squared Error (MSE) | Log Loss (Cross-Entropy) |
| **Use Case** | Price prediction, forecasting | Classification, yes/no decisions |
| **Example** | Predict house price ($200K) | Predict spam/not spam |

---

### Q2: Why do we use the sigmoid function in Logistic Regression?

**A:** The sigmoid function has several important properties:

1. **Bounded Output:** Maps any real number to [0, 1], perfect for probabilities
2. **Smooth and Differentiable:** Allows gradient descent optimization
3. **S-shaped Curve:** Captures the transition from 0 to 1
4. **Interpretable:** Output can be interpreted as probability

**Formula:** σ(z) = 1/(1+e^(-z))

**Properties:**
- When z → ∞, σ(z) → 1
- When z → -∞, σ(z) → 0
- When z = 0, σ(z) = 0.5

---

### Q3: What is the decision boundary in Logistic Regression?

**A:** The decision boundary is the line (or hyperplane in higher dimensions) that separates different classes.

**For Binary Classification:**
- Decision boundary occurs when P(y=1) = 0.5
- This happens when z = 0
- Therefore: β₀ + β₁x₁ + β₂x₂ + ... = 0

**Example:** If β₀ = -4, β₁ = 2
- Decision boundary: -4 + 2x = 0 → x = 2
- If x < 2: Predict class 0
- If x > 2: Predict class 1

**Note:** The decision boundary is **linear**, but the sigmoid function creates a non-linear probability curve.

---

### Q4: What is Log Loss (Binary Cross-Entropy)?

**A:** Log Loss measures the performance of a classification model where predictions are probabilities.

**Formula:**
```
Loss = -(1/m) Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

**Interpretation:**
- **Perfect prediction (ŷ=y):** Loss = 0
- **Wrong with high confidence:** Loss → ∞
- **Lower loss = Better model**

**Why not MSE?**
- MSE is non-convex for logistic regression → hard to optimize
- Log Loss is convex → guaranteed to find global minimum

---

### Q5: How do you handle imbalanced datasets in Logistic Regression?

**A:** Several techniques:

1. **Class Weights**
```python
model = LogisticRegression(class_weight='balanced')
# OR
model = LogisticRegression(class_weight={0: 1, 1: 10})
```

2. **Resampling**
   - **Oversampling:** Duplicate minority class (SMOTE)
   - **Undersampling:** Reduce majority class

3. **Threshold Adjustment**
   - Change decision threshold from 0.5 to optimize F1-score
   ```python
   y_pred = (model.predict_proba(X)[:, 1] >= 0.3).astype(int)
   ```

4. **Use Different Metrics**
   - Don't rely on accuracy alone
   - Use: Precision, Recall, F1-Score, AUC-ROC

5. **Ensemble Methods**
   - Combine multiple models
   - Use algorithms like Random Forest, XGBoost

---

### Q6: What is the difference between Logistic Regression and SVM?

**A:**

| Aspect | Logistic Regression | SVM |
|--------|---------------------|-----|
| **Output** | Probability (0 to 1) | Class label directly |
| **Loss Function** | Log loss | Hinge loss |
| **Decision Boundary** | Probabilistic | Maximum margin |
| **Outliers** | Sensitive to all points | Less sensitive (focus on support vectors) |
| **Kernel Trick** | Not typically used | Can use kernels for non-linear boundaries |
| **Interpretability** | High (coefficients are interpretable) | Lower (especially with kernels) |
| **Speed** | Faster on large datasets | Slower on large datasets |

**When to use Logistic Regression:**
- Need probability estimates
- Interpretability is important
- Large datasets
- Linear decision boundary is sufficient

**When to use SVM:**
- Need maximum margin classifier
- Non-linear decision boundaries (with kernels)
- High-dimensional data

---

### Q7: How do you evaluate a Logistic Regression model?

**A:** Multiple metrics are needed:

**1. Confusion Matrix**
```
                Predicted
              0         1
Actual  0    TN        FP
        1    FN        TP
```

**2. Accuracy**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
⚠️ Not good for imbalanced datasets!

**3. Precision**
```
Precision = TP / (TP + FP)
```
"Of all positive predictions, how many were correct?"

**4. Recall (Sensitivity)**
```
Recall = TP / (TP + FN)
```
"Of all actual positives, how many did we catch?"

**5. F1-Score**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Harmonic mean of Precision and Recall

**6. AUC-ROC Curve**
- Area Under the Receiver Operating Characteristic curve
- Measures model's ability to distinguish classes
- AUC = 1: Perfect model
- AUC = 0.5: Random guessing
- AUC < 0.5: Worse than random

**7. Log Loss**
- Measures probability predictions
- Lower is better

**Example:**
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba[:, 1])
```

---

### Q8: What is regularization in Logistic Regression?

**A:** Regularization prevents overfitting by adding a penalty term to the cost function.

**L1 Regularization (Lasso):**
```
Cost = Log Loss + λ Σ|β|
```
- **Effect:** Can set coefficients to exactly zero (feature selection)
- **Use when:** Many features, want sparse model

```python
model = LogisticRegression(penalty='l1', C=1.0, solver='liblinear')
```

**L2 Regularization (Ridge):**
```
Cost = Log Loss + λ Σβ²
```
- **Effect:** Shrinks coefficients but doesn't eliminate them
- **Use when:** Multicollinearity present, all features relevant

```python
model = LogisticRegression(penalty='l2', C=1.0)  # C is inverse of λ
```

**ElasticNet (L1 + L2):**
```python
model = LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga')
```

**Note:** Parameter `C` is the inverse of regularization strength
- Small C → Strong regularization
- Large C → Weak regularization

---

### Q9: Can Logistic Regression handle non-linear decision boundaries?

**A:** By default, no. But you can create non-linear boundaries using:

**1. Polynomial Features**
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LogisticRegression()
model.fit(X_poly, y)
```

**2. Feature Engineering**
- Create interaction terms: x₁ × x₂
- Add squared terms: x₁², x₂²
- Domain-specific transformations

**3. Kernel Methods** (less common)
- Similar to SVM kernels
- Not standard in scikit-learn's LogisticRegression

**Example:**
Original features: [x₁, x₂]
Polynomial features: [1, x₁, x₂, x₁², x₁x₂, x₂²]

This allows the linear model to capture non-linear patterns!

---

### Q10: What is the maximum likelihood estimation in Logistic Regression?

**A:** Maximum Likelihood Estimation (MLE) is the principle used to find optimal parameters in Logistic Regression.

**Concept:** Find parameters (β) that maximize the likelihood of observing the training data.

**Likelihood Function:**
```
L(β) = Π P(y⁽ⁱ⁾|x⁽ⁱ⁾; β)
     = Π [h(x⁽ⁱ⁾)]^y⁽ⁱ⁾ [1-h(x⁽ⁱ⁾)]^(1-y⁽ⁱ⁾)
```

**Log-Likelihood (easier to optimize):**
```
log L(β) = Σ [y⁽ⁱ⁾ log h(x⁽ⁱ⁾) + (1-y⁽ⁱ⁾) log(1-h(x⁽ⁱ⁾))]
```

**Relationship to Loss:**
- Maximizing log-likelihood = Minimizing log loss
- Log Loss = -log L(β) / m

**Optimization:**
- No closed-form solution (unlike linear regression)
- Use iterative methods: Gradient Descent, Newton-Raphson, BFGS

---

## 8. Quick Cheat Sheet

### Key Equations

| Equation | Formula |
|----------|---------|
| Sigmoid Function | σ(z) = 1/(1+e^(-z)) |
| Linear Combination | z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ |
| Probability | P(y=1\|X) = σ(β^T X) |
| Log-Odds | log(p/(1-p)) = β^T X |
| Log Loss | J(β) = -(1/m)Σ[y·log(ŷ) + (1-y)·log(1-ŷ)] |
| Gradient | ∂J/∂β = (1/m)X^T(h(X) - y) |
| Decision Boundary | β₀ + β₁x₁ + β₂x₂ + ... = 0 |

---

### Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Precision | TP/(TP+FP) | Positive prediction accuracy |
| Recall | TP/(TP+FN) | True positive capture rate |
| F1-Score | 2×(Precision×Recall)/(Precision+Recall) | Harmonic mean |
| Specificity | TN/(TN+FP) | True negative rate |
| AUC-ROC | Area under ROC curve | Separation capability |

---

### Regularization

| Type | Penalty | Effect | Use Case |
|------|---------|--------|----------|
| L1 (Lasso) | λΣ\|β\| | Feature selection | Many irrelevant features |
| L2 (Ridge) | λΣβ² | Coefficient shrinkage | Multicollinearity |
| ElasticNet | λ₁Σ\|β\| + λ₂Σβ² | Both effects | Combination needed |

---

### Assumptions Checklist

- ✅ **Binary/Ordinal outcome** → Check target variable type
- ✅ **Independence** → No repeated measures
- ✅ **No multicollinearity** → VIF < 10
- ✅ **Linear log-odds** → Box-Tidwell test
- ✅ **Large sample size** → 10-15 obs per predictor
- ✅ **No extreme outliers** → Standardized residuals

---

### When to Use

**✅ Use Logistic Regression when:**
- Classification problem (binary or multi-class)
- Need probability estimates
- Linear decision boundary is reasonable
- Interpretability is important
- Fast training needed
- Baseline model

**❌ Don't use when:**
- Target is continuous (use Linear Regression)
- Highly non-linear relationships (try tree-based models)
- Very large feature space without regularization
- Need automatic feature interactions (try tree models)

---

### Python Quick Reference

```python
# Import
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Binary Classification
model = LogisticRegression()
model.fit(X_train, y_train)

# Multinomial Classification
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# With Regularization
model = LogisticRegression(penalty='l2', C=1.0)  # L2
model = LogisticRegression(penalty='l1', C=1.0, solver='liblinear')  # L1

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Coefficients
model.coef_        # Weights
model.intercept_   # Bias

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
```

---

## Tips for Interview Success

1. ✅ **Understand the sigmoid function** and why it's used
2. ✅ **Explain the difference** between Linear and Logistic Regression clearly
3. ✅ **Know evaluation metrics** beyond accuracy (Precision, Recall, F1, AUC)
4. ✅ **Understand log loss** and why it's used instead of MSE
5. ✅ **Be familiar with regularization** (L1 vs L2)
6. ✅ **Know how to handle imbalanced data**
7. ✅ **Understand decision boundaries** and how they're formed
8. ✅ **Practice implementation** both with scikit-learn and from scratch
9. ✅ **Know the assumptions** and how to check them
10. ✅ **Have real-world examples** ready from your domain

---

## Common Use Cases by Industry

### Healthcare
- Disease diagnosis (binary: disease present/absent)
- Patient risk assessment (high/medium/low risk)
- Treatment response prediction

### Finance
- Credit risk assessment (approve/reject loan)
- Fraud detection (fraudulent/legitimate transaction)
- Customer churn prediction

### Marketing
- Email campaign response (will respond/won't respond)
- Customer segmentation (VIP/Regular/Occasional)
- Lead scoring (hot/warm/cold)

### E-commerce
- Product recommendation (will buy/won't buy)
- Customer lifetime value prediction (high/medium/low)
- Cart abandonment prediction

### Technology
- Spam detection (spam/not spam)
- User engagement prediction (will engage/won't engage)
- A/B testing winner prediction

---

**Created:** October 21, 2025  
**Purpose:** Interview Preparation for Machine Learning Engineers  
**Repository:** ML_Algo / Logistic_Regression


