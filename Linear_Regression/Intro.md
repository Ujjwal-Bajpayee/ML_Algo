# LINEAR REGRESSION - INTERVIEW GUIDE

## Table of Contents
1. [What is Linear Regression?](#1-what-is-linear-regression)
2. [Mathematical Foundation](#2-mathematical-foundation)
3. [Types of Linear Regression](#3-types-of-linear-regression)
4. [Real-World Examples](#4-real-world-worked-example)
5. [Assumptions](#5-assumptions-of-linear-regression)
6. [Implementation (Python)](#6-implementation-python)
7. [Common Interview Questions](#7-common-interview-questions--answers)
8. [Quick Cheat Sheet](#8-quick-cheat-sheet)

---

## 1. What is Linear Regression?

Linear Regression is a **supervised machine learning algorithm** that models the relationship between:
- **Independent variable(s) [X]** - Features/Predictors
- **Dependent variable [y]** - Target/Response

### Goal
Find the best-fit line (or hyperplane) that minimizes the error between predicted and actual values.

### Intuition
Imagine plotting points on a graph and drawing a straight line that passes as close as possible to all points. That's linear regression!

---

## 2. Mathematical Foundation

### Equation (Simple Linear Regression)
```
ŷ = β₀ + β₁x
```

Where:
- `ŷ` = Predicted value
- `β₀` = Intercept (bias term)
- `β₁` = Slope (coefficient)
- `x` = Input feature

### Equation (Multiple Linear Regression)
```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

Or in matrix form:
```
ŷ = Xβ
```

Where:
- `X` = Feature matrix (n_samples × n_features)
- `β` = Coefficient vector
- `ŷ` = Predictions

### Cost Function (Mean Squared Error)
```
MSE = (1/n) Σ(yᵢ - ŷᵢ)²
```

**Goal:** Minimize MSE to find optimal β₀ and β₁

### Closed-Form Solution (Normal Equation)
```
β = (XᵀX)⁻¹Xᵀy
```

### Gradient Descent Update Rule
```
β := β - α · ∂(Cost)/∂β
```

Where `α` is the learning rate

---

## 3. Types of Linear Regression

### A. Simple Linear Regression

**Definition:** One independent variable predicts one dependent variable

**Equation:** `y = β₀ + β₁x + ε`

#### Example 1: Predicting House Price based on Size
- X = House size (sq ft)
- y = House price ($)
- **Model:** `Price = 50,000 + 100 × Size`

#### Example 2: Predicting Salary based on Years of Experience
- X = Years of experience
- y = Salary
- **Model:** `Salary = 30,000 + 5,000 × Experience`

---

### B. Multiple Linear Regression

**Definition:** Multiple independent variables predict one dependent variable

**Equation:** `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε`

#### Example 1: Predicting House Price
- X₁ = Size (sq ft)
- X₂ = Number of bedrooms
- X₃ = Age of house
- X₄ = Distance from city center
- **Model:** `Price = 50,000 + 100×Size + 15,000×Bedrooms - 2,000×Age - 500×Distance`

#### Example 2: Predicting Student Performance
- X₁ = Study hours
- X₂ = Previous test score
- X₃ = Attendance %
- **Model:** `Score = 10 + 5×StudyHours + 0.4×PrevScore + 0.3×Attendance`

---

### C. Polynomial Regression (Extension)

**Definition:** Models non-linear relationships using polynomial features

**Equation:** `y = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ`

#### Example: Predicting crop yield based on fertilizer amount
- Too little fertilizer → low yield
- Optimal amount → maximum yield
- Too much fertilizer → yield decreases (parabolic relationship)

---

## 4. Real-World Worked Example

### Problem: Predict salary based on years of experience

#### Data
| Experience (X) | Salary (y) |
|----------------|------------|
| 1              | 40,000     |
| 2              | 45,000     |
| 3              | 50,000     |
| 4              | 55,000     |
| 5              | 60,000     |

#### Step 1: Calculate means
```
X̄ = (1+2+3+4+5)/5 = 3
ȳ = (40+45+50+55+60)/5 = 50 (in thousands)
```

#### Step 2: Calculate slope (β₁)
```
β₁ = Σ[(xᵢ - X̄)(yᵢ - ȳ)] / Σ[(xᵢ - X̄)²]
   = [(-2)(-10) + (-1)(-5) + (0)(0) + (1)(5) + (2)(10)] / [4 + 1 + 0 + 1 + 4]
   = [20 + 5 + 0 + 5 + 20] / 10
   = 50 / 10
   = 5
```

#### Step 3: Calculate intercept (β₀)
```
β₀ = ȳ - β₁X̄
   = 50 - 5(3)
   = 35
```

### Final Model
```
Salary = 35,000 + 5,000 × Experience
```

### Interpretation
- Base salary (0 years experience): **$35,000**
- For each additional year of experience, salary increases by **$5,000**

### Prediction
For 6 years of experience:
```
Salary = 35,000 + 5,000(6) = $65,000
```

---

## 5. Assumptions of Linear Regression

### 1. Linearity
- Relationship between X and y is linear
- **Check:** Residual plots should show random scatter

### 2. Independence
- Observations are independent of each other
- **Check:** No autocorrelation (Durbin-Watson test)

### 3. Homoscedasticity
- Constant variance of residuals
- **Check:** Residuals vs fitted values plot (no funnel shape)

### 4. Normality
- Residuals are normally distributed
- **Check:** Q-Q plot, Shapiro-Wilk test

### 5. No Multicollinearity (for Multiple Regression)
- Independent variables are not highly correlated
- **Check:** VIF (Variance Inflation Factor) < 10

---

## 6. Implementation (Python)

### Method 1: Using scikit-learn (Recommended)

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Experience (years)
y = np.array([40, 45, 50, 55, 60])            # Salary (in thousands)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print(f"Intercept (β₀): {model.intercept_}")
print(f"Coefficient (β₁): {model.coef_[0]}")
print(f"R² Score: {r2_score(y_test, y_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")

# Predict new value
new_experience = np.array([[6]])
predicted_salary = model.predict(new_experience)
print(f"Predicted salary for 6 years: ${predicted_salary[0]*1000}")
```

---

### Method 2: From Scratch (Using Normal Equation)

```python
import numpy as np

class LinearRegressionFromScratch:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
    
    def fit(self, X, y):
        # Add bias term (column of 1s)
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Normal equation: β = (X^T X)^(-1) X^T y
        theta = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
        
        self.intercept = theta[0]
        self.coefficients = theta[1:]
    
    def predict(self, X):
        return self.intercept + X @ self.coefficients

# Usage
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([40, 45, 50, 55, 60])

model = LinearRegressionFromScratch()
model.fit(X, y)

print(f"Intercept: {model.intercept}")
print(f"Coefficients: {model.coefficients}")
print(f"Prediction for X=6: {model.predict(np.array([[6]]))}")
```

---

### Method 3: Using Gradient Descent

```python
import numpy as np

def gradient_descent_linear_regression(X, y, learning_rate=0.01, epochs=1000):
    m = len(y)
    X_bias = np.c_[np.ones((m, 1)), X]  # Add bias term
    theta = np.zeros(X_bias.shape[1])   # Initialize parameters
    
    for epoch in range(epochs):
        # Predictions
        predictions = X_bias @ theta
        
        # Compute gradients
        errors = predictions - y
        gradients = (1/m) * X_bias.T @ errors
        
        # Update parameters
        theta -= learning_rate * gradients
        
        # Optional: Print cost every 100 epochs
        if epoch % 100 == 0:
            cost = (1/(2*m)) * np.sum(errors**2)
            print(f"Epoch {epoch}, Cost: {cost:.4f}")
    
    return theta

# Usage
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([40, 45, 50, 55, 60])

theta = gradient_descent_linear_regression(X, y, learning_rate=0.01, epochs=1000)
print(f"\nIntercept (β₀): {theta[0]}")
print(f"Coefficient (β₁): {theta[1]}")
```

---

## 7. Common Interview Questions & Answers

### Q1: What is the difference between Linear Regression and Logistic Regression?

**A:** Linear Regression predicts continuous values (e.g., house prices), while Logistic Regression predicts probabilities for classification (0 or 1). Linear uses MSE as loss; Logistic uses cross-entropy loss.

---

### Q2: How do you handle outliers in Linear Regression?

**A:** Methods include:
- Remove outliers (if data errors)
- Transform features (log transformation)
- Use robust regression techniques (Huber regression, RANSAC)
- Regularization (Ridge, Lasso)

---

### Q3: What is R² (R-squared) and how do you interpret it?

**A:** R² measures the proportion of variance in y explained by X.

- **Range:** 0 to 1 (can be negative for bad models)
- **Example:** R² = 0.8 means 80% of variance is explained by the model
- **Higher R² = better fit**
- **Formula:** `R² = 1 - (SS_res / SS_tot)`

---

### Q4: What is the difference between Ridge and Lasso regression?

**A:** Both add regularization to prevent overfitting:

**Ridge (L2):** Adds penalty = `λ Σβ²`
- Shrinks coefficients but doesn't set them to zero
- Use when all features are relevant

**Lasso (L1):** Adds penalty = `λ Σ|β|`
- Can set coefficients to exactly zero (feature selection)
- Use for sparse models with many irrelevant features

**ElasticNet:** Combines both L1 and L2 penalties

---

### Q5: When should you NOT use Linear Regression?

**A:** Avoid when:
- Relationship is highly non-linear (use polynomial/tree-based models)
- Target variable is categorical (use classification algorithms)
- Assumptions are severely violated
- Severe multicollinearity exists
- Small dataset with many features (risk of overfitting)

---

### Q6: How do you choose between Gradient Descent and Normal Equation?

**A:** 

**Normal Equation:**
- ✅ Pro: No need to choose learning rate, gives exact solution
- ❌ Con: Slow for large features (O(n³) complexity), needs invertible XᵀX
- 📌 Use when: n < 10,000 features

**Gradient Descent:**
- ✅ Pro: Works well with large datasets, scales better
- ❌ Con: Need to tune learning rate, requires more iterations
- 📌 Use when: n > 10,000 features or online learning

---

### Q7: What is multicollinearity and how does it affect your model?

**A:** Multicollinearity occurs when independent variables are highly correlated.

**Effects:**
- Unstable coefficient estimates
- High variance in predictions
- Difficult to interpret individual feature importance

**Detection:**
- Correlation matrix (|r| > 0.8)
- VIF > 10

**Solutions:**
- Remove correlated features
- Use PCA
- Ridge regression

---

### Q8: How do you evaluate a Linear Regression model?

**A:** 

**Metrics:**
- **R² (R-squared):** Proportion of variance explained (0 to 1)
- **Adjusted R²:** R² adjusted for number of predictors
- **MSE (Mean Squared Error):** Average squared error
- **RMSE (Root Mean Squared Error):** Square root of MSE (same unit as y)
- **MAE (Mean Absolute Error):** Average absolute error

**Visual checks:**
- Residual plots (should be random)
- Q-Q plot (check normality)
- Actual vs Predicted plot

---

### Q9: What is the bias-variance tradeoff in Linear Regression?

**A:** 

**Bias:** Error from incorrect assumptions (underfitting)
- Simple models → high bias → systematic error

**Variance:** Error from sensitivity to training data (overfitting)
- Complex models → high variance → unstable predictions

**Goal:** Find balance
- Use regularization (Ridge/Lasso) to reduce variance
- Add features or polynomial terms to reduce bias

---

### Q10: How do you handle categorical variables in Linear Regression?

**A:** Use encoding techniques:

**One-Hot Encoding (for nominal variables):**
- Color = {Red, Blue, Green}
- Create: Color_Red, Color_Blue, Color_Green (0 or 1)
- Drop one column to avoid multicollinearity (dummy variable trap)

**Label Encoding (for ordinal variables):**
- Education = {High School, Bachelor, Master, PhD}
- Encode as: {0, 1, 2, 3}

---

## 8. Quick Cheat Sheet

### Key Equations

| Equation | Formula |
|----------|---------|
| Simple Linear Regression | `ŷ = β₀ + β₁x` |
| Multiple Linear Regression | `ŷ = β₀ + Σ(βᵢxᵢ)` |
| Slope (β₁) | `β₁ = Σ[(xᵢ-x̄)(yᵢ-ȳ)] / Σ(xᵢ-x̄)²` |
| Intercept (β₀) | `β₀ = ȳ - β₁x̄` |
| Normal Equation | `β = (XᵀX)⁻¹Xᵀy` |
| Cost Function (MSE) | `J(β) = (1/n) Σ(yᵢ - ŷᵢ)²` |
| R-squared | `R² = 1 - (SS_res / SS_tot)` |
| Adjusted R² | `R²_adj = 1 - [(1-R²)(n-1)/(n-k-1)]` |

---

### Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| MSE | `(1/n) Σ(yᵢ - ŷᵢ)²` | Lower is better, penalizes large errors |
| RMSE | `√MSE` | Same unit as target variable |
| MAE | `(1/n) Σ\|yᵢ - ŷᵢ\|` | Less sensitive to outliers |
| R² | `1 - SS_res/SS_tot` | 0 to 1, higher is better |

---

### Regularization

| Type | Formula | Use Case |
|------|---------|----------|
| Ridge (L2) | `Cost = MSE + λ Σβᵢ²` | Shrinks coefficients |
| Lasso (L1) | `Cost = MSE + λ Σ\|βᵢ\|` | Feature selection |
| ElasticNet | `Cost = MSE + λ₁Σ\|βᵢ\| + λ₂Σβᵢ²` | Combination |

---

### Assumptions Checklist

- ✅ **Linearity** → Residual plot
- ✅ **Independence** → Durbin-Watson test
- ✅ **Homoscedasticity** → Scale-location plot
- ✅ **Normality of residuals** → Q-Q plot
- ✅ **No multicollinearity** → VIF < 10

---

### When to Use

**✅ Use Linear Regression when:**
- Predicting continuous values
- Linear relationships
- Interpretable models needed
- Baseline model

**❌ Don't use when:**
- Classification problems
- Highly non-linear relationships
- Complex interactions
- Large number of features (without regularization)

---

### Python Quick Reference

```python
# Train model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Get parameters
model.coef_        # Coefficients
model.intercept_   # Intercept

# Predict
y_pred = model.predict(X_test)

# Evaluate
from sklearn.metrics import r2_score, mean_squared_error
r2_score(y_test, y_pred)
mean_squared_error(y_test, y_pred)
```

---

## Tips for Interview Success

1. ✅ Always mention assumptions when discussing Linear Regression
2. ✅ Be ready to explain the difference between Simple and Multiple LR
3. ✅ Understand when to use Normal Equation vs Gradient Descent
4. ✅ Know how to handle categorical variables and outliers
5. ✅ Be familiar with regularization techniques (Ridge, Lasso)
6. ✅ Explain evaluation metrics clearly (R², RMSE, MAE)
7. ✅ Practice coding Linear Regression from scratch
8. ✅ Understand bias-variance tradeoff
9. ✅ Know the limitations of Linear Regression
10. ✅ Be ready with real-world examples from your domain


