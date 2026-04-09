
# K-Nearest Neighbors (KNN) Algorithm: From Basics to Advanced

## 1. Introduction
K-Nearest Neighbors (KNN) is one of the simplest and most intuitive **machine learning algorithms**. It is a **supervised learning algorithm** used for both **classification** and **regression** tasks.

KNN is called a **lazy learning algorithm** because it does not learn a parametric model during training. Instead, it memorizes the training data and performs computation only at prediction time.

---

## 2. Intuition Behind KNN
The core idea of KNN is:

> **"Similar data points exist close to each other."**

When a new data point appears:
1. Find the **K closest data points** from the training dataset.
2. Look at their labels.
3. Decide the output based on majority vote (classification) or average value (regression).

**Real-life analogy:**
If you move to a new neighborhood and want to guess your neighbors' profession, you would look at the professions of people who live close to you.

---

## 3. Mathematical Foundation

### Distance Metrics
To find the "nearest" neighbors, KNN uses a distance function:

- **Euclidean Distance (most common):**  
  d(p, q) = √[(x₁−x₂)² + (y₁−y₂)²]

- **Manhattan Distance:**  
  d(p, q) = |x₁−x₂| + |y₁−y₂|

- **Minkowski Distance** (generalized form)

Choice of distance metric significantly affects model performance.

---

## 4. Algorithm Steps (Basic Level)

### Training Phase
1. Store the entire training dataset.

### Prediction Phase
1. Choose value of **K**.
2. Compute distance between test point and all training points.
3. Select **K nearest neighbors**.
4. Aggregate their outputs:
   - Classification → Majority vote
   - Regression → Mean of values

---

## 5. How the Data Model Gets Trained

### Important Concept: **No Explicit Training**

Unlike algorithms such as Linear Regression or Neural Networks:

- KNN **does NOT learn weights or parameters**
- No cost function optimization
- No gradient descent

### Training Means:

✅ Data is validated  
✅ Features are scaled/normalized  
✅ Dataset is stored efficiently

So, "training" simply means **saving the dataset in memory**.

---

## 6. Final Output of the Model After Training

After training, the KNN model contains:

- Training feature matrix (X)
- Training labels (y)
- Chosen distance metric
- Value of K

📌 **There are NO learned coefficients or equations.**

**Final trained model = Entire training dataset**

---

## 7. KNN for Classification

### Output
- A **class label**

### Example
If K=5 and nearest labels are:

["Yes", "Yes", "No", "Yes", "No"]

✅ Output → **Yes** (majority)

---

## 8. KNN for Regression

### Output
- A **continuous numeric value**

### Example
If K=4 and the target values are:

[200, 220, 210, 230]

✅ Output → **(200+220+210+230)/4 = 215**

---

## 9. Feature Scaling (Intermediate Level)

Distance-based algorithms are sensitive to feature scales.

### Problem
Feature A: Age (0–100)
Feature B: Salary (0–1,00,000)

Salary dominates distance calculation.

### Solution
- Min-Max Scaling
- Standardization (Z-score)

---

## 10. Choosing K Value

### Small K
- Low bias
- High variance
- Sensitive to noise

### Large K
- High bias
- Low variance
- Smoother decision boundary

### Best Practice
Use **cross-validation** to find optimal K.

---

## 11. Time & Space Complexity (Advanced)

### Training Time
- O(1) (just storing data)

### Prediction Time
- O(n × d)
  - n = number of training samples
  - d = number of features

### Space Complexity
- O(n × d)

---

## 12. Optimizations for Large Datasets

### 1. KD-Tree
- Efficient for low-dimensional data

### 2. Ball Tree
- Better for higher dimensions

### 3. Approximate Nearest Neighbors (ANN)
- Used in large-scale systems (recommendation engines)

---

## 13. Advantages of KNN

✅ Simple and intuitive  
✅ No training time  
✅ Non-linear decision boundaries  
✅ Works well with small datasets

---

## 14. Disadvantages of KNN

❌ Slow prediction for large datasets  
❌ High memory usage  
❌ Sensitive to noise and scaling  
❌ Curse of dimensionality

---

## 15. When to Use KNN

- Small to medium datasets
- Low dimensional data
- Quick baseline model
- Recommendation systems

---

## 16. Summary

| Aspect | KNN |
|------|-----|
| Type | Supervised, Lazy Learning |
| Training Output | Stored dataset |
| Model Parameters | None |
| Prediction | Distance-based |
| Use Cases | Classification & Regression |

---

✅ **Key Takeaway:**

> KNN does not learn a model; it memorizes data and makes decisions based on similarity.

---

*Prepared for learning machine learning concepts step-by-step.*
