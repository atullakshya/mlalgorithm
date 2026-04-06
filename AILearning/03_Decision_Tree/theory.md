# Decision Trees Explained (From Basics to Advanced)

This document explains **Decision Trees**, **Gini Impurity**, and **Entropy** from **absolute basics to practical usage**, written for **freshers with no prior Machine Learning background**.

---

## 1. What Is a Decision Tree? (Very Basic)

A **Decision Tree** is a model that makes decisions by **asking a series of questions**, just like a human does.

Example (Real Life):
- Should I take an umbrella?
  - Is it cloudy?
    - Yes → Take umbrella
    - No → Don’t take umbrella

This question-based flow is exactly how a decision tree works.

---

## 2. Why Is It Called a Tree?

Because its structure looks like an **upside-down tree**:
- **Root Node** → First question
- **Decision Nodes** → Internal questions
- **Branches** → Answers (Yes/No or categories)
- **Leaf Nodes** → Final output

---

## 3. Types of Decision Trees

### 3.1 Classification Tree
- Output is a **category**
- Examples: Spam / Not Spam, Loan Approved / Rejected

### 3.2 Regression Tree
- Output is a **number**
- Examples: House price, Salary prediction

---

## 4. Core Terminology

| Term | Meaning |
|----|--------|
| Feature | Input variable (Age, Salary) |
| Target | Output to predict |
| Node | A question |
| Branch | An answer to a question |
| Leaf | Final decision |

---

## 5. Real-World Example: Loan Approval

### Training Data

| Age | Salary | Credit Score | Loan Approved |
|----|------|-------------|---------------|
| 25 | 30k | Low | No |
| 28 | 40k | Medium | Yes |
| 45 | 80k | High | Yes |
| 35 | 50k | Medium | Yes |
| 23 | 25k | Low | No |

- Features: Age, Salary, Credit Score
- Target: Loan Approved

---

## 6. How Decision Tree Training Works

1. Start with all data at root node
2. Try all possible questions on all features
3. Measure how good each split is
4. Pick the **best split**
5. Repeat for child nodes
6. Stop when nodes are pure or limits reached

The key question is:
> **How do we measure the “best” split?**

Answer: Using **Gini Impurity or Entropy**.

---

## 7. What Is Gini Impurity?

### Simple Meaning
Gini Impurity tells us:
> **How mixed the data is in a node**

Lower Gini → Better

---

### Gini Formula
```
Gini = 1 - Σ(pᵢ²)
```

- pᵢ = proportion of each class

---

### Gini Example

Node data:
- Yes = 6
- No = 4

Probabilities:
- p(Yes) = 0.6
- p(No) = 0.4

Calculation:
```
Gini = 1 - (0.6² + 0.4²)
= 0.48
```

Interpretation: Node is fairly impure.

---

## 8. What Is Entropy?

### Simple Meaning
Entropy measures:
> **How uncertain or confused the node is**

Lower Entropy → Better

---

### Entropy Formula
```
Entropy = - Σ(pᵢ log₂ pᵢ)
```

Based on **Information Theory**.

---

### Entropy Example

Same node:
- p(Yes) = 0.6
- p(No) = 0.4

Calculation:
```
Entropy = - (0.6 log₂ 0.6 + 0.4 log₂ 0.4)
≈ 0.971
```

Interpretation: High uncertainty.

---

## 9. Information Gain (Used with Entropy)

```
Information Gain = Entropy(before) - Entropy(after)
```

The split with **highest information gain** is chosen.

---

## 10. How Splits Are Chosen

| Metric | Optimization Goal |
|----|------------------|
| Gini | Minimize weighted Gini |
| Entropy | Maximize information gain |

Both usually produce similar trees.

---

## 11. Comparison: Gini vs Entropy

| Feature | Gini | Entropy |
|------|------|--------|
| Measures | Impurity | Uncertainty |
| Max (Binary) | 0.5 | 1 |
| Computation | Faster | Slower |
| Theory | Probability | Information Theory |
| sklearn Default | ✅ | ❌ |

---

## 12. Loss and Optimization in Decision Trees

- No gradient descent
- No backpropagation
- Uses **greedy optimization**
- Selects best split at each step

Loss is minimized by reducing:
- Impurity (Gini)
- Uncertainty (Entropy)

---

## 13. Overfitting in Decision Trees

Decision Trees can memorize data.

### Controls:
- Max depth
- Min samples per leaf
- Pruning

---

## 14. Model Output After Training

The trained model is a **tree of rules**, not equations.

Example:
```
IF Credit Score = High → Approve
ELSE IF Salary > 45k → Approve
ELSE → Reject
```

---

## 15. Key Takeaways

- Decision trees mimic human decision-making
- Gini and Entropy measure node quality
- Training is greedy and rule-based
- Trees are interpretable and powerful

---

## 16. Interview One-Liners

- Decision Tree: Rule-based ML model
- Gini: Probability of misclassification
- Entropy: Measure of uncertainty
- Difference: Same goal, different math

---

## 17. Final Conclusion

Decision Trees use **Gini Impurity or Entropy** to learn the best questions from data. Both aim to create **pure, confident decisions**, making decision trees easy to understand, explain, and apply in real-world problems.
