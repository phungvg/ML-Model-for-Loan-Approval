# 📊 Loan Approval

ML Model for Loan Approval using Python and 1 dataset

Predict probability of Full Pay for consumer loans and convert it into approve/reject decisions that maximize profit. 

The pipeline guards against leakage, handles missing data, selects a model via CV, and picks a decision threshold by scanning profit.

---

## ⚙️ Setup

### 🔁 1. Clone the Repository

Use Git to download the project from GitHub:

```bash
git clone https://github.com/phungvg/ML-Model-for-Loan-Approval.git
cd ML-Model-for-Loan-Approval
```
### ☁️ 2. Install Git LFS
Large files (e.g., CSVs) are managed with Git LFS.

Download and install Git LFS: git-lfs.github.com

Then run:

```bash
git lfs install
git lfs pull
```
## == IT SHOULD BE ON YOUR DEVICE FROM THIS STEP ==

## Needed Tools
### 🛠️ Install Dependencies
The project requires Python libraries. 

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U pip
pip install numpy pandas scikit-learn matplotlib joblib
```

### ▶️ Run the Project

- [x] **demo_train.py**  
  - Best method: Random Forest(use 5 fold-cv)  
  
- [x] **output**
  - Include: Confusion matrix, pr_curve, metrics.json, model_rf.joblib

---

### 📈 Reference results
Selected model: Random Forest (CV ROC-AUC ≈ 0.941 on 200k subsample)

Test Accuracy: 0.943

ROC-AUC: 0.957

PR-AUC: 0.991

Profit-optimized threshold: 0.47

Estimated profit: $203.8M (Approved 59,400 / 69,346)

(Numbers can vary slightly by environment; seed = 42.)
