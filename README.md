# Musculoskeletal Abnormality Detection from X-ray Images

This project builds a machine learning pipeline to detect **musculoskeletal abnormalities** from radiographic (X-ray) images of the upper extremities. The goal is to classify each radiograph as either **normal** or **abnormal**.

The model is trained using **PyTorch** and predicts binary labels:

- **0 – Negative:** Normal radiograph  
- **1 – Positive:** Abnormal radiograph  

The system loads X-ray images, preprocesses them, trains and generates predictions for a test dataset.

---

# Installation

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install torch torchvision scikit-learn pandas numpy pillow
```

---

# Running the Project

Run the full pipeline using:

```bash
python main.py
```

This will:

1. Load and preprocess the dataset  
2. Train the classification model  
3. Save the best model weights to:

```
outputs/models/best_model.pth
```

4. Generate predictions for the test set:

```
outputs/test.csv
```

---

# Output Files

| File | Description |
|-----|-------------|
| `best_model.pth` | Saved trained model weights |
| `test.csv` | Predictions for the test dataset |

---

# Model Training

The training process includes:

- Data preprocessing and normalization  
- Train / validation split  
- Model optimization using stochastic gradient descent  
- Evaluation using **Cohen's Kappa score**

Threshold search is used to determine the best probability threshold for classification.
