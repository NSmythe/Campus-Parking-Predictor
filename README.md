# Campus Parking Predictor (COMP 365)
**Predicting Space Availability across NC A&T Campus Lots via Machine Learning**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/188QoHZ_zIiOE8J3QoDVwJJijmRMpGZ1g)

## Team Parallel Perfection
- **Aniya Teasley:** Team Lead & Data Preparation Lead
- **Daiyana Brooks:** Data Collection Lead
- **Natalia Smythe:** Machine Learning Lead
- **Brianna Williams:** Testing & Evaluation Lead

---

## Project Overview
The **Campus Parking Predictor** applies supervised machine learning to forecast whether designated NC A&T campus parking areas (Obermeyer Lot, Laurel Deck, Stadium Lot) are likely to have available spaces based on:
1. **`parking_area`**: Target lot identifier
2. **`day_of_week`**: Operational day (Monday–Friday)
3. **`time_of_day`**: Peak class turnover blocks (08:00 AM – 04:00 PM)

---

## Model Benchmark and Results (Checkpoint 3)
Evaluated on an identical stratified held-out test partition (N = 94 records, 25% split):

| Model / Approach | Accuracy | Balanced Accuracy | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority-Class Baseline** | 61.70% | 50.00% | 0 | 0 | 36 | 58 |
| **Logistic Regression** | 74.47% | 68.77% | 18 | 18 | 6 | 52 |
| **Decision Tree (`max_depth=5`)** | 72.34% | 66.52% | 15 | 21 | 5 | 53 |
| **Random Forest (`n_estimators=50`)** | **76.60%** | **72.08%** | **19** | **17** | **5** | **53** |

*Note on Metric Verification:* Balanced accuracy is calculated as the unweighted mean of sensitivity and specificity, preventing naive models from scoring high by merely predicting the majority class during peak rush hours.

---

## How to Run the Technical Artifact

### Option 1: Run in Google Colab (Zero Setup)
Click the badge at the top or open:
[Google Colab Notebook Link](https://colab.research.google.com/drive/188QoHZ_zIiOE8J3QoDVwJJijmRMpGZ1g)

### Option 2: Run Locally via Python
1. Clone this repository:
   ```bash
   git clone https://github.com/NSmythe/Campus-Parking-Predictor.git
   cd Campus-Parking-Predictor

2. Install dependencies:
pip install -r requirements.txt

3. Execute the full training and inference pipeline:
python campus_parking_predictor.py

Ethics and Privacy Assurance

Zero Personally Identifiable Information (PII): Tracks aggregate lot space availability only. Strictly no vehicle license plates, vehicle makes, or student identifiers are collected or processed.
