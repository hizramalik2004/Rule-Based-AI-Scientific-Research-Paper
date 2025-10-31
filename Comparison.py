import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv('loan_dataset.csv')

# --- Simple Rule-Based AI ---
def simple_loan_ai(applicant_data):
    income = applicant_data['annual_income']
    credit_score = applicant_data['credit_score']
    employed = applicant_data['employed']
    if income >= 50000 and credit_score >= 650 and employed == 1:
        return 1
    elif income >= 30000 and credit_score >= 700 and employed == 1:
        return 1
    else:
        return 0

# --- Advanced Rule-Based AI ---
def advanced_loan_ai(applicant_data):
    income = applicant_data['annual_income']
    credit_score = applicant_data['credit_score']
    employed = applicant_data['employed']
    loan_amount = applicant_data['loan_amount']
    dti = applicant_data['debt_to_income_ratio']
    loans = applicant_data['existing_loans']
    
    score = 0
    if income >= 75000:
        score += 3
    elif income >= 50000:
        score += 2
    elif income >= 30000:
        score += 1

    if credit_score >= 750:
        score += 4
    elif credit_score >= 700:
        score += 3
    elif credit_score >= 650:
        score += 2
    elif credit_score >= 600:
        score += 1

    if employed == 1:
        score += 2

    if dti <= 0.3:
        score += 2
    elif dti <= 0.5:
        score += 1

    if loan_amount > income * 0.5:
        score -= 2
    elif loan_amount > income * 0.3:
        score -= 1

    if loans > 2:
        score -= 1

    if score >= 8:
        return 1
    elif score >= 6:
        return 1
    else:
        return 0

# Apply both AIs
df['simple_ai_pred'] = df.apply(simple_loan_ai, axis=1)
df['advanced_ai_pred'] = df.apply(advanced_loan_ai, axis=1)

# --- Combined Comparison Table ---
comparison = pd.DataFrame({
    'AI Type': ['Simple AI', 'Advanced AI'],
    'Accuracy': [
        accuracy_score(df['loan_approved'], df['simple_ai_pred']),
        accuracy_score(df['loan_approved'], df['advanced_ai_pred'])
    ],
    'Precision': [
        precision_score(df['loan_approved'], df['simple_ai_pred']),
        precision_score(df['loan_approved'], df['advanced_ai_pred'])
    ],
    'Recall': [
        recall_score(df['loan_approved'], df['simple_ai_pred']),
        recall_score(df['loan_approved'], df['advanced_ai_pred'])
    ],
    'F1 Score': [
        f1_score(df['loan_approved'], df['simple_ai_pred']),
        f1_score(df['loan_approved'], df['advanced_ai_pred'])
    ]
})

print("\n--- Combined Performance Comparison ---\n")
print(comparison)

# --- Graph for Combined Comparison ---
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, comparison.iloc[0, 1:], width, label='Simple AI', color='skyblue')
plt.bar(x + width/2, comparison.iloc[1, 1:], width, label='Advanced AI', color='orange')

plt.ylim(0, 1)
plt.xticks(x, metrics)
plt.title('Comparison of Simple vs Advanced Rule-Based AIs')
plt.ylabel('Performance Score')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

for i, v in enumerate(comparison.iloc[0, 1:]):
    plt.text(x[i] - width/2, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
for i, v in enumerate(comparison.iloc[1, 1:]):
    plt.text(x[i] + width/2, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
