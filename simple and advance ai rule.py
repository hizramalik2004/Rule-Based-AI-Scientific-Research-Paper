import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load loan dataset
df = pd.read_csv('loan_dataset.csv')

# Simple Rule-based AI: 
def simple_loan_ai(applicant_data):
    """
    Simple rule-based AI for loan approval
    Basic criteria: Income, Credit Score, and Employment Status
    """
    income = applicant_data['annual_income']
    credit_score = applicant_data['credit_score']
    employed = applicant_data['employed']
    
    # Simple rules
    if income >= 50000 and credit_score >= 650 and employed == 1:
        return 1  # Approve
    elif income >= 30000 and credit_score >= 700 and employed == 1:
        return 1  # Approve
    else:
        return 0  # Reject

# Predict using simple AI
df['simple_ai_pred'] = df.apply(simple_loan_ai, axis=1)

# Evaluation for Simple AI
accuracy_simple = accuracy_score(df['loan_approved'], df['simple_ai_pred'])
precision_simple = precision_score(df['loan_approved'], df['simple_ai_pred'])
recall_simple = recall_score(df['loan_approved'], df['simple_ai_pred'])
f1_simple = f1_score(df['loan_approved'], df['simple_ai_pred'])

print('Simple Loan AI Evaluation:')
print(f'Accuracy: {accuracy_simple:.4f}')
print(f'Precision: {precision_simple:.4f}')
print(f'Recall: {recall_simple:.4f}')
print(f'F1 Score: {f1_simple:.4f}\n')


import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load loan dataset
df = pd.read_csv('loan_dataset.csv')

# Advanced Rule-based AI: 
def advanced_loan_ai(applicant_data):
    """
    Advanced rule-based AI for loan approval
    Multiple factors with weighted scoring
    """
    income = applicant_data['annual_income']
    credit_score = applicant_data['credit_score']
    employed = applicant_data['employed']
    loan_amount = applicant_data['loan_amount']
    debt_to_income = applicant_data['debt_to_income_ratio']
    existing_loans = applicant_data['existing_loans']
    
    score = 0
    
    # Income scoring (0-3 points)
    if income >= 75000:
        score += 3
    elif income >= 50000:
        score += 2
    elif income >= 30000:
        score += 1
    
    # Credit score scoring (0-4 points)
    if credit_score >= 750:
        score += 4
    elif credit_score >= 700:
        score += 3
    elif credit_score >= 650:
        score += 2
    elif credit_score >= 600:
        score += 1
    
    # Employment status (0-2 points)
    if employed == 1:
        score += 2
    
    # Debt-to-income ratio (0-2 points)
    if debt_to_income <= 0.3:
        score += 2
    elif debt_to_income <= 0.5:
        score += 1
    
    # Loan amount consideration (-2 to 0 points)
    if loan_amount > income * 0.5:
        score -= 2
    elif loan_amount > income * 0.3:
        score -= 1
    
    # Existing loans penalty
    if existing_loans > 2:
        score -= 1
    
    # Decision based on total score
    if score >= 8:
        return 1  # Strong approval
    elif score >= 6:
        return 1  # Approve
    else:
        return 0  # Reject

# Predict using advanced AI
df['advanced_ai_pred'] = df.apply(advanced_loan_ai, axis=1)

# Evaluation for Advanced AI
accuracy_advanced = accuracy_score(df['loan_approved'], df['advanced_ai_pred'])
precision_advanced = precision_score(df['loan_approved'], df['advanced_ai_pred'])
recall_advanced = recall_score(df['loan_approved'], df['advanced_ai_pred'])
f1_advanced = f1_score(df['loan_approved'], df['advanced_ai_pred'])

print('Advanced Loan AI Evaluation:')
print(f'Accuracy: {accuracy_advanced:.4f}')
print(f'Precision: {precision_advanced:.4f}')
print(f'Recall: {recall_advanced:.4f}')
print(f'F1 Score: {f1_advanced:.4f}\n')



import pandas as pd
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

df['simple_ai_pred'] = df.apply(simple_loan_ai, axis=1)

# --- Advanced Rule-Based AI ---
def advanced_loan_ai(applicant_data):
    income = applicant_data['annual_income']
    credit_score = applicant_data['credit_score']
    employed = applicant_data['employed']
    loan_amount = applicant_data['loan_amount']
    debt_to_income = applicant_data['debt_to_income_ratio']
    existing_loans = applicant_data['existing_loans']
    
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
        
    if debt_to_income <= 0.3:
        score += 2
    elif debt_to_income <= 0.5:
        score += 1
        
    if loan_amount > income * 0.5:
        score -= 2
    elif loan_amount > income * 0.3:
        score -= 1
        
    if existing_loans > 2:
        score -= 1
        
    if score >= 8:
        return 1
    elif score >= 6:
        return 1
    else:
        return 0

df['advanced_ai_pred'] = df.apply(advanced_loan_ai, axis=1)

# --- Evaluation Metrics ---
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

simple_scores = [
    accuracy_score(df['loan_approved'], df['simple_ai_pred']),
    precision_score(df['loan_approved'], df['simple_ai_pred']),
    recall_score(df['loan_approved'], df['simple_ai_pred']),
    f1_score(df['loan_approved'], df['simple_ai_pred'])
]

advanced_scores = [
    accuracy_score(df['loan_approved'], df['advanced_ai_pred']),
    precision_score(df['loan_approved'], df['advanced_ai_pred']),
    recall_score(df['loan_approved'], df['advanced_ai_pred']),
    f1_score(df['loan_approved'], df['advanced_ai_pred'])
]

# --- Plot Simple AI Performance ---
plt.figure(figsize=(6,4))
plt.bar(metrics, simple_scores, color='skyblue', alpha=0.8)
plt.ylim(0, 1)
plt.title('Performance Metrics - Simple Rule-Based AI')
plt.ylabel('Score')
plt.grid(axis='y', linestyle='--', alpha=0.6)
for i, v in enumerate(simple_scores):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

# --- Plot Advanced AI Performance ---
plt.figure(figsize=(6,4))
plt.bar(metrics, advanced_scores, color='orange', alpha=0.8)
plt.ylim(0, 1)
plt.title('Performance Metrics - Advanced Rule-Based AI')
plt.ylabel('Score')
plt.grid(axis='y', linestyle='--', alpha=0.6)
for i, v in enumerate(advanced_scores):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()