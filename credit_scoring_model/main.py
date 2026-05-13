# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Step 1: Load dataset
data = pd.read_csv("credit_data.csv")

# Step 2: Split features and target
X = data.drop("target", axis=1)
y = data["target"]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate model
y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

# Step 6: Save model
with open("credit_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel trained and saved successfully!")

#------------------------------------------------------------------
# USER INPUT SECTION
#------------------------------------------------------------------

print("\nEnter customer details:")

age = int(input("Age: "))
income = int(input("Income: "))
loan_amount = int(input("Loan Amount: "))
credit_score = int(input("Credit Score: "))
loan_term = int(input("Loan Term (months): "))
missed_payments = int(input("Missed Payments: "))
employment_years = int(input("Employment Years: "))

# Prepare input
user_data = [[
    age,
    income,
    loan_amount,
    credit_score,
    loan_term,
    missed_payments,
    employment_years
]]

# Prediction
prediction = model.predict(user_data)

# Result
print("\nResult:")
if prediction[0] == 1:
    print("Eligible for Credit")
else:
    print("Not Eligible for Credit")
