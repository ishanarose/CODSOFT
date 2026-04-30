import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("C:/Users/hp/Downloads/Titanic-Dataset.csv")

# ---------------------------
# DATA CLEANING
# ---------------------------

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

# Drop unnecessary columns
df = df.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'], errors='ignore')

# Convert categorical to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encoding
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# ---------------------------
# CHECK DATA
# ---------------------------
print("Missing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)

# ---------------------------
# MODEL
# ---------------------------

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------------------
# PREDICTION
# ---------------------------

y_pred = model.predict(X_test)

# ---------------------------
# RESULTS
# ---------------------------

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))
df.to_excel("cleaned_titanic.xlsx",index=False)