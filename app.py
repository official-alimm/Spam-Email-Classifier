import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("emails.csv", sep=",")
df.columns = df.columns.str.strip().str.lower()

print(df.head())
print(df.columns)

# Clean column names (fix errors)
df.columns = df.columns.str.strip().str.lower()

print("Columns:", df.columns)  # Debug check

# Check column exists
if 'email' not in df.columns:
    print("❌ ERROR: 'email' column not found!")
    exit()

# Features and labels
X = df['email']
y = df['label']

# Convert text to numbers
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vectorized, y)

print("\n✅ Model trained successfully!")

# Test loop
while True:
    text = input("\nEnter email (type 'exit' to quit): ")
    if text.lower() == 'exit':
        break

    result = model.predict(vectorizer.transform([text]))
    print("Result:", result[0])