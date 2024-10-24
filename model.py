# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import joblib

# Set random seed for reproducibility
seed = 42

# Read the original dataset
iris_df = pd.read_csv("data/iris.csv")

# Shuffle the dataset (useful if the data is sorted in some way)
iris_df = iris_df.sample(frac=1, random_state=seed)

# Selecting features and target
X = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Species']  # You don't need double brackets here unless you need a DataFrame

# Split data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y
)

# Create an instance of the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=seed)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")  # Limit to 2 decimal places



# save the model to disk
joblib.dump(clf, "rf_model.sav")
