import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
#DATASET LOAD
try:
    data = pd.read_csv("house_data.csv")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("house_data.csv not found.")
    exit()
#Data Cleaning
    print(data.isnull().sum())

data = data.dropna()

print(data.head())

#Model Traning
# Step 5: Feature aur Target select karo
X = data.drop("Price", axis=1)
y = data["Price"]

# Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression Model
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully!")

# Step 6: Prediction

y_pred = model.predict(X_test)

# Model Evaluation
from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

#Actual vs Predicted House Price Graph
# Step 7: Actual vs Predicted Price Graph

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")

plt.title("Actual vs Predicted House Price")

plt.savefig("IMAGES/actual_vs_predicted.png", dpi=300, bbox_inches="tight")

plt.show()

#Model Performance
# Step 8: Model Performance

print("\n===== Model Performance =====")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# Final Result
if r2 >= 0.80:
    print("Excellent Model Performance")
elif r2 >= 0.60:
    print("Good Model Performance")
else:
    print("Model Needs Improvement")

    # Compare Actual vs Predicted Values

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted Price")
print(comparison.head(10))



plt.scatter(data["Area"], data["Price"])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("House Price Prediction")
plt.show()
