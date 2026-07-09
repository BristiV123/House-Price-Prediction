import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("house_data.csv")

print(data.head())

X = data[['Area', 'Bedrooms', 'Bathrooms', 'Parking']]
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)

print("Predicted House Prices:")
print(prediction)
score = r2_score(y_test, prediction)

print("R2 Score:", score)
plt.scatter(data["Area"], data["Price"])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("House Price Prediction")
plt.show()
