import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    'Area': [800, 900, 1000, 1100, 1200, 1300, 1400],
    'Price': [3000000, 3500000, 4000000, 4500000, 5000000, 5500000, 6000000]
}

df = pd.DataFrame(data)

X = df[['Area']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

area_input = int(input("Enter area in sq ft: "))
prediction = model.predict([[area_input]])

print("Predicted House Price:", int(prediction[0]))
