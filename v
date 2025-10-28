import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

abass = pd.read_csv('/content/accident.csv')

# Remove rows with missing values in 'Speed_of_Impact'
abass.dropna(subset=['Speed_of_Impact'], inplace=True)

x = pd.DataFrame(abass['Age'])
y = pd.DataFrame(abass['Speed_of_Impact'])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(regressor.intercept_)
