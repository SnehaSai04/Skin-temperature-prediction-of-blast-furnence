import pandas as pd
import numpy as np

data = pd.read_excel(r"C:\Users\User\OneDrive\Desktop\DataBlastFurnence.xlsx")
data = data.dropna()
x = data.iloc[:, 1:25]
y = data.iloc[:, 25:]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1000)

from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.25)
lasso_reg.fit(x_train, y_train)

a = lasso_reg.predict([[311727, 3.15, 129, 4, 213, 3.34, 3.2, 7296, 23.08, 32, 24.56, 1060, 2.99, 1.5, 112, 135, 107, 130, 0, 121, 2, 22.22, 21, 3.88]])
print(a)
print(lasso_reg.score(x_test, y_test) * 100)