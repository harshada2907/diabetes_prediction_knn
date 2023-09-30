import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

data = pd.read_csv("dbsep23.csv")
print(data.shape)
print(data)


data[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = data[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NAN)

print(data.isnull().sum())

data.fillna({
	"Glucose" : data["Glucose"].mean(),
	"BloodPressure" : data["BloodPressure"].mean(),
	"SkinThickness" : data["SkinThickness"].mean(),
	"Insulin" : data["Insulin"].mean(),
	"BMI" : data["BMI"].mean()
}, inplace = True)

print(data.isnull().sum())

features = data.drop("Outcome", axis = "columns")
target = data["Outcome"]

k = int(len(data) ** 0.5)
if k % 2 == 0:
	k = k + 1

mms = MinMaxScaler()
nfeatures = mms.fit_transform(features)

print(features)
print(nfeatures)

x_train, x_test, y_train, y_test = train_test_split(nfeatures, target)

model = KNeighborsClassifier(n_neighbors = k, metric = "euclidean")
model.fit(x_train, y_train)

cr = classification_report(y_test, model.predict(x_test))
print(cr)

preg = int(input("How many times pregnant : "))
gl = float(input("Enter your glucose level : "))
bp = float(input("Enter your blood pressure : "))
st = float(input("Enter your skin thickness : "))
ins = float(input("Enter your insulin level : "))
bmi = float(input("Enter your bmi : "))
dpf = float(input("Enter your diabetes pedigree function : "))
age = float(input("Enter your age : "))

d = [[preg, gl, bp, st, ins, bmi, dpf, age]]

ans = model.predict(d)
print(ans)
if ans[0] == 0:
	print("no")
else:
	print("yes")
