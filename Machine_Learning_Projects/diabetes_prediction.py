import pandas as pd
from pandas.core.algorithms import mode
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('diabetes.csv')

# print(df.isna().sum())
# print(df.columns)
# print(df.shape)
df["SkinThickness"]=df["SkinThickness"].replace(0,df["SkinThickness"].mean())
# print(df)
df["Insulin"]=df["Insulin"].replace(0,df["Insulin"].mean())
# print(df)

plt.figure(figsize=(12,7))
plt.hist("Glucose",data=df,edgecolor="k")
plt.title("Glucose Hist Plot")
# plt.show()

plt.figure(figsize=(12,7))
plt.hist("BMI",data=df,edgecolor="k")
plt.title("BMI Hist Plot")
# plt.show()

plt.figure(figsize=(12,7))
plt.scatter("Pregnancies","Insulin",data=df)
plt.title("Pregnancies Vs Insulin")
plt.xlabel("Pregnancies")
plt.ylabel("Insulin")
# plt.show()

plt.figure(figsize=(12,7))
plt.scatter("SkinThickness","Insulin",data=df)
plt.title("SkinThickness Vs Insulin")
plt.xlabel("SkinThickness")
plt.ylabel("Insulin")
# plt.show()

plt.figure(figsize=(12,7))
plt.scatter("BloodPressure","Insulin",data=df)
plt.title("BloodPressure Vs Insulin")
plt.xlabel("BloodPressure")
plt.ylabel("Insulin")
# plt.show()

plt.figure(figsize=(12,7))
plt.scatter("Glucose","BMI",data=df)
plt.title("Glucose Vs BMI")
plt.xlabel("Glucose")
plt.ylabel("BMI")
# plt.show()


def remove_outlier(dataFrame):
    for column_name in dataFrame.columns:
        Q1=df[column_name].quantile(0.25)
        Q3=df[column_name].quantile(0.75)
        IQR=Q3-Q1
        lower_limit=Q1-1.5*IQR
        upper_limit=Q1+1.5*IQR
        print(f"{column_name} >> Lower limit: {lower_limit} \n Upper limit: {upper_limit} ")
        dataFrame=dataFrame[(dataFrame[column_name]>lower_limit)|(dataFrame[column_name]< upper_limit)]
    return dataFrame

remove_outlier(df)

X=df.drop(['Outcome'],axis=1)
y=df['Outcome']
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=42)
logReg=LogisticRegression()
logReg.fit(X_train,y_train)
logReg.score(X_test,y_test)
predictions=logReg.predict(X_test)
cm=confusion_matrix(y_test,predictions)
sns.heatmap(cm,annot=True)
print(classification_report(y_test,predictions))