import pandas as pd
import numpy as np
import re
from sklearn import preprocessing
import numpy
import matplotlib.pyplot as plot
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
le = preprocessing.LabelEncoder()
df = pd.read_excel(r"traindata.xlsx")
df = df.dropna()
df= df[df['Sales'] != 0]
week = (df["ISO_Week"]).tolist()
df['SKU']= le.fit_transform(df['SKU'])
weeklist = []
for x in week:
    data1  = re.sub("2018-","", x)
    weeklist.append(data1)
df["ISO_Week"] = np.asarray(weeklist)
x = df.iloc[:, :-2].values
y = df.iloc[:, 2].values

linearRegressor = LinearRegression()
linearRegressor.fit(x, y)

#----------------------TEST DATA--------------------------
df1 = pd.read_excel(r"testdata.xlsx")
weektest = (df1["ISO_Week"]).tolist()
weektestlist = []
for x in weektest:
    data1  = re.sub("2018-","", x)
    weektestlist.append(data1)
df1["ISO_Week"] = np.asarray(weektestlist)
df1['SKU']= le.fit_transform(df1['SKU'])
test = df1.iloc[:, :-2].values
sku  = []
W_date =[]
y_pre = []
for x in test:
    yPrediction = linearRegressor.predict([[int(x[0]), int(x[1])]])
    if x[0] == 0:
        sku.append("ProductA")
    elif x[0] == 1:
        sku.append("ProductB")
    else:
        sku.append("ProductC")
    W_date.append("2018-"+str(x[1]))
    y_pre.append(yPrediction[0])

# Dictionary with list object in values
data = {
    'SKU' : sku,
    'Week' : W_date,
    'Forecast' : y_pre
}
df3 = pd.DataFrame(data) 
df3.to_excel('output.xlsx')