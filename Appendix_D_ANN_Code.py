import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import metrics

df = pd.read_csv(r"C:\Users\Jayasuriya Nandhagop\Desktop\Energy Management- Main project\FinalCleanData.csv", header=0, delimiter=',')
df2 = df.dropna(axis=0, how='any')
pd.set_option('display.max_columns', None)
X = df2[["T_ctrl", "RH_out", "T_out", "T_stp_heat", "Wind speed (m/s)"]].values
y = df2["auxHeat1"].values
y = y/5
print(y)
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#print(X)
#print(y)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
print(X_train)
print(X_test)
plt.figure(figsize=(12,10))
plt.imshow(plt.imread("../input/aritificialneural-network/ann.png"))
ann = Sequential()                          # Initializing the ANN
ann.add(Dense(units=12, activation="relu"))  #Adding First Hidden Layer
ann.add(Dense(units=12, activation="relu"))  # Adding Second Hidden Layer
ann.add(Dense(units=1))
ann.compile(optimizer="Adam", loss="mean_squared_error")
ann.fit(x=X_train, y=y_train, epochs=50, batch_size=30, validation_data=(X_test,y_test),callbacks=EarlyStopping(monitor='val_loss',patience=4))
print(pd.DataFrame(ann.history.history))
plt.style.use("ggplot")
pd.DataFrame(ann.history.history).plot(figsize=(15,15))
plt.show()
print(ann.evaluate(X_train,y_train))
print(ann.evaluate(X_test,y_test))
predictions = ann.predict(X_test)
predictions_df = pd.DataFrame(np.ravel(predictions),columns=["Predictions"])
comparison_df = pd.concat([pd.DataFrame(y_test,columns=["Real Values"]), predictions_df],axis=1)
print(comparison_df)
print(y_test.shape)       # The actual values are 1D arrays
print(predictions.shape)
plt.figure(figsize=(18,18))
r = np.ravel(predictions)
sns.scatterplot(r, y_test)
plt.title("The Scatterplot of Relationship between Actual Values and Predictions")
plt.xlabel("Predictions")
plt.ylabel("Actual Values")
plt.show()
print("MAE:",metrics.mean_absolute_error(y_test,predictions))
print ("MSE:",metrics.mean_squared_error(y_test,predictions))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,predictions)))
print(metrics.explained_variance_score(y_test,predictions))
plt.figure(figsize=(18,18))
sns.distplot(y_test-predictions,bins=50)
plt.show()

