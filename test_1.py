import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = pd.DataFrame()
df = pd.read_csv('samples_data.csv')
df = df.drop(columns=['condition'])
print(df)

encoder = LabelEncoder()
df['material'] = encoder.fit_transform(df['material'])
X = df.drop(columns=['damage'])
y = df['damage']
print (df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf=RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(rf.score(X_test, y_test))
print(classification_report(y_test, y_pred))
print(y_pred)

importances = rf.feature_importances_
features = X.columns

plt.figure(figsize=[8,8])
plt.barh(features, importances)
plt.xlabel('Importance')
plt.title('Feature Importances with material')
plt.show()

X = df.drop(columns = 'material')
y = df['material']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(rf.score(X_test, y_test))
print(classification_report(y_test, y_pred))
print(y_pred)
importances = rf.feature_importances_
features = X.columns

plt.figure(figsize=[8,8])
plt.barh(features, importances)
plt.xlabel('Importance')
plt.title('Feature Importances without material')
plt.show()


