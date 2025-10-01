import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

df = pd.DataFrame()
df = pd.read_csv('samples_data.csv')
df = df.drop(columns=['condition'])
encoder = LabelEncoder()
df['material'] = encoder.fit_transform(df['material'])
X = df.drop(columns=['damage'])
y = df['damage']
print (df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Random forest distribution' : RandomForestClassifier(n_estimators=100,random_state=42),
    'gradient boosting' : GradientBoostingClassifier(random_state=42),
    'Logistic regression' : LogisticRegression(random_state=42)
}
results = {}
for name, model in models.items():
     model.fit(X_train,y_train)
     y_pred = model.predict(X_test)
     acc = accuracy_score(y_test,y_pred)
     f1 = f1_score(y_test, y_pred)
     results[name] = {'acc':acc, 'f1':f1}
     print(f'{name}: {acc} | {f1}')
     print(classification_report(y_test, y_pred))

results_df = pd.DataFrame(results)
results_df.plot(kind='bar', figsize=(8,5))
plt.title('Accuracy vs F1-score')
plt.ylabel("score")
plt.show()