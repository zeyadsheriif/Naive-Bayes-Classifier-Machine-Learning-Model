import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

data = '/content/titanic_dataset.csv'
df = pd.read_csv(data, header=None, sep=',')

df.shape
df.info()
df.describe()
df.dtypes

col_names = ['PassengerId',	'Survived',	'Pclass',	'Name',	'Sex',	'Age',	'SibSp',	'Parch',	'Ticket',	'Fare',	'Cabin'	,'Embarked']
df.columns = col_names
df.columns

df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

#check for missing data
print('missing values -> {}'.format (df.isna().sum()))  # -> why ??

df.dropna(inplace= True)

categorical = [var for var in df.columns if df[var].dtype=='O']
categorical.remove('Sex')
categorical.remove('Survived')

print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n\n', categorical)

df.drop(0, inplace=True)
df.reset_index(drop=True, inplace=True)

for var in categorical:
    print(df[var].value_counts())

df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1})
df['Sex'] = df['Sex'].astype('int')
df['Survived'] = df['Survived'].astype('int')

# find numerical variables
numerical = [var for var in df.columns if df[var].dtype!='O']
print('There are {} numerical variables\n'.format(len(numerical)))
print('The numerical variables are :', numerical)

X = df.drop(['Survived'], axis=1)
y = df['Survived']

X=pd.get_dummies(X,columns=X[categorical[:8]].columns,dtype='int64')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# check the shape of X_train and X_test
X_train.shape, X_test.shape

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# train a Gaussian Naive Bayes classifier on the training set
bnb =  BernoulliNB()

# fit the model
bnb.fit(X_train, y_train)

y_pred = bnb.predict(X_test)
y_pred

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

scores = cross_val_score(bnb, X, y, cv=5, scoring='accuracy')
print("Cross-validation scores:", scores)
print("Average cross-validation score:", scores.mean())

# Print the Confusion Matrix and slice it into four pieces
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])

# visualize confusion matrix with seaborn heatmap
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                                 index=['Predict Positive:1', 'Predict Negative:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

# print the first 10 predicted probabilities of two classes- 0 and 1
y_pred_prob = bnb.predict_proba(X_test)[0:10]
y_pred_prob

# store the probabilities in dataframe
y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Survived', 'Not_Survived'])
y_pred_prob_df

