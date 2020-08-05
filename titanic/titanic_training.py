import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def get_title(name1):
    if '.' in name1:
        return name1.split(',')[1].split('.')[0].strip()
    else:
        return 'No title in name'


def shorter_titles(xs):
    title = xs['Title']
    if title in ['Capt', 'Col', 'Major']:
        return 'Officer'
    elif title in ['Jonkheer', 'Don', 'the Countess', 'Lady', 'Sir', 'Dona']:
        return 'Royalty'
    elif title == 'Mme':
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    else:
        return title


df = pd.read_csv('train.csv')
df.loc[df['Age'] > 65, 'Age'] = df['Age'].median()
df['Embarked'].fillna('S', inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Title'] = df['Name'].map(lambda x: get_title(x))
df['Title'] = df.apply(shorter_titles, axis=1)
df.drop(['Name','Cabin','Ticket'],axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
df.Sex.replace(('male', 'female'), (0, 1), inplace=True)
df.Embarked.replace(('S', 'C', 'Q'), (0, 1, 2), inplace=True)
df.Title.replace(('Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev', 'Officer', 'Royalty'), (0),
                 inplace=True)

x = df.drop('Survived', axis=1)
y = df['Survived']
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)
randomforests = RandomForestClassifier()
randomforests.fit(x_train, y_train)
y_pred = randomforests.predict(x_val)
acc_randomforests = round(accuracy_score(y_pred, y_val) * 100, 2)
print(f'Accuracy: {acc_randomforests}')
pickle.dump(randomforests, open('titanic_model.sav', 'wb+'))
