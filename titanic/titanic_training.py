import pandas as pd
import pickle
import numpy as np
from keras.layers import Sequential
from keras.layers import Dropout, Dense

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
df.Title.replace(('Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev', 'Officer', 'Royalty'), (0,1,2,3,4,5,6,7),inplace=True)

###################################################################

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(8,)))
model.add(Dropout(rate=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
x = df.drop('Survived',axis=1)
y = df['Survived']
model_train = model.fit(x,y, epochs=500, batch_size=50, verbose=0, validation_split=0.06)
model.save('titanic_NN.h5')