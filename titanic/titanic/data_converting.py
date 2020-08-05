import pandas as pd


def convert_data(pclass, sex, age, sibsp, parch, fare, embarked, name):
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

    df = pd.DataFrame(
        {'PClass': pclass, 'Sex': sex, 'Age': age, 'SibSp': sibsp, 'Parch': parch, 'Fare': fare, 'Embarked': embarked,
         'Name': name}, index =[0,1,2,3,4,5,6,7] )
    df.loc[df['Age'] > 65, 'Age'] = df['Age'].median()
    df['Embarked'].fillna('S', inplace=True)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Title'] = df['Name'].map(lambda x: get_title(x))
    df['Title'] = df.apply(shorter_titles, axis=1)
    df.Sex.replace(('male', 'female'), (0, 1), inplace=True)
    df.Embarked.replace(('S', 'C', 'Q'), (0, 1, 2), inplace=True)
    df.Title.replace(('Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev', 'Officer', 'Royalty'), (0),
                     inplace=True)
    pclass, sex, age, sibsp, parch, fare, embarked, name = df['PClass'][0], df['Sex'][0], df['Age'][0], df['SibSp'][0], df['Parch'][0], \
                                                           df['Fare'][0], df['Embarked'][0], df['Title'][0]
    return [[pclass, sex, age, sibsp, parch, fare, embarked, name]]

