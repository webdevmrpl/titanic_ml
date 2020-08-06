from django.shortcuts import render
from django.views.decorators.cache import never_cache
from keras.models import load_model
from .data_converting import convert_data

@never_cache
def index(request):
    return render(request, 'index.html')


# pclass, sex, age, sibsp, parch, fare, embarked, title
@never_cache
def result(request):
    pclass = float(request.POST['pclass'])
    sex = request.POST['sex']
    age = float(request.POST['age'])
    sibsp = float(request.POST['sibsp'])
    parch = float(request.POST['parch'])
    fare = float(request.POST['fare'])
    embarked = request.POST['embarked']
    title = request.POST['name']
    model_predict = load_model('titanic_NN.h5')
    # 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked', 'Title'
    prediction = model_predict.predict(convert_data(pclass, sex, age, sibsp, parch, fare, embarked, title))
    if prediction < 0.5:
        prediction = 'Not Survived'
    else:
        prediction = 'Survived'

    context = {'prediction': prediction}

    return render(request, 'result.html', context=context)
