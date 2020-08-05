from django.shortcuts import render
from django.views.decorators.cache import never_cache

from .data_converting import convert_data
import pickle

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

    context = {'prediction': get_real_predict(convert_data(pclass, sex, age, sibsp, parch, fare, embarked, title))}

    return render(request, 'result.html', context=context)


def get_real_predict(x):
    randomforests = pickle.load(open('titanic_model.sav', 'rb'))
    prediction = randomforests.predict(x)
    return prediction
