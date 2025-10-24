from flask import Flask, render_template, request, session, url_for, Response
import pandas as pd
import numpy as np
from werkzeug.utils import redirect
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from random import randint
import time
import json

accuracy =[]

app = Flask(__name__)
global dt1, LR1, RF1, NB1, KNN1, SVM1


def f(x_train, x_test, y_train, y_test):
    global X_trains, X_tests, y_trains, y_tests
    X_trains = pd.DataFrame(x_train)
    X_tests = pd.DataFrame(x_test)
    y_trains = pd.DataFrame(y_train)
    y_tests = pd.DataFrame(y_test)
    print("HELLO ++++++++++++++++++++++++++++++++++++++++++++++WORLD")
    print(X_trains)

    return X_trains, X_tests, y_trains, y_tests


def scores(score):
    global score1
    score1 = []
    # if sc

    return score1


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload')
def registration():
    return render_template('uploaddataset.html')


@app.route('/traintest')
def traintestvalue():
    return render_template('traintestdataset.html')


@app.route('/modelperformance')
def modelperformances():
    return render_template('modelperformance.html')


@app.route('/prediction')
def predictions():
    return render_template('prediction.html')


@app.route('/bar_chart22222222222222222')
def bar_charts():
    return render_template('bar_chart.html')


labels = [
    'JAN', 'FEB', 'MAR', 'APR',
    'MAY', 'JUN', 'JUL', 'AUG',
    'SEP', 'OCT', 'NOV', 'DEC'
]

values = [
    967.67, 1190.89, 1079.75, 1349.19,
    2328.91, 2504.28, 2873.83, 4764.87,
    4349.29, 6458.30, 9907, 16297
]

colors = [
    "#F7464A", "#46BFBD", "#FDB45C", "#FEDCBA",
    "#ABCDEF", "#DDDDDD", "#ABCABC", "#4169E1",
    "#C71585", "#FF4500", "#FEDCBA", "#46BFBD"]


@app.route("/data")
def chart_data(data=None):
    data_set = []

    for x in range(0, 12):
        y = randint(1, 12)
        data_set.append(y)

    data = {}

    data['set'] = data_set

    js = json.dumps(data)

    resp = Response(js, status=200, mimetype='application/json')

    return resp


@app.route("/bar_chart")
def hello(data=None):
    data = {}
    data['title'] = 'Chart'
    print(data['title'])
    print(data)

    return render_template('index1.html', data=data)


@app.route('/uploaddataset', methods=["POST", "GET"])
def uploaddataset_csv_submitted():
    if request.method == "POST":
        csvfile = request.files['csvfile']
        result = csvfile.filename
        file = "G:/Diabetes Prediction Using Data Science/" + result
        print(file)

        session['filepath'] = file

        return render_template('uploaddataset.html', msg='sucess')
    return render_template('uploaddataset.html')


@app.route('/viewdata', methods=["POST", "GET"])
def viewdata():
    session_var_value = session.get('filepath')
    print("Hello world")
    print("session variable is=====" + session_var_value)
    df = pd.read_csv(session_var_value)
    # print(df)
    x = pd.DataFrame(df)

    return render_template("view.html", data=x.to_html())

    # return render_template('view.html', name=session_var_value, data=df.to_html())


@app.route('/traintestdataset', methods=["POST", "GET"])
def traintestdataset_submitted():
    if request.method == "POST":
        value = request.form['traintestvalue']
        print("train test value is=============" + value)
        value1 = float(value)
        print(value1)
        filepath = session.get('filepath')
        df1 = pd.read_csv(filepath)
        X = df1[
            ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
             'Age']]
        y = df1['Outcome']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=value1)

        f(X_train, X_test, y_train, y_test)
        print("world------------------------------------------------------HELLO")
        print(X_trains)

        X_train1 = pd.DataFrame(X_train)
        X_trainlen = len(X_train)
        # session['X_trainlen']=X_trainlen

        y_test1 = pd.DataFrame(y_test)
        y_testlen = len(y_test)
        # session['y_testlen'] = y_testlen

        return render_template('traintestdataset.html', msg='sucess', data=X_train1.to_html(),
                               X_trainlenvalue=X_trainlen, y_testlenval=y_testlen)
    return render_template('traintestdataset.html')


@app.route('/modelperformance', methods=["POST", "GET"])
def selected_model_submitted():
    if request.method == "POST":
        selectedalg = int(request.form['algorithm'])


        print(X_trains)
        print(X_tests)
        print(
            "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(y_trains)
        print(
            "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(y_tests)

        if (selectedalg == 1):
            model = tree.DecisionTreeClassifier()

            model.fit(X_trains, y_trains)
            model.fit(X_trains, y_trains)
            y_pred = model.predict(X_tests)
            accuracyscore = accuracy_score(y_tests, y_pred)
            # accuracy_score = model.score(X_trains,y_trains)
            abc4 = accuracyscore
            accuracy.append(abc4)
            print(accuracyscore)



            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore,
                                   model="DecisionTree")

        if (selectedalg == 2):
            model = linear_model.LogisticRegression()
            model.fit(X_trains, y_trains)
            y_pred = model.predict(X_tests)
            accuracyscore = accuracy_score(y_tests, y_pred)
            # accuracy_score = model.score(X_trains,y_trains)
            print(accuracyscore)
            abc5 = accuracyscore
            accuracy.append(abc5)

            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore,
                                   model="LogisticRegression")

        if (selectedalg == 3):
            model = RandomForestClassifier(n_estimators=20)
            model.fit(X_trains, y_trains)
            y_pred = model.predict(X_tests)
            accuracyscore = accuracy_score(y_tests, y_pred)
            # accuracy_score = model.score(X_trains,y_trains)
            print(accuracyscore)
            abc6 = accuracyscore
            accuracy.append(abc6)
            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore,
                                   model="RandomForest")

        if (selectedalg == 4):
            model = GaussianNB()
            model.fit(X_trains, y_trains)
            y_pred = model.predict(X_tests)
            accuracyscore = accuracy_score(y_tests, y_pred)
            # accuracy_score = model.score(X_trains,y_trains)
            print(accuracyscore)
            abc7 = accuracyscore
            accuracy.append(abc7)
            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore,
                                   model="NaiveBayes")

        if (selectedalg == 5):
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(X_trains, y_trains)
            y_pred = model.predict(X_tests)
            accuracyscore = accuracy_score(y_tests, y_pred)
            # accuracy_score = model.score(X_trains,y_trains)
            print(accuracyscore)
            abc8 = accuracyscore
            accuracy.append(abc8)
            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore,
                                   model="KNearestNeighbors")

        if (selectedalg == 6):
            model = SVC(kernel='linear')
            model.fit(X_trains, y_trains)
            y_pred = model.predict(X_tests)
            accuracyscore = accuracy_score(y_tests, y_pred)
            # accuracy_score = model.score(X_trains,y_trains)
            print(accuracyscore)
            abc9 = accuracyscore
            accuracy.append(abc9)
            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore,
                                   model="SupportVectorMachine")

    return render_template('modelperformance.html')


@app.route('/prediction', methods=["POST", "GET"])
def prediction():
    if request.method == "POST":
        list1 = []
        pre = request.form['pre']
        glu = request.form['glu']
        bp = request.form['bp']
        ski = request.form['ski']
        ins = request.form['ins']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']
        list1.extend([pre, glu, bp, ski, ins, bmi, dpf, age])
        print(list1)

        model = SVC()
        model.fit(X_trains, y_trains)
        predi = model.predict([list1])
        print(predi)
        pre = predi
        print(pre)

        return render_template('prediction.html', msg='predictsucess', predvalue=predi)
    return render_template('prediction.html')

@app.route('/graph',methods=['POST','GET'])
def dpr():
    selectedalg = 1
    selectedalga = 2
    selectedalgb = 3
    selectedalgc = 4
    selectedalgd = 5
    selectedalge = 6

    print(X_trains)
    print(X_tests)
    print(
        "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(y_trains)
    print(
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(y_tests)

    if (selectedalg == 1):
        model1= tree.DecisionTreeClassifier()

        model1.fit(X_trains, y_trains)
        model1.fit(X_trains, y_trains)
        y_pred = model1.predict(X_tests)
        accuracyscore1 = accuracy_score(y_tests, y_pred)
        # accuracy_score = model.score(X_trains,y_trains)
        abc4 = accuracyscore1
        accuracy.append(abc4)
        print(accuracyscore1)



    if (selectedalga == 2):
        model2 = linear_model.LogisticRegression()
        model2.fit(X_trains, y_trains)
        y_pred = model2.predict(X_tests)
        accuracyscore2 = accuracy_score(y_tests, y_pred)
        # accuracy_score = model.score(X_trains,y_trains)
        print(accuracyscore2)
        abc5 = accuracyscore2
        accuracy.append(abc5)

    if (selectedalgb == 3):
        model3 = RandomForestClassifier(n_estimators=20)
        model3.fit(X_trains, y_trains)
        y_pred = model3.predict(X_tests)
        accuracyscore3 = accuracy_score(y_tests, y_pred)
        # accuracy_score = model.score(X_trains,y_trains)
        print(accuracyscore3)
        abc6 = accuracyscore3
        accuracy.append(abc6)

    if (selectedalgc == 4):
        model4 = GaussianNB()
        model4.fit(X_trains, y_trains)
        y_pred = model4.predict(X_tests)
        accuracyscore4 = accuracy_score(y_tests, y_pred)
        # accuracy_score = model.score(X_trains,y_trains)
        print(accuracyscore4)
        abc7 = accuracyscore4
        accuracy.append(abc7)


    if (selectedalgd == 5):
        model5 = KNeighborsClassifier(n_neighbors=5)
        model5.fit(X_trains, y_trains)
        y_pred = model5.predict(X_tests)
        accuracyscore5 = accuracy_score(y_tests, y_pred)
        # accuracy_score = model.score(X_trains,y_trains)
        print(accuracyscore5)
        abc8 = accuracyscore5
        accuracy.append(abc8)


    if (selectedalge == 6):
        model6 = SVC(kernel='linear')
        model6.fit(X_trains, y_trains)
        y_pred = model6.predict(X_tests)
        accuracyscore6 = accuracy_score(y_tests, y_pred)
        # accuracy_score = model.score(X_trains,y_trains)
        print(accuracyscore6)
        abc9 = accuracyscore6
        accuracy.append(abc9)

        return render_template('graph.html', msg="accuracy_score",
                               score1=accuracyscore1,
                               score2=accuracyscore2,
                               score3=accuracyscore3,
                               score4=accuracyscore4,
                               score5=accuracyscore5,
                               score6=accuracyscore6,
                               model1="LogisticRegression",
                               model2="DecisionTree",
                               model3="RandomForest",
                               model4="NaiveBayes",
                               model5="KNearestNeighbors",
                               model6="SupportVectorMachine")


      # return render_template('modelperformance.html', msg="accuracy_score",
                               #)

       # return render_template('modelperformance.html', msg="accuracy_score",
                              # )
        #return render_template('modelperformance.html', msg="accuracy_score",
                             #  )
       # return render_template('modelperformance.html', msg="accuracy_score",
                              # )

        #return render_template('modelperformance.html', msg="accuracy_score",
                              # )


    #return render_template('modelperformance.html')





if __name__ == '__main__':
    app.secret_key = ".."
    app.run()