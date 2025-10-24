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

"""import random
from bokeh.models import (HoverTool, FactorRange, Plot, LinearAxis, Grid,Range1d)
from bokeh.models.glyphs import VBar
from bokeh.plotting import figure
from bokeh.charts import Bar
from bokeh.embed import components
from bokeh.models.sources import ColumnDataSource
from flask import Flask, render_template"""

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

    """if x_train == 1:
        print("b is greater than a")
        print(X_se)
    else:
        X_se = pd.DataFrame(x_train)
        print("HELLO ++++++++++++++++++++++++++++++++++++++++++++++WORLD")
        print(X_se)"""

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

"""@app.route('/bar_charts')
def bar():
    bar_labels=labels
    bar_values=values
    return render_template('bar_chart.html', title='Bitcoin Monthly Price in USD', max=17000, labels=bar_labels, values=bar_values)"""


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
        # print(X)
        # print(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=value1)
        # print(X_train)
        # print(X_test)
        # print(y_train)
        # print(y_test)
        f(X_train, X_test, y_train, y_test)
        print("world------------------------------------------------------HELLO")
        print(X_trains)

        # X_se=pd.DataFrame(X_train)

        # X_se= X_train
        # print(X_se)
        # session['X_se'] = X_se.to_json()
        # print(X_se.to_json())

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
        # X_se = session.get('X_se')
        # print(X_se)
        # x_train=1
        # f(x_train)
        print(
            "*********************************************************************************************************")
        print(X_trains)
        print(
            "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
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
            print(accuracyscore)

            # pred1 = model.predict(X_tests)
            # accuracy = accuracy_score(pred1,y_test)
            # final_accuracy = accuracy * 100
            # print(f"RFC {accuracy}")
            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore,
                                   model="DecisionTree")

        if (selectedalg == 2):
            model = linear_model.LogisticRegression()
            model.fit(X_trains, y_trains)
            y_pred = model.predict(X_tests)
            accuracyscore = accuracy_score(y_tests, y_pred)
            # accuracy_score = model.score(X_trains,y_trains)
            print(accuracyscore)
            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore,
                                   model="LogisticRegression")

        if (selectedalg == 3):
            model = RandomForestClassifier(n_estimators=20)
            model.fit(X_trains, y_trains)
            y_pred = model.predict(X_tests)
            accuracyscore = accuracy_score(y_tests, y_pred)
            # accuracy_score = model.score(X_trains,y_trains)
            print(accuracyscore)
            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore,
                                   model="RandomForest")

        if (selectedalg == 4):
            model = GaussianNB()
            model.fit(X_trains, y_trains)
            y_pred = model.predict(X_tests)
            accuracyscore = accuracy_score(y_tests, y_pred)
            # accuracy_score = model.score(X_trains,y_trains)
            print(accuracyscore)
            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore,
                                   model="NaiveBayes")

        if (selectedalg == 5):
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(X_trains, y_trains)
            y_pred = model.predict(X_tests)
            accuracyscore = accuracy_score(y_tests, y_pred)
            # accuracy_score = model.score(X_trains,y_trains)
            print(accuracyscore)
            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore,
                                   model="KNearestNeighbors")

        if (selectedalg == 6):
            model = SVC(kernel='linear')
            model.fit(X_trains, y_trains)
            y_pred = model.predict(X_tests)
            accuracyscore = accuracy_score(y_tests, y_pred)
            # accuracy_score = model.score(X_trains,y_trains)
            print(accuracyscore)
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


"""def create_hover_tool():
    pass


def create_bar_chart(data, title, x_name, y_name, hover_tool=None, width=1200, height=300):

    source = ColumnDataSource(data)
    xdr = FactorRange(factors=data[x_name])
    ydr = Range1d(start=0, end=max(data[y_name]) * 1.5)

    tools = []
    if hover_tool:
        tools = [hover_tool, ]

    plot = figure(title=title, x_range=xdr, y_range=ydr, plot_width=width,
                  plot_height=height, h_symmetry=False, v_symmetry=False,
                  min_border=10, toolbar_location="above", tools=tools,
                  responsive=True, outline_line_color="#666666")

    glyph = VBar(x=x_name, top=y_name, bottom=0, width=.8,
                 fill_color="#6599ed")
    plot.add_glyph(source, glyph)

    xaxis = LinearAxis()
    yaxis = LinearAxis()

    plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
    plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))
    plot.toolbar.logo = None
    plot.min_border_top = 0
    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = "#999999"
    plot.yaxis.axis_label = "Bugs found"
    plot.ygrid.grid_line_alpha = 0.1
    plot.xaxis.axis_label = "Days after app deployment"
    plot.xaxis.major_label_orientation = 1
    return plot


@app.route('/chart',methods=["POST","GET"])
def chart(num_bars):

    if num_bars <= 0:
        num_bars = 1
    data = {"days": [], "bugs": [], "costs": []}
    for i in range(1, num_bars + 1):
        data['days'].append(i)
        data['bugs'].append(random.randint(1,100))
        data['costs'].append(random.uniform(1.00, 1000.00))

    hover = create_hover_tool()
    plot = create_bar_chart(data, "Bugs found per day", "days",
                            "bugs", hover)
    script, div = components(plot)
    script, div = components(plot)
    #return render_template('bar_chart.html', bars_count=num_bars,the_div=div, the_script=script)"""
@app.route('/graph')
def graph():


    return render_template('graph.html', )

if __name__ == '__main__':
    app.secret_key = ".."
    app.run()