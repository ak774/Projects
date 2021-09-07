import os
import shutil

from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from flask import Flask, session
import pickle
import pandas as pd
import numpy as np
basedir = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join('static', 'csv')

app = Flask(__name__)
model = pickle.load(open('tree_fin.pickle', 'rb'))
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['csv','xls'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('index.html')

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


@app.route('/', methods=['POST'])
def upload_file():

    # shutil.rmtree(UPLOAD_FOLDER)
    # os.mkdir(UPLOAD_FOLDER)
    disp_div = 'none'
    disp_div_tumor = 'none'

    d = request.form.to_dict()
    # print("dddd;",d)
    button_name = 'None'
    if (len(d)!=0):
        button_name = list(d.items())[-1][0]

    file = request.files['file']
    print("file:",file)
    if file.filename == '':
        flash('No file selected for uploading','red')
        # return redirect(request.url)
        return render_template('index.html', disp_div = disp_div)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        shutil.rmtree(UPLOAD_FOLDER)
        os.mkdir(UPLOAD_FOLDER)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('File successfully uploaded!', 'green')
        print(UPLOAD_FOLDER)
        print("==>",os.path.join(UPLOAD_FOLDER, sorted(os.listdir(app.config['UPLOAD_FOLDER']))[0]))
        csv_file = pd.read_csv(os.path.join(UPLOAD_FOLDER, sorted(os.listdir(app.config['UPLOAD_FOLDER']))[0]))
        '''Data Processing'''
        csv_shape = csv_file
        csv_shape.drop('PassengerId', axis=1, inplace=True)
        csv_shape['Age'] = csv_shape['Age'].fillna(csv_shape.Age.median())
        csv_shape['Cabin'] = csv_shape['Cabin'].replace(np.NaN, 'M')
        csv_shape.dropna(inplace=True)
        csv_shape['has_sibling'] = csv_shape['SibSp'].apply(lambda x: 1 if x > 0 else 0)
        csv_shape['has_child'] = csv_shape['Parch'].apply(lambda x: 1 if x > 0 else 0)
        csv_shape['is_not_cheap'] = csv_shape['Fare'].apply(lambda x: 1 if x > 26 else 0)
        csv_shape['Cabin'] = csv_shape['Cabin'].apply(lambda x: x[0])
        csv_shape = pd.concat([csv_shape, pd.get_dummies(csv_shape['Sex'], drop_first=True)], axis=1)
        csv_shape.drop(['Sex'], axis=1, inplace=True)
        csv_shape.drop(['Ticket'], axis=1, inplace=True)
        csv_shape['Title'] = (csv_shape['Name'].apply(lambda x: x.split(" ")[1])).apply(lambda x: x.split(".")[0])
        csv_shape['is_mr']=csv_shape['Title'].apply(lambda x:1 if x=='Mr' else 0)
        csv_shape.drop(['Name'],axis=1,inplace=True)
        csv_shape['is_missing'] = csv_shape['Cabin'].apply(lambda x: 0 if x == 'M' else 0)
        csv_shape.drop(['Cabin'], axis=1, inplace=True)
        csv_shape = pd.concat([csv_shape, pd.get_dummies(csv_shape['Embarked'], drop_first=True)], axis=1)
        csv_shape.drop(['Title'], axis=1, inplace=True)
        csv_shape.drop(['Embarked'], axis=1, inplace=True)
        pred=model.predict(csv_shape)
        return render_template('index.html', csv_shape=pd.DataFrame(pred).to_html())



        # return redirect('/')
    else:
        flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif', 'red')
        # return redirect(request.url)
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)

