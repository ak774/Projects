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
<<<<<<< HEAD
model_scale=pickle.load(open('ml_scale.pickle','rb'))
model = pickle.load(open('ml_model.pickle', 'rb'))
model_descale=pickle.load(open('ml_scale_y.pickle','rb'))
=======
model = pickle.load(open('logistic_final.pickle', 'rb'))
model_scale = pickle.load(open('log_scale.pickle','rb'))
>>>>>>> 53dee8a6a609ef1d813afee543418c7a8a7904c8
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
<<<<<<< HEAD
        '''Data Processing'''
        df = csv_file
        num_var = [feat for feat in df.columns if df[feat].dtype != 'O']
        cat_var = [feat for feat in df.columns if feat not in num_var]
        df.drop(['id', 'player_fifa_api_id', 'player_api_id', 'date'], axis=1, inplace=True)
        feat_null = []
        for feat in num_var:
            if feat in df.columns and df[feat].isnull().sum() < 1000:
                feat_null.append(feat)
        for feat in feat_null:
            df[feat] = df[feat].fillna(df[feat].mean())
        df.dropna(inplace=True)
        num_var_new = [feat for feat in df.columns if df[feat].dtype != 'O']
        df = pd.concat([df.drop(['preferred_foot'], axis=1), pd.get_dummies(df['preferred_foot'], drop_first=True)],
                       axis=1)
        d = {'medium': 2, 'high': 3, 'low': 1, 'None': 0, 'le': 0, 'norm': 0, 'stoc': 0, 'y': 0}
        df['attacking_work_rate'] = df['attacking_work_rate'].map(d)
        d = {'medium': 2, 'high': 3, 'low': 1, '5': 0, 'ean': 0, 'o': 0, '1': 0, 'ormal': 0, '7': 0, '2': 0,
             '8': 0, '4': 0, 'tocky': 0, '0': 0, '3': 0, '6': 0, '9': 0, 'es': 0}
        df['defensive_work_rate'] = df['defensive_work_rate'].map(d)
        for feat in df.columns:
            df[feat] = np.sqrt(df[feat])
        X = df.drop(['overall_rating'], axis=1)
        scale=model_scale.transform(X)
        pred=model.predict(scale)
        descaled = model_descale.inverse_transform(pred)
        op=np.power(descaled,2)
        return render_template('index.html', csv_shape=pd.DataFrame(op).to_html())
=======
        '''Data Processsing'''
        csv_shape = csv_file
        csv_shape['married_long'] = np.where(csv_shape.yrs_married > 6, 1, 0)
        csv_shape['young'] = np.where(csv_shape.age < 32, 1, 0)
        val_rate = csv_shape['rate_marriage'].unique()
        for val in val_rate:
            csv_shape['rate_' + str(val)] = np.where(csv_shape['rate_marriage'] == val, 1, 0)
        csv_shape.drop('rate_marriage', axis=1, inplace=True)
        scaled=model_scale.transform(csv_shape)
        pred=model.predict(scaled)
        return render_template('index.html', csv_shape=pd.DataFrame(pred).to_html())
>>>>>>> 53dee8a6a609ef1d813afee543418c7a8a7904c8



        # return redirect('/')
    else:
        flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif', 'red')
        # return redirect(request.url)
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)

