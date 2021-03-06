import os
import shutil

from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from flask import Flask, session
import pickle
import pandas as pd
import numpy as np
from preprocessing.preprocessing import Preprocessing
from clustering.clustering import Clustering
from prediction.prediction import Prediction
from database.database import Database
import logging
basedir = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join('static', 'csv')

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

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

    try:
        '''Method for controlling the flow of application.
        input:DataFrame
        output:Ratings in form of html response.
                                                        '''
        # shutil.rmtree(UPLOAD_FOLDER)
        # os.mkdir(UPLOAD_FOLDER)
        disp_div = 'none'
        disp_div_tumor = 'none'

        d = request.form.to_dict()
        # print("dddd;",d)
        button_name = 'None'
        if (len(d) != 0):
            button_name = list(d.items())[-1][0]

        file = request.files['file']
        print("file:", file)
        if file.filename == '':
            flash('No file selected for uploading', 'red')
            # return redirect(request.url)
            return render_template('index.html', disp_div=disp_div)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            shutil.rmtree(UPLOAD_FOLDER)
            os.mkdir(UPLOAD_FOLDER)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File successfully uploaded!', 'green')
            print(UPLOAD_FOLDER)
            print("==>", os.path.join(UPLOAD_FOLDER, sorted(os.listdir(app.config['UPLOAD_FOLDER']))[0]))
            csv_file = pd.read_csv(os.path.join(UPLOAD_FOLDER, sorted(os.listdir(app.config['UPLOAD_FOLDER']))[0]))
            '''Data Processing'''
            zomato = csv_file
            logging.basicConfig(level=logging.DEBUG,
                                format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                                datefmt='%m-%d %H:%M',
                                filename='./logs/log',
                                filemode='w')
            logging.info('The data is successfully inserted.')
            obj=Database()
            zomato=obj.database(zomato)
            obj=Preprocessing()
            df,names=obj.process(zomato)
            scaled=obj.transform(df)
            obj1=Clustering()
            cluster=obj1.cluster(scaled)
            obj2=Prediction()
            pred_final=obj2.predict(scaled,cluster,names)

            logging.info('The data is successfully displayed.')


            return render_template('index.html', csv_shape=pred_final.to_html())

    # return redirect('/')
    except ValueError:
        flash("final Error Occurred! %s" % ValueError)
        return render_template('index.html')
    except KeyError:
        flash("Error Occurred! %s" % KeyError)
        return render_template('index.html')
    except Exception as e:
        flash("Error Occurred! %s" % e)
        return render_template('index.html')




if __name__ == '__main__':
    app.run(debug=True, port=5000)

