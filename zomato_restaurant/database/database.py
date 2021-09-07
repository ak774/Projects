import sqlite3
from flask import Flask, flash, request, redirect, render_template
import logging
import pandas as pd
import numpy as np

class Database:
    def database(self,df):
        '''Method for inserting input data into database.
        input:DataFrame
        output:Data frame from database.
        '''

        try:
            cnx = sqlite3.connect('test.db')
            df.to_sql(name='zomato', con=cnx)
            df_database = pd.read_sql('select * from zomato', cnx)
            logging.basicConfig(level=logging.DEBUG,
                                format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                                datefmt='%m-%d %H:%M',
                                filename='./logs/log',
                                filemode='w')
            logging.info('The data is successfully inserted in Database.')
            return df_database
        except ValueError:
            flash("  database Error Occurred! %s" % ValueError)
            return render_template('index.html')
        except KeyError:
            flash("Error Occurred! %s" % KeyError)
            return render_template('index.html')
        except Exception as e:
            flash("Error Occurred! %s" % e)
            return render_template('index.html')