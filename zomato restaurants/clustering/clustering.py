import pickle
from flask import Flask, flash, request, redirect, render_template
import logging
class Clustering:
    def cluster(self,cluster):
                '''Method for segmenting input data data into clusters.
        input:scaled DataFrame
        output:predicted clusters for each row of data frame
                                                        '''
        try:
            directory = './models/cluster.pickle'
            zomato_scaled = cluster
            model_cluster = pickle.load(open(directory, 'rb'))
            zomato_cluster = model_cluster.predict(zomato_scaled)
            logging.basicConfig(level=logging.DEBUG,
                                format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                                datefmt='%m-%d %H:%M',
                                filename='./logs/log',
                                filemode='w')
            logging.info('The data is successfully clustered.')
            return zomato_cluster
        except ValueError:
            flash("Error Occurred! %s" % ValueError)
            return render_template('index.html')
        except KeyError:
            flash("Error Occurred! %s" % KeyError)
            return render_template('index.html')
        except Exception as e:
            flash("Error Occurred! %s" % e)
            return render_template('index.html')