import pickle
import numpy as np
import pandas as pd
from flask import Flask, flash, request, redirect, render_template
import math
import logging
class Prediction:
    
    def predict(self,df,model_cluster,names):
        '''Method for predicting rating based on input dataframe and clusters.
        input:DataFrame,clusters,name of restaurants
        output:Predicted TRating for restaurants
                                                        '''
        try:
            model_scale_y = pickle.load(open('./models/min_max_y.pickle', 'rb'))
            model_0 = pickle.load(open('./models/random_0.pickle', 'rb'))
            model_1 = pickle.load(open('./models/random_1.pickle', 'rb'))
            model_2 = pickle.load(open('./models/random_2.pickle', 'rb'))
            model_3 = pickle.load(open('./models/random_3.pickle', 'rb'))
            model_4 = pickle.load(open('./models/random_4.pickle', 'rb'))
            model_5 = pickle.load(open('./models/random_5.pickle', 'rb'))
            zomato_scaled=df
            zomato_cluster = model_cluster
            cluster = pd.DataFrame(data=zomato_cluster, columns=["Cluster"])
            final_data = pd.DataFrame(data=zomato_scaled, columns=[i for i in range(np.shape(zomato_scaled)[1])])
            final_data = pd.concat([final_data, cluster], axis=1)
            data_0 = final_data[final_data['Cluster'] == 0]
            data_1 = final_data[final_data['Cluster'] == 1]
            data_2 = final_data[final_data['Cluster'] == 2]
            data_3 = final_data[final_data['Cluster'] == 3]
            data_4 = final_data[final_data['Cluster'] == 4]
            data_5 = final_data[final_data['Cluster'] == 5]
            pred_0 = model_0.predict(data_0.drop(['Cluster'], axis=1))
            pred_1 = model_1.predict(data_1.drop(['Cluster'], axis=1))
            pred_2 = model_2.predict(data_2.drop(['Cluster'], axis=1))
            pred_3 = model_3.predict(data_3.drop(['Cluster'], axis=1))
            pred_4 = model_4.predict(data_4.drop(['Cluster'], axis=1))
            pred_5 = model_5.predict(data_5.drop(['Cluster'], axis=1))
            l = [0 for i in range(len(cluster))]
            j = 0
            for i in cluster[cluster['Cluster'] == 0].index:
                l[i] = pred_0[j]
                j += 1
            j = 0
            for i in cluster[cluster['Cluster'] == 1].index:
                l[i] = pred_1[j]
                j += 1
            j = 0
            for i in cluster[cluster['Cluster'] == 2].index:
                l[i] = pred_2[j]
                j += 1
            j = 0
            for i in cluster[cluster['Cluster'] == 3].index:
                l[i] = pred_3[j]
                j += 1
            j = 0
            for i in cluster[cluster['Cluster'] == 4].index:
                l[i] = pred_4[j]
                j += 1
            j = 0
            for i in cluster[cluster['Cluster'] == 5].index:
                l[i] = pred_5[j]
                j += 1
            l = np.array(l)
            l_scaled = model_scale_y.inverse_transform(l.reshape(-1, 1))
            for i in range(len(l_scaled)):
                l_scaled[i] = math.pow(l_scaled[i], 2)
            l_scaled=np.round(l_scaled,1)
            pred = pd.DataFrame(data=l_scaled, columns=['Prediction'])
            names = names.to_frame()
            pred.set_index(names.index, inplace=True)
            pred = pd.concat([names, pred], axis=1)
            pred_final = pred.groupby('name').mean()
            logging.basicConfig(level=logging.DEBUG,
                                format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                                datefmt='%m-%d %H:%M',
                                filename='./logs/log',
                                filemode='w')
            logging.info('The data is successfully predicted.')
            return pred_final
        except ValueError:
            flash("Error Occurred! %s" % ValueError)
            return render_template('index.html')
        except KeyError:
            flash("Error Occurred! %s" % KeyError)
            return render_template('index.html')
        except Exception as e:
            flash("Error Occurred! %s" % e)
            return render_template('index.html')