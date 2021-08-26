import numpy as np
import pandas as pd
from flask import Flask, flash, request, redirect, render_template
import pickle
import logging
class Preprocessing:

    def process(self,df):
        '''Method for preprocessing input dataframe.
        input:DataFrame
        output:Processed data frame ready for prediction
                                                        '''       
        
        try:
            zomato=df
            zomato.drop(['url', 'phone', 'dish_liked'], axis=1, inplace=True)
            zomato.dropna(how='any', inplace=True)
            zomato['approx_cost(for two people)'] = zomato['approx_cost(for two people)'].astype(str)
            zomato['approx_cost(for two people)'] = zomato['approx_cost(for two people)'].apply(
                lambda x: x.replace(',', '.'))
            zomato['approx_cost(for two people)'] = zomato['approx_cost(for two people)'].astype(float)
            zomato.dropna(inplace=True)
            zomato.drop(['address'], axis=1, inplace=True)
            zomato = pd.concat([zomato, pd.get_dummies(zomato['online_order'], drop_first=True)], axis=1)
            zomato.drop(['online_order'], axis=1, inplace=True)
            df_yes = pd.get_dummies(zomato['book_table'], drop_first=True)
            df_yes.rename(columns={'Yes': 'yes_book_table'}, inplace=True)
            zomato = pd.concat([zomato, df_yes], axis=1)
            zomato.drop(['book_table'], axis=1, inplace=True)
            b = zomato.groupby('listed_in(city)')['votes'].count().sort_values(ascending=False).index

            l = []
            for i in range(21):
                if b[i] in b[:21]:
                    l.append(b[i])
            b = zomato.groupby('name')['votes'].count().sort_values(ascending=False).index
            for i in l:
                zomato[i] = np.where(zomato['location'] == i, 1, 0)
            for i in b[:25]:
                zomato[i] = np.where(zomato['name'] == i, 1, 0)
            zomato.drop(['reviews_list', 'cuisines', 'listed_in(city)'], axis=1, inplace=True)
            zomato.drop(['menu_item'], axis=1, inplace=True)
            zomato = pd.concat(
                [zomato.drop(['listed_in(type)'], axis=1), pd.get_dummies(zomato['listed_in(type)'], drop_first=True)],
                axis=1)
            a = zomato.groupby('rest_type')['votes'].count().sort_values(ascending=False).index
            for i in a[:10]:
                zomato[i] = np.where(zomato['rest_type'] == i, 1, 0)
            zomato.drop(['location', 'rest_type'], axis=1, inplace=True)
            names = zomato['name']
            zomato.drop(['name'], axis=1, inplace=True)
            for col in zomato.columns:
                zomato[col] = np.sqrt(zomato[col])
            logging.basicConfig(level=logging.DEBUG,
                                format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                                datefmt='%m-%d %H:%M',
                                filename='./logs/log',
                                filemode='w')
            logging.info('The data is successfully processed.')
            return zomato,names;
        except ValueError:
            flash("Error Occurred! %s" % ValueError)
            return render_template('index.html')
        except KeyError:
            flash("Error Occurred! %s" % KeyError)
            return render_template('index.html')
        except Exception as e:
            flash("Error Occurred! %s" % e)
            return render_template('index.html')

    def transform(self,df)
        '''Method for Scaling input dataframe using min_max scaler.
        input: processedDataFrame
        output:Scaled data frame ready for prediction
                                                        '''
    
    
        try:
            model_scale = pickle.load(open('./models/min_max.pickle', 'rb'))

            zomato_scaled=model_scale.transform(df)
            logging.basicConfig(level=logging.DEBUG,
                                format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                                datefmt='%m-%d %H:%M',
                                filename='./logs/log',
                                filemode='w')
            logging.info('The data is successfully transformed.')
            return zomato_scaled
        except ValueError:
            flash("Error Occurred! %s" % ValueError)
            return render_template('index.html')
        except KeyError:
            flash("Error Occurred! %s" % KeyError)
            return render_template('index.html')
        except Exception as e:
            flash("Error Occurred! %s" % e)
            return render_template('index.html')