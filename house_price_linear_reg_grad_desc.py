#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 13:54:16 2022

@author: ustokowska
"""
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import avg, sum, col,lit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
import plotly.express as px
import datetime

def create_session_object():
   connection_parameters = {
      "account": os.getenv("ACCOUNT"),
      "user": os.getenv("USER_SNOW"),
      "password": os.getenv("PASSWORD"),
      "role": os.getenv("ROLE"),
      "warehouse": os.getenv("WAREHOUSE"),
      "database": "HOUSING",
      "schema": "HOUSING"
   }
   session = Session.builder.configs(connection_parameters).create()
   #print(session.sql('select current_warehouse(), current_database(), current_schema()').collect())
   return session

#create snowpark session
my_session = create_session_object()

#read raw house sales data
df_raw = my_session.table("HOUSE_PRICE_PREDICTION")
df_raw_pandas = df_raw.toPandas()
#st.write(df_raw_pandas.describe())

#feature engineering to extract zipcode from the statezip column
df_raw_pandas['STATEZIP'] = df_raw_pandas['STATEZIP'].str[-5:]
#st.write(df_raw_pandas)

#url to zipcode per lat lon dataset
url_zips = 'https://raw.githubusercontent.com/sfc-gh-ustokowska/streamlit/main/zipcodes.csv'
#dataframe with zipcode per lat lon dataset
df_zips = pd.read_csv(url_zips, index_col = 0)

#convert the type of statezip column from string to int
df_raw_pandas= df_raw_pandas.astype({'STATEZIP':'int'})
df_raw_pandas_zips = pd.merge(df_raw_pandas, df_zips, how='left', left_on = 'STATEZIP', right_on = 'Zipcode')
df_raw_pandas_zips_no_na = df_raw_pandas_zips.dropna()

#linear regression
reg = LinearRegression()

#feature to predict
labels = df_raw_pandas_zips_no_na['PRICE']
#feature engineering to convert date values to 1 if date is before year 2000, 0 if after
conv_dates = [1 if values >= datetime.date(2000,1,1) else 0 for values in df_raw_pandas_zips_no_na['RECORD_DATE']]
df_raw_pandas_zips_no_na['RECORD_DATE'] = conv_dates
#feature engineering to remove unwanted features from the dataset 
train1 = df_raw_pandas_zips_no_na.drop(['PRICE','STREET','CITY','City','COUNTRY','State','StateCode','Zipcode'],axis=1)

#prepare train and test dataset (90% of the dataset goes for training, 10% for testing)
x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size = 0.1, random_state = 2)

#linear regression
reg.fit(x_train,y_train)
#st.write('Linear regression score: ','{:.2f}'.format(reg.score(x_test,y_test)))

#gradient descent
clf = ensemble.GradientBoostingRegressor(n_estimators = 100, max_depth = 5, min_samples_split = 2, learning_rate = 0.1, loss = 'ls')
clf.fit(x_train,y_train)
#st.write('Gradient descent score: ','{:.2f}'.format(clf.score(x_test,y_test)))

#y_test = np.exp(y_test) # Convert back to actual (non-log) values
#y_pred = np.exp(clf.predict(x_test)) # Convert back to actual (non-log) values
#print(clf.predict(x_test))
#print(reg.predict(x_test))

#test_df = x_test.iloc[105:106]
#st.write(test_df)
#st.write(test_df.index[0])
#df_sample = df_raw_pandas_zips_no_na.filter(items=[test_df.index[0],'PRICE'], axis=0)
#st.write('Actual price: ',df_sample.filter(items=['PRICE']))
#st.write('Actual price: ','${:,.2f}'.format(df_sample['PRICE'].squeeze()))
#st.write('Predicted price: ','${:,.2f}'.format(round(clf.predict(test_df).item(),2)))
#diff = abs(df_sample['PRICE'].squeeze() - round(clf.predict(test_df).item(),2))
#st.write('Difference between actual vs predicted price: ','${:,.2f}'.format(diff))

#UI
with st.form("my_form"):
   #submit function to run necessary calculations (lat, lon, record date)
   def submit_function():
      record_date = record_date_function()
      lat, lon = calc_lat_lon()
      waterfront_score = waterfront_feat_eng()
      attributes_list = [[record_date,bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront_score,view_score,condition_score,sqft_above,sqft_basement,yr_built,yr_renovated,zipcode,lat,lon]]
      attributes_df = pd.DataFrame(attributes_list,columns=['RECORD_DATE','BEDROOMS','BATHROOMS','SQFT_LIVING','SQFT_LOT','FLOORS','WATERFRONT','VIEW_SCORE','CONDITION_SCORE','SQFT_ABOVE','SQFT_BASEMENT','YR_BUILT','YR_RENOVATED','STATEZIP','lat','lon'])
      #print(attributes_df.dtypes)
      st.write('Predicted price: ','${:,.2f}'.format(round(clf.predict(attributes_df).item(),2)))
      return attributes_df
   #function to calculate if the building was built before or after year 2000
   def record_date_function():
      if yr_built >= 200:
         record_date = 1
      else:
         record_date = 0
      return record_date
   #function to calculate lat and lon based on zipcode
   def calc_lat_lon():
      df_with_proper_record = df_zips[df_zips.Zipcode == zipcode]
      lat = df_with_proper_record['lat']
      lon = df_with_proper_record['lon']
      return lat.squeeze(), lon.squeeze()
   #function to convert waterfront yes/no into 1/0
   def waterfront_feat_eng():
      if waterfront == 'Yes':
         waterfront_score = 1
      else:
         waterfront_score = 0
      return waterfront_score
   st.header('Information about the property')
   bedrooms = st.number_input('Number of bedrooms',value=0)
   bathrooms = st.number_input('Number of bathrooms',value=0)
   sqft_living = st.number_input('Living area in SQFT')
   sqft_lot = st.number_input('Lot area in SQFT')
   floors = st.number_input('Number of floors',value=0)
   sqft_above = st.number_input('SQFT above sea level')
   sqft_basement = st.number_input('Basement area in SQFT')
   yr_built = st.number_input('Built year',value=0,max_value=2022)
   yr_renovated = st.number_input('Year of last renovation',value=0,max_value=2022)
   zipcode = st.number_input('Zipcode',value=0)
   waterfront = st.selectbox(
      'Does the property have a waterfront?',
      ('Yes', 'No'))
   view_score = st.selectbox(
      "What is the property's view score?",
      (1,2,3,4,5))
   condition_score = st.selectbox(
      "What is the property's condition score?",
      (1,2,3,4,5))
   submitted = st.form_submit_button('Submit')
   if submitted:
      #st.write(submit_function())
      submit_function()