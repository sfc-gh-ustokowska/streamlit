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
import pydeck as pdk

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

my_session = create_session_object()

#LOCATION MAP COMPONENT
url = 'https://raw.githubusercontent.com/sfc-gh-ustokowska/streamlit/main/1.csv'
url2 = 'https://raw.githubusercontent.com/sfc-gh-ustokowska/streamlit/main/2.csv'
url3 = 'https://raw.githubusercontent.com/sfc-gh-ustokowska/streamlit/main/3.csv'
url_zips = 'https://raw.githubusercontent.com/sfc-gh-ustokowska/streamlit/main/zipcodes.csv'

df1 = pd.read_csv(url, index_col = 1)
df2 = pd.read_csv(url2, index_col = 1)
df3 = pd.read_csv(url3, index_col = 1)

df = pd.concat([df1, df2, df3])
df['mean'] = df.mean(axis=1)

df_zips = pd.read_csv(url_zips, index_col = 0)

df_mean_per_zip = df[['RegionName','mean']]

df_full = pd.merge(df_mean_per_zip, df_zips, how='left', left_on = 'RegionName', right_on = 'Zipcode')
df_full_no_na = df_full.dropna()

#read raw house sales data
df_raw = my_session.table("HOUSE_PRICE_PREDICTION")
df_raw_pandas = df_raw.toPandas()

#feature engineering to extract zipcode from the statezip column
df_raw_pandas['STATEZIP'] = df_raw_pandas['STATEZIP'].str[-5:]

#convert the type of statezip column from string to int
df_raw_pandas= df_raw_pandas.astype({'STATEZIP':'int'})

#merge zipcode per lat lon dataframe with raw house sales dataframe
df_raw_pandas_zips = pd.merge(df_raw_pandas, df_zips, how='left', left_on = 'STATEZIP', right_on = 'Zipcode')
#remove NA values from the dataet
df_raw_pandas_zips_no_na = df_raw_pandas_zips.dropna()


midpoint = (np.average(df_full_no_na['lat']), np.average(df_full_no_na['lon']))
st.pydeck_chart(pdk.Deck(
			map_style='mapbox://styles/mapbox/outdoors-v11',
			initial_view_state=pdk.ViewState(
			latitude=midpoint[0],
			longitude=midpoint[1],
			zoom=3),
			layers=[pdk.Layer(
					'ScatterplotLayer',
					data=df_full_no_na[df_full_no_na['mean'] < 100000],
					get_position=['lon', 'lat'],
					get_color='[114, 223, 121, 160]',
					radius_min_pixels=4,
	    			radius_max_pixels=15,
					),
					pdk.Layer(
					'ScatterplotLayer',
					data=df_full_no_na[(df_full_no_na['mean'] >= 100000) & (df_full_no_na['mean'] < 200000)],
					get_position=['lon', 'lat'],
					get_color='[0, 153, 0, 160]',
					radius_min_pixels=4,
	    			radius_max_pixels=15,
					),
					pdk.Layer(
					'ScatterplotLayer',
					data=df_full_no_na[(df_full_no_na['mean'] >= 500000) & (df_full_no_na['mean'] < 1000000)],
					get_position=['lon', 'lat'],
					get_color='[240, 202, 9, 160]',
					radius_min_pixels=4,
	    			radius_max_pixels=15,
					),
					pdk.Layer(
					'ScatterplotLayer',
				    data=df_full_no_na[df_full_no_na['mean'] >= 1000000],
					get_position=['lon', 'lat'],
					get_color='[255, 0, 0, 160]',
					radius_min_pixels=4,
	    			radius_max_pixels=15,
					)
				]
		))

#VISUALIZATION
col1, col2 = st.columns([6,6])

with col1:
   #price per bedrooms
   st.write("House price per number of bedrooms")
   df_price_bed = df_raw_pandas[['PRICE','BEDROOMS']]
   fig = px.scatter(
      x=df_price_bed['BEDROOMS'],
      y=df_price_bed['PRICE'],
   )
   fig.update_layout(
      xaxis_title="Bedrooms",
      yaxis_title="Price",
      yaxis_range=[-4,8000000]
   )
   st.plotly_chart(fig.update_traces(marker=dict(color='#ffb90f')), use_container_width=True)
   #st.write(fig)

   #price vs condition
   st.write("Price vs house condition")
   df_price_cnd = df_raw_pandas[['PRICE','CONDITION_SCORE']]
   fig = px.scatter(
      x=df_price_cnd['CONDITION_SCORE'],
      y=df_price_cnd['PRICE']
   )
   fig.update_layout(
      xaxis_title="Condition score",
      yaxis_title="Price",
      yaxis_range=[-4,8000000]
   )
   st.plotly_chart(fig.update_traces(marker=dict(color='#ffb90f')), use_container_width=True)

with col2:
   #price per sqft
   st.write("House price per sq feet")
   df_price_sqfeet = df_raw_pandas[['SQFT_LIVING','PRICE']]
   fig = px.scatter(
      x=df_price_sqfeet['PRICE'],
      y=df_price_sqfeet['SQFT_LIVING'],
   )
   fig.update_layout(
      xaxis_title="Price",
      yaxis_title="Sqft living",
      xaxis_range=[-4,5000000]
   )
   st.plotly_chart(fig.update_traces(marker=dict(color='#ffb90f')), use_container_width=True)
   
   #price vs floors
   st.write("Price vs floors")
   df_price_floors = df_raw_pandas[['PRICE','FLOORS']]
   fig = px.scatter(
      x=df_price_floors['FLOORS'],
      y=df_price_floors['PRICE'],
   )
   fig.update_layout(
      xaxis_title="Floors",
      yaxis_title="Price",
      yaxis_range=[-4,8000000]
   )
   st.plotly_chart(fig.update_traces(marker=dict(color='#ffb90f')), use_container_width=True)

#PRICE PREDICTION
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
   zipcode = st.number_input('Zipcode',value=94305)
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