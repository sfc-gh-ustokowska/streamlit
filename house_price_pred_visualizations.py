#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 13:54:16 2022

@author: ustokowska
"""
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import avg, sum, col,lit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
import plotly.express as px


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
   print(session.sql('select current_warehouse(), current_database(), current_schema()').collect())
   return session

#create snowpark session
my_session = create_session_object()

#read raw house sales data
df_raw = my_session.table("HOUSE_PRICE_PREDICTION")
df_raw_pandas = df_raw.toPandas()
st.write(df_raw_pandas.describe())

#feature engineering to extract zipcode from the statezip column
df_raw_pandas['STATEZIP'] = df_raw_pandas['STATEZIP'].str[-5:]
st.write(df_raw_pandas)

#url to zipcode per lat lon dataset
url_zips = 'https://raw.githubusercontent.com/sfc-gh-ustokowska/streamlit/main/zipcodes.csv'
#dataframe with zipcode per lat lon dataset
df_zips = pd.read_csv(url_zips, index_col = 0)

#convert the type of statezip column from string to int
df_raw_pandas= df_raw_pandas.astype({'STATEZIP':'int'})

#merge zipcode per lat lon dataframe with raw house sales dataframe
df_raw_pandas_zips = pd.merge(df_raw_pandas, df_zips, how='left', left_on = 'STATEZIP', right_on = 'Zipcode')
#remove NA values from the dataet
df_raw_pandas_zips_no_na = df_raw_pandas_zips.dropna()

#count of houses per number of bedrooms
st.bar_chart(df_raw_pandas['BEDROOMS'].value_counts())

#price per sq feet
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
st.write(fig)

#price per lon
df_price_lon = df_raw_pandas_zips_no_na[['PRICE','lon']]
fig = px.scatter(
   x=df_price_lon['PRICE'],
   y=df_price_lon['lon'],
)
fig.update_layout(
   xaxis_title="Price",
   yaxis_title="Longitude",
   xaxis_range=[-4,15000000]
)
st.write(fig)

#price per lat
df_price_lat = df_raw_pandas_zips_no_na[['PRICE','lat']]
fig = px.scatter(
   x=df_price_lat['PRICE'],
   y=df_price_lat['lat'],
)
fig.update_layout(
   xaxis_title="Price",
   yaxis_title="Latitude",
  # xaxis_range=[-4,15000000]
)
st.write(fig)

#price per bedrooms
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
st.write(fig)

#price per sqftliving + sqftbasement
df_price_sqft_total = df_raw_pandas[['PRICE','SQFT_LIVING','SQFT_BASEMENT']]
df_price_sqft_total['TOTAL_SQFT'] = df_raw_pandas['SQFT_LIVING'] + df_raw_pandas['SQFT_BASEMENT']
fig = px.scatter(
   x=df_price_sqft_total['TOTAL_SQFT'],
   y=df_price_sqft_total['PRICE'],
)
fig.update_layout(
   xaxis_title="TOTAL_SQFT",
   yaxis_title="Price",
   yaxis_range=[-4,10000000]
)
st.write(fig)

#price vs waterfront
df_price_wtfr = df_raw_pandas[['PRICE','WATERFRONT']]
fig = px.scatter(
   x=df_price_wtfr['WATERFRONT'],
   y=df_price_wtfr['PRICE'],
)
fig.update_layout(
   xaxis_title="WATERFRONT",
   yaxis_title="Price",
   yaxis_range=[-4,15000000]
)
st.write(fig)

#price vs floors
df_price_floors = df_raw_pandas[['PRICE','FLOORS']]
fig = px.scatter(
   x=df_price_floors['FLOORS'],
   y=df_price_floors['PRICE'],
)
fig.update_layout(
   xaxis_title="FLOORS",
   yaxis_title="Price",
   yaxis_range=[-4,8000000]
)
st.write(fig)

#price vs condition
df_price_cnd = df_raw_pandas[['PRICE','CONDITION_SCORE']]
fig = px.scatter(
   x=df_price_cnd['CONDITION_SCORE'],
   y=df_price_cnd['PRICE'],
)
fig.update_layout(
   xaxis_title="CONDITION_SCORE",
   yaxis_title="Price",
   yaxis_range=[-4,8000000]
)
st.write(fig)

#price per zip
df_price_zip = df_raw_pandas_zips_no_na[['PRICE','STATEZIP']]
fig = px.scatter(
   x=df_price_zip['STATEZIP'],
   y=df_price_zip['PRICE'],
)
fig.update_layout(
   xaxis_title="STATEZIP",
   yaxis_title="PRICE",
  # xaxis_range=[-4,15000000]
)
st.write(fig)

#linear regression
reg = LinearRegression()

labels = df_raw_pandas_zips_no_na['PRICE']
conv_dates = [1 if values== 2000 else 0 for values in df_raw_pandas_zips_no_na['RECORD_DATE']]
df_raw_pandas_zips_no_na['RECORD_DATE'] = conv_dates
train1 = df_raw_pandas_zips_no_na.drop(['PRICE','STREET','CITY','COUNTRY'],axis=1)

x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size = 0.10, random_state = 2)
st.write(reg.fit(x_train,y_train))
st.write(reg.score(x_test,y_test))

