#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 13:54:16 2022

@author: ustokowska
"""
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import avg, sum, col,lit
import streamlit as st
import pandas as pd
import os
import numpy as np
import pydeck as pdk

url = 'https://raw.githubusercontent.com/sfc-gh-ustokowska/streamlit/main/1.csv'
url2 = 'https://raw.githubusercontent.com/sfc-gh-ustokowska/streamlit/main/2.csv'
url3 = 'https://raw.githubusercontent.com/sfc-gh-ustokowska/streamlit/main/3.csv'
url_zips = 'https://raw.githubusercontent.com/sfc-gh-ustokowska/streamlit/main/zipcodes.csv'

df1 = pd.read_csv(url, index_col = 1)
df2 = pd.read_csv(url2, index_col = 1)
df3 = pd.read_csv(url3, index_col = 1)

df = pd.concat([df1, df2, df3])
df['mean'] = df.mean(axis=1)

#st.write(df.head(100))

df_zips = pd.read_csv(url_zips, index_col = 0)

df_mean_per_zip = df[['RegionName','mean']]

df_full = pd.merge(df_mean_per_zip, df_zips, how='left', left_on = 'RegionName', right_on = 'Zipcode')
df_full_no_na = df_full.dropna()

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