#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 12:51:06 2022

@author: ustokowska
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.header('st.write')
st.subheader('Display text')
st.write('Hello, *World!* :sunglasses:')

st.subheader('Display numbers')
st.write(1234)

st.subheader('Display DataFrame')
data = [[1, 10],[2, 20],[3, 30],[4, 40]]
df = pd.DataFrame(data,columns=['first column','second column'])
st.write(df)

df_2 = pd.DataFrame({
     'first column': [1, 2, 3, 4],
     'second column': [10, 20, 30, 40]
     })

st.subheader('Accept multiple arguments')
st.write('Below is a DataFrame:',df_2, 'Above is a DataFrame.')

st.subheader('Display charts')
df2 = pd.DataFrame(
     np.random.randn(200, 3),
     columns=['a', 'b', 'c'])
c = alt.Chart(df2).mark_circle().encode(
     x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
st.write(c)
