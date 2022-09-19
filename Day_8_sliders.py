#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 13:23:20 2022

@author: ustokowska
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import time, datetime

st.header('st.slider')
st.subheader('Slider')
age = st.slider('How old are you?',0,130,step=1,value=25)
st.write('I am', age, 'years old')

st.subheader('Range slider')
range_of_values = st.slider('Select a range of values',0.0,100.0,(25.0,75.0))
st.write('Values:',range_of_values)

st.subheader('Range time slider')
range_time_slider = st.slider('Schedule your appointment',value=(time(11, 30), time(12, 45)))
st.write('Your appointment is scheduled for:', range_time_slider)

st.subheader('Datetime slider')
start_time = st.slider(
     "When do you start?",
     value=datetime(2020, 1, 1, 9, 30),
     format="MM/DD/YY - hh:mm")
st.write("Start time:", start_time)