#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 13:54:16 2022

@author: ustokowska
"""

import streamlit as st

st.header('st.checkbox')
st.write('What would you like to order?')
icecream = st.checkbox('Ice Cream')
coffee = st.checkbox('Coffe')
cola = st.checkbox('Cola')

if icecream: 
    st.write("Great! Here's some more 🍦")
if coffee:
    st.write("Okay, here's some coffee ☕️")
if cola:
    st.write("Here you go 🥤")