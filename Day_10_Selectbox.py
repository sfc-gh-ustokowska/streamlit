#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 13:29:40 2022

@author: ustokowska
"""

import streamlit as st

st.header('st.selectbox')

fav_color = st.selectbox('Wjat is your favorite color?', ('Blue','Red','Green'))

st.write('Your favorite color is ',fav_color)