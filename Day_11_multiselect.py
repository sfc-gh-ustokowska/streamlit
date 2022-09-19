#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 13:33:29 2022

@author: ustokowska
"""

import streamlit as st

st.header('st.multiselect')

options = st.multiselect('Choose your favorite colors',('Yellow','Red','Green','Blue'))

st.write('You selected', options)