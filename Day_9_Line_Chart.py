#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 13:18:00 2022

@author: ustokowska
"""

import streamlit as st
import pandas as pd
import numpy as np

st.header('Line chart')

data = np.random.randn(20, 3)
df = pd.DataFrame(data,columns=['a','b','c'])

st.line_chart(df)