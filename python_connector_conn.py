#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 13:54:16 2022

@author: ustokowska
"""
import os
import snowflake.connector
import streamlit as st

def init_connection():
  con = snowflake.connector.connect(
    user = os.getenv("USER_SNOW"),
    password = os.getenv("PASSWORD"),
    account = os.getenv("ACCOUNT"),
    role = os.getenv("ROLE"),
    warehouse = os.getenv("WAREHOUSE")
  )
  return con

def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()

conn = init_connection()

query = "CREATE OR REPLACE DATABASE EMPLOYEES;"
rows = run_query(query)

for row in rows:
    st.write(row)