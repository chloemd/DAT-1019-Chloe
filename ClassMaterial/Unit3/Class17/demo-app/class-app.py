# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 20:07:28 2020

@author: chloe
"""
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px

# cache to load in data once at the beginning
@st.cache
def load_data():
    df = pd.read_csv('iowa_train2.csv')
    return df


df = load_data()

st.title("Our First Data Application")
st.write(df)


page = st.sidebar.radio('Section',
                        ['Data Explorer', 'Model Explorer'])

if page == 'Data Explorer':
    grouping = df.groupby('Neighborhood')['SalePrice'].mean()
    st.header("Average Sale Price by Neighborhood")
    st.line_chart(grouping)