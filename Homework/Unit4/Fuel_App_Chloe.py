# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 18:06:19 2020

@author: chloe 
"""



title = 'Fuel Economy Data'



import streamlit as st
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import seaborn as sns
import xgboost as xgb
from matplotlib.pyplot import style
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder
from sklearn.pipeline import make_pipeline
from pdpbox import pdp
import plotly.express as px
import plotly.graph_objects as go


style.use('ggplot')
st.title("Fuel Economy Data")

@st.cache
def load_data():
    df = pd.read_csv(r"C:\Users\chloe\Data Science\DAT-1019-Chloe\Homework\Unit4\data\fuel_eco_clean.csv")
    return df

@st.cache
def load_urls():
    urls = pd.read_csv(r"C:\Users\chloe\Data Science\DAT-1019-Chloe\Homework\Unit4\data\fuelgov_img_urls.csv")
    return urls

@st.cache
def create_groupby_object(x_axis, y_axis):
    data = df.groupby(x_axis)[y_axis].mean()
    return data

@st.cache
def create_plotly_graph_data(x_axis, y_axis):
    data = df.groupby(x_axis)[y_axis].mean().to_frame().reset_index()
    return data

@st.cache
def return_stats(df, y_cols):
    avg_stats = df.groupby(['Year', 'Make', 'Model'])[y_cols]
    avg_stats = avg_stats.mean().reset_index()
    return avg_stats

st.cache()
df = load_data()
st.cache() 
urls = load_urls()



page = st.sidebar.radio('Section',
                        ['Data Explorer', 'Vehicle Details'])



if page == 'Data Explorer':
    st.markdown("""
                <style>
                table td:nth-child(1) {
                    display: none
                    }
                table th:nth-child(1) {
                    display: none
                    }
                </style>
                """, unsafe_allow_html=True)
    
    
    x_cols = ['Year', 'Class', 'Drive', 'Transmission', 'Fuel Type', 'Engine Cylinders', 'Engine Displacement']
    y_cols = ['Combined MPG (FT1)', 'City MPG (FT1)', 'Fuel Economy Score',
              'Highway MPG (FT1)', 'Annual Fuel Cost (FT1)', 'Tailpipe CO2 (FT1)']
    
    st.cache()
    avg_stats = return_stats(df, y_cols)
    
    sort_table = st.sidebar.selectbox('Sort Data',
                        ['Ascending', 'Descending'])
    sort_bool = sort_table == 'Ascending'
    
    x_axis = st.sidebar.selectbox(
        'X-Axis',
         x_cols,
         index=0)
    
    y_axis = st.sidebar.selectbox(
             'Y-Axis',
             y_cols
            # df.select_dtypes(include=np.number).columns.tolist()
            )
    
    want_min = ['Annual Fuel Cost (FT1)', 'Tailpipe CO2 (FT1)']
    
    if y_axis in want_min:
        if sort_bool:
            st.subheader(f"5 Best Vehicles for {y_axis}")
        else:
            st.subheader(f"5 Worst Vehicles for {y_axis}")
    elif y_axis not in want_min:
        if sort_bool:
            st.subheader(f"5 Worst Vehicles for {y_axis}")
        else:
            st.subheader(f"5 Best Vehicles for {y_axis}")
            
    st.table(avg_stats[avg_stats[y_axis] > 0].sort_values(y_axis, ascending=sort_bool).iloc[:5][['Year', 'Make', 'Model', y_axis]])
    
    chart_type = st.sidebar.selectbox(
            'Select a chart type:',
            ['Line', 'Bar', 'Box'])
    
    
    st.subheader(f"Breaking Down {y_axis} by: {x_axis}")
    if chart_type == 'Line':
        data = create_groupby_object(x_axis, y_axis)
        st.line_chart(data)
    elif chart_type == 'Bar':
        data = create_groupby_object(x_axis, y_axis)
        st.bar_chart(data)
        # chart = px.bar(df, x=x_axis, y=y_axis)
        # st.write(chart)
    elif chart_type == 'Box':
        chart = px.box(df, x=x_axis, y=y_axis)
        st.write(chart)
        #if df[x_axis].nunique() > 8:
        #    chart.set_xticklabels(rotation=90)
        #st.pyplot(chart)
   
if page == 'Vehicle Details':
    st.markdown("""
                <style>
                table td:nth-child(1) {
                    display: none
                    }
                table th:nth-child(1) {
                    display: none
                    }
                </style>
                """, unsafe_allow_html=True)
    car_info_cols = ['Class', 'Transmission','Combined MPG (FT1)', 'City MPG (FT1)', 'Highway MPG (FT1)', 'Annual Fuel Cost (FT1)', 'Fuel Economy Score']
    unique_years = df['Year'].unique().tolist()
    year = st.sidebar.selectbox(
        'Model Year',
        unique_years,
        index=len(unique_years)-1)
    
    make = st.sidebar.selectbox(
        'Make',
        df[df['Year'] == year]['Make'].unique().tolist(), 
        index=0)
    
    model = st.sidebar.selectbox(
        'Model',
        df[(df['Year'] == year) & (df['Make'] == make)]['Model'].unique().tolist(),
        index=0)
    
    url = urls[(urls['Year'] == year) & (urls['Make'] == make) & (urls['Model'] == model)]['Image Url'].item()
    st.sidebar.markdown(f"![Picture of: {year} {make} {model}]({url})", unsafe_allow_html=True)
    st.subheader(f"{year} {make} {model}")
    st.table(df[(df['Year'] == year) & (df['Make'] == make) & (df['Model'] == model)][car_info_cols])

    
    
