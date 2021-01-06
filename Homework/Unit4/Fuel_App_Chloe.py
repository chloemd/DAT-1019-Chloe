# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 18:06:19 2020

@author: chloe
"""

import streamlit as st

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
def return_stats(df):
    avg_stats = df.groupby(['Year', 'Make', 'Model'])[['Combined MPG (FT1)', 
                                                 'City MPG (FT1)', 'Highway MPG (FT1)',
                                                 'Annual Fuel Cost (FT1)', 'Tailpipe CO2 (FT1)']]
    avg_stats = avg_stats.mean().reset_index()
    return avg_stats

st.cache()
df = load_data()
st.cache() 
urls = load_urls()
st.cache()
avg_stats = return_stats(df)


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
    y_cols = ['Combined MPG (FT1)', 'City MPG (FT1)', 'Highway MPG (FT1)', 'Annual Fuel Cost (FT1)', 'Tailpipe CO2 (FT1)']
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
            st.subheader(f"5 Best Vehicles for: {y_axis}")
        else:
            st.subheader(f"5 Worst Vehicles for: {y_axis}")
    elif y_axis not in want_min:
        if sort_bool:
            st.subheader(f"5 Worst Vehicles for: {y_axis}")
        else:
            st.subheader(f"5 Best Vehicles for: {y_axis}")
            
    st.table(avg_stats.sort_values(y_axis, ascending=sort_bool).iloc[:5][['Year', 'Make', 'Model', y_axis]])
    
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
    year = st.sidebar.selectbox(
        'Model Year',
        df['Year'].unique().tolist(),
        index=1)
    
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
    #st.markdown(f"![Picture of: {year} {make} {model}]({url})", unsafe_allow_html=True)
    
    

if page in ['Model Explorer', 'Causal Impact']:

    st.cache()
    pipe = make_pipeline(OneHotEncoder(use_cat_names=True), xgb.XGBRegressor())
    
    st.cache()
    X_train, X_val, y_train, y_val = train_test_split(df.drop('SalePrice', axis=1), df['SalePrice'], test_size=0.2, random_state=1985)    
    
if page == 'Model Explorer':
    num_rounds      = st.sidebar.number_input('Number of Boosting Rounds',
                                 min_value=100, max_value=5000, step=100)
    
    tree_depth      = st.sidebar.number_input('Tree Depth',
                                 min_value=2, max_value=8, step=1, value=3)
    
    learning_rate   = st.sidebar.number_input('Learning Rate',
                                    min_value=.001, max_value=1.0, step=.05, value=0.1)
    
    validation_size = st.sidebar.number_input('Validation Size',
                                      min_value=.1, max_value=.5, step=.1, value=0.2)
    
    random_state    = st.sidebar.number_input('Random State', value=1985)
        
    pipe[1].set_params(n_estimators=num_rounds, max_depth=tree_depth, learning_rate=learning_rate)
    
    X_train, X_val, y_train, y_val = train_test_split(df.drop('SalePrice', axis=1), df['SalePrice'], test_size=validation_size, random_state=random_state)
    
    pipe.fit(X_train, y_train)
    
    mod_results = pd.DataFrame({
            'Train Size': X_train.shape[0],
            'Validation Size': X_val.shape[0],
            'Boosting Rounds': num_rounds,
            'Tree Depth': tree_depth,
            'Learning Rate': learning_rate,
            'Training Score': pipe.score(X_train, y_train),
            'Validation Score': pipe.score(X_val, y_val)
            }, index=['Values'])
    
    st.subheader("Model Results")
    st.table(mod_results)
    

    st.subheader("Real vs Predicted Validation Values")

    chart = sns.regplot(x=pipe.predict(X_val), y=y_val)
    st.pyplot(chart.figure)
    
if page == 'Causal Impact':
        
        pipe.fit(X_train, y_train)
    
        # transform the dataset for PDPBox -- necessary step
        dataset = pipe[0].fit_transform(X_train)
        columns = dataset.columns.tolist()
        pipe.fit(X_train, y_train)
    
        pdp_col = st.sidebar.selectbox(
        'Column',
         df.columns.tolist(),
         index=1)
        
        st.subheader("Choose A Column To Explore Its Impact on SalePrice")
        
        # if the column is numeric
        if is_numeric_dtype(X_train[pdp_col]):
            # create the partial dependence figure for a numeric column
            pdp_fig = pdp.pdp_isolate(
                    model=pipe[1], dataset=dataset, model_features=columns, 
                    feature=pdp_col)
            fig, axes = pdp.pdp_plot(pdp_fig, pdp_col, plot_lines=True, frac_to_plot=100)
            
            st.pyplot(fig)
            
        else:
            # grab the columns to use for one hot encoding
            cols_to_use = [col for col in columns if pdp_col in col]
            pdp_fig = pdp.pdp_isolate(
                    model=pipe[1], dataset=dataset, model_features=columns,
                    feature=cols_to_use)
            
            fig, axes = pdp.pdp_plot(pdp_fig, pdp_col, plot_lines=True, frac_to_plot=100)
            
            # format tick labels if there are more than 9 unique values
            if len(cols_to_use) > 8:
                xtick_labels = [col.split('_')[1] for col in cols_to_use]
                axes['pdp_ax'].set_xticklabels(xtick_labels, rotation='vertical');
                st.pyplot(fig)
            else:
                st.pyplot(fig)
        
# pred_results = pd.DataFrame()
#         pred_results['true'] = y_val
#         pred_results['predicted'] = pipe.predict(X_val)
#         plotly_chart = px.scatter(pred_results, x='true', y='predicted', trendline='ols')    