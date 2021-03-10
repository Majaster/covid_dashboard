import streamlit as st
import requests
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
# csfont = {'fontname':'Nexa Bold'} # Tuning font for plots
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from PIL import Image
import io

plt.style.use('fivethirtyeight')
# ------------------------------------------------------


# HTML requests to retrieve data -----------------------

# Total cases of covid
# @st.cache(allow_output_mutation=True)
# def get_cases():
#     response = requests.get("https://datascientist.pythonanywhere.com/cases")
#     cases = pd.read_json(response.content.decode('utf-8'))
#     cases.sort_values(by=['Date'], inplace=True)
#     cases.index = cases['Date']
#     cases.drop(['Date'], axis=1, inplace=True)
#     return cases

# df_cases = get_cases()

# Daily refreshed data from John Hopkins University (Github dataset)
@st.cache(allow_output_mutation=True)
def get_covid_cases():
    response = requests.get("https://datascientist.pythonanywhere.com/covid")
    covid = pd.read_json(response.content.decode('utf-8'))
    return covid

data = get_covid_cases()

# Daily refreshed data from John Hopkins University (owid dataset)
@st.cache(allow_output_mutation=True)
def get_covid_owid():
    response = requests.get("https://datascientist.pythonanywhere.com/covid_owid")
    covid_owid = pd.read_json(response.content.decode('utf-8'))
    return covid_owid

covid_owid = get_covid_owid()

# Select weekly cases
weekly = covid_owid[['date','location','weekly_cases']]

# Pivot to have dates in index
df_weekly_cases = weekly.pivot(index='date', columns='location', values='weekly_cases')

# Reorganize columns
countries = list(df_weekly_cases.columns)
df_weekly_cases.columns = countries




# Select countries
countries = ['Canada', 'Germany', 'United Kingdom', 'US', 'France', 'China']
df = data[data['Country'].isin(countries)]

# Create a summary column of all cases
df['Cases'] = df[['Confirmed', 'Recovered', 'Deaths']].sum(axis=1)

# Pivot the data
df_cases = df.pivot(index='Date', columns='Country', values='Cases')
# df_confirmed = df.pivot(index='Date', columns='Country', values='Confirmed')
# df_deaths = df.pivot(index='Date', columns='Country', values='Deaths')

# Restructure columns in right order
countries = list(df_cases.columns)
df_cases.columns = countries
# df_confirmed.columns = countries
# df_deaths.columns = countries


# Start of the main page ------------------------------

# Display an image at the top
# response = requests.get("https://datascientist.pythonanywhere.com/image")
# image = Image.open(response.content)
# st.image(image, caption=None, use_column_width=True)


"""
# Statistiques à propos du COVID-19

Évolution de la pandémie due au coronavirus en France et dans le monde.

"""


# st.subheader('Cases data')
# st.dataframe(df_cases, height=100)


# ----------------------------------------------

#                  FRANCE

# ----------------------------------------------
st.subheader('Nombre de cas confirmés en France')

#---- create figure ----

fwidth = 10.  # total width of the figure in inches
fheight = 5. # total height of the figure in inches

fig = plt.figure(figsize=(fwidth, fheight), constrained_layout=True)

#---- define margins -> size in inches / figure dimension ----

left_margin  = 1.2 / fwidth
right_margin = 0.2 / fwidth
bottom_margin = 0.5 / fheight
top_margin = 0.25 / fheight

#---- create axes ----

# dimensions are calculated relative to the figure size

x = left_margin    # horiz. position of bottom-left corner
y = bottom_margin  # vert. position of bottom-left corner
w = 1 - (left_margin + right_margin) # width of axes
h = 1 - (bottom_margin + top_margin) # height of axes

ax = fig.add_axes([x, y, w, h])

#---- Define the Ylabel position ----

# Location are defined in dimension relative to the figure size  

# xloc =  0.15 / fwidth 
# yloc =  y + h / 2.  

# ax.set_ylabel('yLabel', fontsize=16, verticalalignment='top', horizontalalignment='center')             
# ax.yaxis.set_label_coords(xloc, yloc, transform = fig.transFigure)

# matplotlib gridspec
# spec = fig.add_gridspec(3, 4, wspace=0.0, hspace=0.0) 


# Create our own color map (blue gradient)
# N = 256
# vals = np.ones((N, 4))
# vals[:, 0] = np.linspace(50/256, 0.05, N)
# vals[:, 1] = np.linspace(75/256, 0.08, N)
# vals[:, 2] = np.linspace(180/256, 0.3, N)
# custom_cm = ListedColormap(vals)


# ax.set_facecolor('#e4eef9')


# Weekly cases in France 
plt.plot(df_weekly_cases.index, df_weekly_cases['France'].values, lw=3)


# Checkbox to switch between daily / cumulated cases

cases_switch = st.checkbox('Voir cas cumulés')
if cases_switch:
    plt.plot(df_cases.index, df_cases['France'], lw=3) 

st.pyplot()


# ----------------------------------------------

#                  MONDE

# ----------------------------------------------
st.subheader('Nombre de cas total dans le monde')
"""
Le nombre de cas total est la somme du nombre de cas confirmés, soignés et décédés.
"""

# Figure parameters
fwidth = 10.  # total width of the figure in inches
fheight = 6. # total height of the figure in inches

fig = plt.figure(figsize=(fwidth, fheight), constrained_layout=True)

#---- define margins -> size in inches / figure dimension ----

left_margin  = 1.2 / fwidth
right_margin = 1.2 / fwidth
bottom_margin = 1 / fheight
top_margin = 0.15 / fheight

#---- create axes ----

# dimensions are calculated relative to the figure size

x = left_margin    # horiz. position of bottom-left corner
y = bottom_margin  # vert. position of bottom-left corner
w = 1 - (left_margin + right_margin) # width of axes
h = 1 - (bottom_margin + top_margin) # height of axes

ax = fig.add_axes([x, y, w, h])

# data to use
data = df_cases
# data = df_cases.drop(['US'], axis=1)

# Maximum of cases from data
data_max = int(data.max().max())

# Colors and style
colors = {'Canada':'#13a0ac', 'China':'#f10505', 'France':'#1d5ccb', 'Germany':'#cf1e81', 
          'US':'#e96920', 'United Kingdom':'#0e711a'}
# colors = {'Canada':'#13a0ac', 'China':'#f10505', 'France':'#1d5ccb', 'Germany':'#cf1e81', 
#           'United Kingdom':'#0e711a'}

# Create the visualization
plot = data.plot(figsize=(fwidth,fheight), color=list(colors.values()), linewidth=2, legend=False)

# Convert y-axis values in millions for readability
def millions(x, pos):
#     The two args are the value and tick position
    return '%1.0fM' % (x * 1e-6)


formatter = FuncFormatter(millions)
plot.yaxis.set_major_formatter(formatter)
# plot.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
plot.grid(color='#d4d4d4')
plot.set_xlabel('Date', fontsize=12)
plot.set_ylabel('Nombre total de cas', fontsize=12, labelpad=15)

# Assign countries
for country in list(colors.keys()):
    plt.text(x = data.index[-1], y = data[country].max() * 1.015, 
              color=colors[country], s=country, weight='bold', fontsize=10)

# Labels
plt.text(x = data.index[1], y = data_max * (-0.32),
          s = 'Source: https://github.com/datasets/covid-19/blob/master/data/countries-aggregated.csv', 
          fontsize=12, alpha=.75)


# Show plot
st.pyplot()











