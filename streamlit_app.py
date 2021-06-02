import streamlit as st
import requests
import json
import numpy as np
import pandas as pd
from PIL import Image
import pydeck as pdk
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# Matplotlib style
plt.style.use('fivethirtyeight')

# Colors dictionary
cdict = {'orange':'#e4631a','lightblue':'#17c3d4','red':'#f10505','blue':'#1d5ccb',
         'purple':'#cf1e81','green':'#0e711a','brown':'#7c3e25','emeraud':'#15c286',
         'pink':'#f360a1','lightorange':'#efa614','black':'#1e1e1e'}

# ------------------------------------------------------

# Set page configuration
icon = Image.open("Images/covid_icon.png")
st.set_page_config(
    page_title="Covid Dashboard",
    page_icon=icon,
    layout="centered",
    initial_sidebar_state="auto",
)

# Retrieve data -----------------------------------------

# Daily refreshed data from John Hopkins University (Github dataset)
@st.cache(allow_output_mutation=True)
def get_covid_cases():
    response = requests.get("https://datascientist.pythonanywhere.com/covid")
    covid = pd.read_json(response.content.decode('utf-8'))
    return covid

covid = get_covid_cases()

# Daily refreshed data from John Hopkins University (owid dataset)
@st.cache(allow_output_mutation=True)
def get_covid_owid():
    response = requests.get("https://datascientist.pythonanywhere.com/covid_owid")
    covid_owid = pd.read_json(response.content.decode('utf-8'))
    return covid_owid

covid_owid = get_covid_owid()

# ---------------------------------------------------------

# Manipulate data -----------------------------------------

# Select weekly cases
weekly = covid_owid[['date','location','weekly_cases']]

# Pivot to have dates in index
df_weekly_cases = weekly.pivot(index='date', columns='location', values='weekly_cases')

# Reorganize columns
countries = list(df_weekly_cases.columns)
df_weekly_cases.columns = countries



# Select countries
countries = ['Canada','Germany','United Kingdom','US','France','China','Italy','Spain']
df = covid[covid['Country'].isin(countries)]

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


# Cases percentage
# Source : https://fr.vikidia.org/wiki/Liste_des_pays_par_population
populations = {'Canada':37742154, 'Germany': 83783942, 'United Kingdom': 67886011, 
               'US': 331002651 , 'France': 65273511, 'China':1439323776,
               'Italy': 60461826, 'Spain':46754778}

percapita = df_cases.copy()
for country in percapita.columns:
    percapita[country] = percapita[country]/populations[country]*100


# ---------------------------------------------------

#               BEGINNING OF PAGE

# ---------------------------------------------------
st.image("Images/covid_banner.jpg")

"""
# Évolution du COVID-19 en France et dans le monde

Vous trouverez dans un premier temps des graphiques/cartes correspondant 
à ce qui se passe en France, puis vous verrez les cas de contamination dans 
plusieurs pays à travers le monde (choisis plus ou moins arbitrairement).

Les données sont celles de l'Université John Hopkins (JHU) collectées 
et actualisées quotidiennement.

"""

# ----------------------------------------------

#                   FRANCE

# ----------------------------------------------
st.subheader('Nombre de cas testés positif en France')

"""
Cette première courbe montre le nombre de cas testés positif 
au covid-19 (moyenne sur 7 jours). Nous apercevons nettement 
les vagues de mars 2020, novembre 2020 et avril 2021.
"""

# Figure 
# (dimensions are calculated relative to the figure size)
fwidth = 10.  # total width of the figure in inches
fheight = 5. # total height of the figure in inches

fig = plt.figure(figsize=(fwidth, fheight))

# Define margins
left_margin  = 1.2 / fwidth
right_margin = 0.2 / fwidth
bottom_margin = 0.5 / fheight
top_margin = 0.25 / fheight

# Create axes
x = left_margin    # horiz. position of bottom-left corner
y = bottom_margin  # vert. position of bottom-left corner
w = 1 - (left_margin + right_margin) # width of axes
h = 1 - (bottom_margin + top_margin) # height of axes

ax = fig.add_axes([x, y, w, h])

# Define the Ylabel position
# xloc =  0.15 / fwidth 
# yloc =  y + h / 2.  

# ax.set_ylabel('yLabel', fontsize=16, verticalalignment='top', horizontalalignment='center')             
# ax.yaxis.set_label_coords(xloc, yloc, transform = fig.transFigure)

# data to use
data = df_weekly_cases

# Weekly cases in France 
plt.plot(data.index, data['France'].values, lw=3)

# Checkbox to switch between daily / cumulated cases
cases_switch = st.checkbox('Afficher le nombre de cas cumulés')
if cases_switch:
    plt.plot(df_cases.index, df_cases['France'], lw=3) 

# Source at bottom
plt.text(x=0., y=-0.15, transform=ax.transAxes,
         s='Source: https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/jhu/full_data.csv', 
         fontsize=11, fontstyle='italic',alpha=.5)

st.pyplot(fig)


# ----------------------------------------------

#                 MAP FRANCE

# ----------------------------------------------
# Departements
# last_day_dep = pd.read_csv('last_day_dep.csv')
# last_day_dep.dropna(inplace=True)
# last_day_dep = last_day_dep.iloc[:10,:]

# Read data (Regions)
last_day_regions = pd.read_csv('last_day_regions.csv', parse_dates=['date'])

# Converts date objects into datetime
last_day_regions.date = pd.to_datetime(last_day_regions.date, 
                                       format='%Y-%m-%d', errors='coerce')

# Data to use
data = last_day_regions

# Convert date from datetime to french strings
Mois=['Janvier','Février','Mars','Avril','Mai','Juin','Juillet',
      'Août','Septembre','Octobre','Novembre','Décembre']

last_day = data['date'][0]
day = str(last_day.day)
month = last_day.month
year = str(last_day.year)

# Write a sentence to indicate the day corresponding to data
st.write("Nombre d'hospitalisations au jour du "+ day + " " 
         + Mois[month-1].lower() + " " + year + " :")

# Adding point so we can have map default to the center of the data
# midpoint = (np.average(data['lat']), np.average(data['lon']))

# Center of France
midpoint = (46.227638,2.213749)

st.pydeck_chart(
    pdk.Deck(
             map_style='mapbox://styles/mapbox/light-v10',
             initial_view_state=pdk.ViewState(
             latitude=midpoint[0],
             longitude=midpoint[1],
             zoom=4,
             pitch=0,
             ),
     layers=[
       pdk.Layer(
        'ScatterplotLayer',
        data=data,
        get_position=['lon','lat'],
        get_color=[0, 142, 218, 60],
        get_radius=['hospitalises'],
        radiusMin=10000,
        radiusScale=20,
        auto_highlight=True,
        )
     ]
))


# -------------------------------------------------------------------

#                               MONDE

# -------------------------------------------------------------------
st.subheader('Nombre de cas total dans le monde')
"""
Le nombre de cas total est la somme du nombre de cas confirmés, soignés 
et décédés.

Le premier graphe montre le nombre total et le second correspond 
au pourcentage de la population par pays.
"""

# Plot of total cases in the world ----------------------------------
fwidth = 10.  # total width of the figure in inches
fheight = 6. # total height of the figure in inches

fig = plt.figure(figsize=(fwidth, fheight))

# Define margins
left_margin  = 1.2 / fwidth
right_margin = 0.2 / fwidth
bottom_margin = 0.5 / fheight
top_margin = 0.25 / fheight

# Create axes
x = left_margin    # horiz. position of bottom-left corner
y = bottom_margin  # vert. position of bottom-left corner
w = 1 - (left_margin + right_margin) # width of axes
h = 1 - (bottom_margin + top_margin) # height of axes

ax = fig.add_axes([x, y, w, h])

# data to use
data = df_cases

# Sorting columns according to last row (ie date) descending way 
# in order to plot the legend accordingly to these values
data.sort_values(by=data.index[-1], inplace=True, axis=1, ascending=False)

# Convert y-axis values in millions for readability
def millions(x, pos):
#     The two args are the value and tick position
    return '%1.0fM' % (x * 1e-6)

formatter = FuncFormatter(millions)
ax.yaxis.set_major_formatter(formatter)

# Attribute colors to countries
colors = {'US':cdict['orange'],'China':cdict['red'],'France':cdict['blue'],
          'Spain':cdict['purple'],'Germany':cdict['black'],'Italy':cdict['green'],
          'Canada':cdict['brown'],'United Kingdom':cdict['lightorange']}

# Assign countries
n_country = len(data.columns)
top_label = 0.8
low_label = 0.2
step_label = (top_label - low_label) / n_country
i=0
for country in data.columns:
    plt.plot(data.index, data[country].values, c=colors[country], lw=2, label=country)
#     plt.text(x = 1.02, y = top_label - step_label * i, transform=ax.transAxes,
#              c=colors[i], s=country, weight='bold', fontsize=12)
#     plt.text(x = data.index[-1], y = data[country].max() * 1.015, 
#              c=colors[i], s=country, weight='bold', fontsize=12)

# Legend
plt.legend(loc=(1.02,0.5))

# Source at bottom
plt.text(x=0., y=-0.12, transform=ax.transAxes,
         s='Source: https://github.com/datasets/covid-19/blob/master/data/countries-aggregated.csv', 
         fontsize=11, fontstyle='italic', alpha=.5)

# Show plot
st.pyplot(fig)


# Plot of total cases in the world (ratio) ------------------------------
fig = plt.figure(figsize=(fwidth, fheight))
ax = fig.add_axes([x, y, w, h])

# data to use
data = percapita

# Sorting columns according to last row (ie date) descending way 
# in order to plot the legend accordingly to these values
data.sort_values(by=data.index[-1], inplace=True, axis=1, ascending=False)

def rate(x, pos):
    return '%1.0f%%' % x
formatter = FuncFormatter(rate)
ax.yaxis.set_major_formatter(formatter)
# ax.yaxis.set_major_formatter(ticker.PercentFormatter(1000))
# manipulate
# vals = ax.get_yticks()
# ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])

# Assign countries
n_country = len(data.columns)
top_label = 0.8
low_label = 0.2
step_label = (top_label - low_label) / n_country
i=0
for country in data.columns:
    plt.plot(data.index, data[country].values, c=colors[country], lw=2, label=country)

# Legend
plt.legend(loc=(1.02,0.5))

# Source at bottom
plt.text(x=0., y=-0.12, transform=ax.transAxes,
         s='Source: https://github.com/datasets/covid-19/blob/master/data/countries-aggregated.csv', 
         fontsize=11, fontstyle='italic', alpha=.5)

# Show plot
st.pyplot(fig)





                 


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






