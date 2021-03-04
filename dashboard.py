import streamlit as st
import requests
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# csfont = {'fontname':'Nexa Bold'} # Tuning font for plots
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from PIL import Image
# ------------------------------------------------------


# HTML requests to retrieve data -----------------------

# Total cases of covid
@st.cache(allow_output_mutation=True)
def get_cases():
    response = requests.get("https://datascientist.pythonanywhere.com/cases")
    cases = pd.read_json(response.content.decode('utf-8'))
    cases.sort_values(by=['Date'], inplace=True)
    cases.index = cases['Date']
    cases.drop(['Date'], axis=1, inplace=True)
    return cases

df_cases = get_cases()

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

#                  DASHBOARD

# ----------------------------------------------
st.subheader('Nombre de cas confirmés en France')
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('fivethirtyeight')
# fig = plt.figure(figsize=(10,5), constrained_layout=True)

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

plt.plot(df_cases.index, df_cases['France'], lw=3)
# df_cases.plot(lw=2)



# Display the dashboard
st.pyplot()











