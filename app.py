# %% Libraries
from logging import PlaceHolder
import dash 
import dash_core_components as dcc
from dash_core_components.Dropdown import Dropdown
from dash_core_components.Markdown import Markdown 
import dash_html_components as html 
from dash.dependencies import Input, Output
import dash_table
import plotly.express as px 
import pandas as pd
import sys 
import os
from random_forest import import_datasets
from random_forest import DataSet
from random_forest import initialize_models, train_models, import_models
import layout
from datetime import datetime as dt
import markdown
model = {}
data = {}
# %% External stylesheets 
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# %% Funcs

data = import_datasets()
import_models()

# %%
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# %%
app.layout = layout.layout

from callbacks.pred_page_callbacks import *
from callbacks.viz_page_callbacks import *



# %%
if __name__ == '__main__':
    app.run_server(debug=True)
