# %% Libraries
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
# %% External stylesheets 
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# %% TEXTS

introduction = """
# Pricing Different Car Brands 
I have created a number of different models for the following car brands:

"""

bmw_text ="""
**PERCOCET**

"""

audi_text = """
_Molly percocet_
"""

about_me_text = """
`I AM from anCient Greece`
"""

# %% Funcs

data = import_datasets()


# %%
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# %%

df = pd.read_csv('data/bmw.csv')

# %%

# %%
# %%
app.layout = html.Div(
    children=[
    dcc.Markdown(introduction),

    dcc.Tabs(id='car-tabs', value='tab-1', 
    children=[


        # Visualizations
        dcc.Tab(label="Visualizations", value='viz-tab', 
        children=[dcc.Markdown(bmw_text),

        dcc.Dropdown(
            id = 'make_selector',
            options = [
                {'label': 'Audi', 'value': 'audi'},
                {'label': 'BMW', 'value': 'bmw'},
                {'label': 'C-Class', 'value': 'cclass'},
                {'label': 'Ford Focus', 'value': 'focus'},
                {'label': 'Ford', 'value': 'ford'},
                {'label': 'Hyundai', 'value': 'hyundi'},
                {'label': 'Mercedez-Benz', 'value': 'merc'},
                {'label': 'Skoda', 'value': 'skoda'},
                {'label': 'Toyota', 'value': 'toyota'},
                {'label': 'Vauxhall', 'value': 'vauxhall'},
                {'label': 'VW', 'value': 'vw'}
            ],
            value = 'audi'
            
        ),

        html.Div(id='summary-stats'),

        dcc.Graph(id='price-mileage-scatter' ),
        dcc.Graph(id='box-plots' )
        
        ]),

        # About
        dcc.Tab(label="About", value="about-tab",
        children=[
            dcc.Markdown(about_me_text)
        ])

    ])



])

#%% Graphing Functions

@app.callback(
    Output(component_id = 'price-mileage-scatter', component_property='figure'),
    [Input(component_id = 'make_selector', component_property="value")]
)
def MileagePriceScatterplot(input_value):
    """Generates a scatter plot of mileage and price

    Args:
        input_value (string): string key for the data dict

    Returns:
        plotly.px: plotly express figure generated on given dataset
    """
    d = str(input_value)
    return px.scatter(data[d].df, x='mileage', 
    y='price',  
    color="model", 
    labels={"price":"Price", "model":"Model", "mileage":"Miles"},
    title = "Price and Mileage by Model", 
    )


@app.callback(
    Output(component_id = 'box-plots', component_property='figure'),
    [Input(component_id = 'make_selector', component_property="value")]
)
def BoxPlotter(input):
    d = str(input)
    df = data[d].df
    fig = px.box(df, x = "model", y = 'price')
    return(fig)


# %% Summary funcs

@app.callback(
    Output(component_id = 'summary-stats', component_property='children'),
    [Input(component_id = 'make_selector', component_property='value')]
)
def generate_summary_stats(input_value):
    d = str(input_value)
    df = data[d].df.describe()
    d = df.to_dict('rows')
    columns = [{"name": i, "id": i,} for i in (df.columns)]
    return(dash_table.DataTable(data = d, columns = columns))
# %%
if __name__ == '__main__':
    app.run_server(debug=True)

# %%