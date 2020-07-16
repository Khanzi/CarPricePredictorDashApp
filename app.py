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
from random_forest import DataSet, PredData
from random_forest import initialize_models, train_models, show_scores
from datetime import datetime as dt
import markdown
models = {}
# %% External stylesheets 
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# %% TEXTS

# %% Funcs

data, predict_x = import_datasets()


# %%
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# %%

df = pd.read_csv('data/bmw.csv')

# %%

# %%
# %%
app.layout = html.Div(
    children=[
    dcc.Markdown(markdown.introduction),

    dcc.Tabs(id='car-tabs', value='tab-1', 
    children=[

        # Predictions
        dcc.Tab(label='Predictor', value = 'pred-tab',
        children=[
            html.Div([
                dcc.Dropdown(id = 'predict-brand-selector', placeholder = "Select Brand",
                    options=[{'label':x[:-4].capitalize(), 'value': x[:-4]} for x in os.listdir('data')], value = 'Audi'
                ),

                dcc.Dropdown(id = 'predict-model-selector', placeholder = 'Select Model',
                ),
                
                dcc.Input(id = 'predict-year-selector', type = 'number', placeholder = 'Year',
                min=1950, max = 2025, step = 1, value = 2020,
                ),

                dcc.Dropdown(id = 'predict-transmission-selector', placeholder = "Select Transmission"
                ),

                dcc.Dropdown(id = 'predict-fueltype-selector', placeholder = "Select Fuel Type"
                
                ),

                dcc.Dropdown(id = 'predict-enginesize-selector', placeholder = 'Select Engine Size'),

                dcc.Input(id = 'predict-mpg-selector', type = 'number', 
                placeholder = 'MPG', min = 1, max = 150),

                dcc.Input(id = 'predict-mileage-selector', type = 'number',
                placeholder = 'ODO Miles', min = 0),

                html.Div(id = 'predict-selected-car-preview')
                
            ])
        ]
        
        ),


        # Visualizations
        dcc.Tab(label="Visualizations", value='viz-tab', 
        children=[dcc.Markdown(markdown.visualizations_text),

        html.Div(
            [
                html.Div(
                    [

                    dcc.Dropdown( id = 'make_selector', # TODO: Clean this up
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
                        value = 'audi',
                        style={'width': '48%', 'display': 'inline-block'}
                    
                ),
                    ]
                ),
                html.Div(
                    [
                    
                    dcc.Dropdown( # TODO: Clean this up
                        id = 'box-component-selector',
                        options = [
                            {'label': 'Year', 'value': 'year'},
                            {'label': 'Price', 'value': 'price'},
                            {'label': 'Mileage', 'value': 'mileage'},
                            {'label': 'Tax', 'value': 'tax'},
                            {'label': 'MPG', 'value': 'mpg'},
                            {'label': 'Engine Size (L)', 'value': 'engineSize'}
                        ],
                        value = 'mileage',
                        style={'width': '48%', 'display': 'inline-block'}
                    )]
                )
            ], className = "row"
        ),

        html.H2("Sample of Data"),


        html.Div(id='summary-stats'),

        dcc.Graph(id='price-mileage-scatter' ),

        dcc.Graph(id='box-plots' )
        
        ]),

        # About
        dcc.Tab(label="About", value="about-tab",
        children=[
            dcc.Markdown(markdown.about_me_text)
        ])

    ])



])

#%% Graphing Functions

@app.callback(
    Output(component_id = 'price-mileage-scatter', component_property='figure'),
    [Input(component_id = 'make_selector', component_property="value"),
    Input(component_id = 'box-component-selector', component_property='value')]
)
def ScatterPlot(input_value, value):
    d = str(input_value)
    v = str(value)
    title = "Price and {} by Brand".format(v.capitalize())
    return px.scatter(data[d].df, x=v, 
    y='price',  
    color="model", 
    labels={"price":"Price", "model":"Model", v:v.capitalize()},
    title = title, 
    )


@app.callback(
    Output(component_id = 'box-plots', component_property='figure'),
    [Input(component_id = 'make_selector', component_property="value"),
    Input(component_id = 'box-component-selector', component_property='value')]
)
def BoxPlotter(data_select, component):
    d = str(data_select)
    c = str(component)
    title = "{} {} Distributions".format(d.capitalize(), c.capitalize())
    df = data[d].df
    fig = px.box(df, x = "model", y = c, title = title)
    return(fig)


# %% Summary funcs

@app.callback(
    Output(component_id = 'summary-stats', component_property='children'),
    [Input(component_id = 'make_selector', component_property='value')]
)
def preview_data(input_value):
    d = str(input_value)
    df = data[d].df.head(5)
    d = df.to_dict('rows')
    columns = [{"name": i, "id": i,} for i in (df.columns)]
    return(dash_table.DataTable(data = d, columns = columns ))


#%% Server funcs 

### Predictor Page
@app.callback(
    Output('predict-model-selector', 'options'),
    [Input('predict-brand-selector', 'value')]
)
def predictor_selector_update_models(selected_model):
    sm = str(selected_model).lower()
    return [{'label': x, 'value': x} for x in data[sm].df.model.unique()]

@app.callback(
    Output('predict-transmission-selector', 'options'),
    [Input('predict-brand-selector', 'value')]
)
def predictor_selector_update_transmission(selected_model):
    sm = str(selected_model).lower()
    return [{'label': x, 'value': x} for x in data[sm].df.transmission.unique()]

@app.callback(
    Output('predict-fueltype-selector', 'options'),
    [Input('predict-brand-selector', 'value')]
)
def predictor_selector_update_fueltype(selected_model):
    sm = str(selected_model).lower()
    return [{'label': x, 'value': x} for x in data[sm].df.fuelType.unique()]

@app.callback(
    Output('predict-enginesize-selector', 'options'),
    [Input('predict-brand-selector', 'value')]
)
def predictor_selector_update_enginesize(selected_model):
    sm = str(selected_model).lower()
    return [{'label': x, 'value': x} for x in data[sm].df.engineSize.unique()]

@app.callback(
    Output('predict-selected-car-preview', 'children'),
    [
        Input('predict-model-selector', 'value'),
        Input('predict-year-selector', 'value'),
        Input('predict-transmission-selector', 'value'),
        Input('predict-fueltype-selector', 'value'),
        Input('predict-enginesize-selector', 'value'),
        Input('predict-mpg-selector', 'value'),
        Input('predict-mileage-selector', 'value')
    ]

)
def predictor_selected_data(model, year, transmission, fueltype, enginesize,mpg,mileage):
    preview = "Model: {} {}  Transmission: {} \n Fueltype: {} \n Engine Size: {}L \n MPG: {} \n ODO Miles: {} ".format(model, year, transmission, fueltype, enginesize, mpg, mileage)
    return(preview)
# %%
if __name__ == '__main__':
    app.run_server(debug=True)

# %%