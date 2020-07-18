# All the callbacks and associated functions for the visualizations page

from dash.dependencies import Input, Output
from app import *
from random_forest import *

@app.callback(
    Output(component_id = 'summary-stats', component_property='children'),
    [Input(component_id = 'viz-brand-selector', component_property='value')]
)
def preview_data(input_value):
    d = str(input_value)
    df = data[d].df.head(5)
    d = df.to_dict('rows')
    columns = [{"name": i, "id": i,} for i in (df.columns)]
    return(dash_table.DataTable(data = d, columns = columns ))

@app.callback(
    Output(component_id = 'box-plots', component_property='figure'),
    [Input(component_id = 'viz-brand-selector', component_property="value"),
    Input(component_id = 'viz-component-selector', component_property='value')]
)
def BoxPlotter(data_select, component):
    d = str(data_select)
    c = str(component)
    title = "{} {} Distributions".format(d.capitalize(), c.capitalize())
    df = data[d].df
    fig = px.box(df, x = "model", y = c, title = title)
    return(fig)

@app.callback(
    Output('viz-component-selector', 'options'),
    [Input('viz-brand-selector', 'value')]
)
def viz_selector_update_component(selected_brand):
    sb = str(selected_brand).lower()
    return [{'label': x.capitalize(), 'value': x} for x in data[sb].df.columns.unique()]


@app.callback(
    Output(component_id = 'price-mileage-scatter', component_property='figure'),
    [Input(component_id = 'viz-brand-selector', component_property="value"),
    Input(component_id = 'viz-component-selector', component_property='value')]
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


