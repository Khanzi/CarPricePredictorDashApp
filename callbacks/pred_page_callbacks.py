# pred_page_callbacks.py
""" 
Here I define all the callbacks for the prediction page
of the app.

"""


# Import
from dash.dependencies import Input, Output
from app import *
from random_forest import *



# Based on the make selected return the options for different models.
@app.callback(
    Output('predict-model-selector', 'options'),
    [Input('predict-brand-selector', 'value')]
)
def predictor_selector_update_models(selected_model):
    sm = str(selected_model).lower()
    return [{'label': x, 'value': x} for x in data[sm].df.model.unique()]

# For the selected brand return the options for different transmissions
@app.callback(
    Output('predict-transmission-selector', 'options'),
    [Input('predict-brand-selector', 'value')]
)
def predictor_selector_update_transmission(selected_model):
    sm = str(selected_model).lower()
    return [{'label': x, 'value': x} for x in data[sm].df.transmission.unique()]

# For the selected brand return the options for different fueltypes
@app.callback(
    Output('predict-fueltype-selector', 'options'),
    [Input('predict-brand-selector', 'value')]
)
def predictor_selector_update_fueltype(selected_model):
    sm = str(selected_model).lower()
    return [{'label': x, 'value': x} for x in data[sm].df.fuelType.unique()]

# For the selected brand return the options for the different engine sizes
@app.callback(
    Output('predict-enginesize-selector', 'options'),
    [Input('predict-brand-selector', 'value')]
)
def predictor_selector_update_enginesize(selected_model):
    sm = str(selected_model).lower()
    return [{'label': x, 'value': x} for x in data[sm].df.engineSize.unique()]

# For the selected options display the selections as a single row dataframe
@app.callback(
    Output('predict-selected-car-preview', 'children'),
    [
        Input('predict-brand-selector', 'value'),
        Input('predict-model-selector', 'value'),
        Input('predict-year-selector', 'value'),
        Input('predict-transmission-selector', 'value'),
        Input('predict-fueltype-selector', 'value'),
        Input('predict-enginesize-selector', 'value'),
        Input('predict-mpg-selector', 'value'),
        Input('predict-mileage-selector', 'value')
    ]

)
def predictor_selected_data(brand, model, year, transmission, fueltype, enginesize,mpg,mileage):
    brand = str(brand).lower()
    new_row = [[model, year, transmission, mileage, fueltype, mpg, enginesize]]
    predict_x[brand].newdata(new_row)
    d = predict_x[brand].df 
    c = [{"name": i, "id": i,} for i in (d.columns)] 
    d = d.to_dict('rows')
    return(dash_table.DataTable(data = d, columns = c)) #