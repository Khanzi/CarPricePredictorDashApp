# Car Price Predictor Dash App

In this project I wanted to create a [shiny] like app in python use [Plotly Dash]().

The end goal is to have a lightweight web app where you can explore and visualize datasets from the famous [cars dataset]() 
as well as use some pre-trained `sklearn` models to predict the price of a hypothetical car.

### Known Issues

* [ ] Models implement the `tax` column for predictions. Update models to get rid of this
  * It's a circular logic: You won't know the tax of a vehicle before you know the price
