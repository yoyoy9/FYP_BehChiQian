import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import joblib
from datetime import datetime, timedelta
from scipy.special import inv_boxcox

# Set page configuration
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Load the XGBoost Model
@st.cache_resource
def load_xgboost_model():
    xgboost_model_path = 'xgboost_model.pkl'
    xgboost_model = joblib.load(xgboost_model_path)
    return xgboost_model

xgboost_model = load_xgboost_model()

# Load Data
@st.cache_data
def load_xgboost_data():
    X_train = pd.read_csv('X_train_xgboost.csv', index_col=0)
    y_train = pd.read_csv('y_train_xgboost.csv', index_col=0)
    X_test = pd.read_csv('X_test_xgboost.csv', index_col=0)
    y_test = pd.read_csv('y_test_xgboost.csv', index_col=0)
    
    # Ensure the index is properly converted to datetime
    X_train.index = pd.to_datetime(X_train.index)
    y_train.index = pd.to_datetime(y_train.index)
    X_test.index = pd.to_datetime(X_test.index)
    y_test.index = pd.to_datetime(y_test.index)
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_xgboost_data()

# Load the lambda value for Box-Cox transformation
boxcox_lambda = 0.1436823782945898  # Replace with your actual lambda value

# ============================
# Function to Inverse Box-Cox Transformation
# ============================

def inverse_boxcox(y_transformed, lambda_value, shift=0):
    # Apply the inverse Box-Cox transformation
    y_original = inv_boxcox(y_transformed, lambda_value)
    
    # Adjust for the shift applied before the original transformation
    return y_original - shift


# ============================
# Generate forecast chart with confidence intervals
# ============================
def generate_forecast_chart(dates, actual_sales, predicted_sales, title, lower_bound=None, upper_bound=None, theme='plotly_white', actual_color='green', predicted_color='blue'):
    fig = go.Figure()

    # Make sure dates are properly formatted as datetime objects for Plotly
    dates = pd.to_datetime(dates)

    # Plot actual sales
    fig.add_trace(go.Scatter(x=dates, y=actual_sales, mode='lines+markers', name='Actual Sales', line=dict(color=actual_color), connectgaps=True))

    # Plot predicted sales
    fig.add_trace(go.Scatter(x=dates, y=predicted_sales, mode='lines+markers', name='Predicted Sales', line=dict(color=predicted_color), connectgaps=True))

    # Plot confidence intervals if provided
    if lower_bound is not None and upper_bound is not None:
        fig.add_trace(go.Scatter(x=dates, y=upper_bound, mode='lines', name='Upper Bound', line=dict(color='lightblue'), fill=None))
        fig.add_trace(go.Scatter(x=dates, y=lower_bound, mode='lines', name='Lower Bound', line=dict(color='lightblue'), fill='tonexty'))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Total Sales',
        template=theme,
        hovermode='x unified',
        autosize=True,
        height=350,  # Reduce height for better mobile display
    )

    fig.update_xaxes(rangeslider_visible=True)
    return fig


def calculate_errors(actual_sales, predicted_sales):
    if len(actual_sales) != len(predicted_sales):
        st.error("Mismatch in the length of actual and predicted data.")
        return None, None, None

    mse = mean_squared_error(actual_sales, predicted_sales)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_sales, predicted_sales)
    mean_actual = np.mean(actual_sales)
    rmse_pct = (rmse / mean_actual) * 100
    mae_pct = (mae / mean_actual) * 100
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual_sales - predicted_sales) / actual_sales)) * 100
    
    return {'MAE (%)': mae_pct, 'RMSE (%)': rmse_pct, 'MAPE (%)': mape}


def train_and_predict(X_train, y_train, X_test, y_test, days_to_forecast, boxcox_lambda):

    # Define future dates using the last date from y_test
    if isinstance(y_test.index[-1], datetime):
        last_date = y_test.index[-1]
    else:
        last_date = pd.to_datetime(y_test.index[-1])
   
    last_date = y_test.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_forecast, freq='D')
    
    # Fit the model and make predictions
    xgb_model = XGBRegressor().fit(X_train, y_train)
    y_pred_test_transformed = xgb_model.predict(X_test)

    # Ensure predictions are generated
    if y_pred_test_transformed is None or len(y_pred_test_transformed) == 0:
        st.error("XGBoost prediction failed.")
        return None, None, None, None, None
    
    y_test_display = inverse_boxcox(y_test.values.flatten(), boxcox_lambda)
    y_pred_test_display = inverse_boxcox(y_pred_test_transformed, boxcox_lambda)

    # Generate future data for forecasting
    future_X = pd.DataFrame(index=future_dates)
    for col in X_test.columns:
        future_X[col] = np.linspace(X_test[col].iloc[-1], X_test[col].iloc[-1] * 1.1, num=days_to_forecast)  # Simplified extrapolation
        
    y_pred_future_transformed = xgb_model.predict(future_X)
    y_pred_future_display = inverse_boxcox(y_pred_future_transformed, boxcox_lambda)
    
    # Generate Confidence Intervals (95%)
    ci_margin = 1.96 * np.std(y_pred_future_display)
    lower_bound = y_pred_future_display - ci_margin
    upper_bound = y_pred_future_display + ci_margin

    return y_test_display, y_pred_test_display, y_pred_future_display, X_test.index, future_dates, y_pred_test_transformed, lower_bound, upper_bound
def show_performance_metrics(metrics):
    st.subheader('📊 Performance Metrics of XGBoost')
    # Pass an index to the DataFrame to avoid the error
    metrics_df = pd.DataFrame([metrics], index=['Values'])
    st.table(metrics_df)

    
# Streamlit Interface
st.title("📈 Fashion Sales Forecasting Dashboard - XGBoost")
st.markdown("""
This application provides sales forecasts for a men's product from Oxwhite, 
specifically for the **Men's Premium Weight Cotton Crew Neck Tee**. The data used 
for this model includes product details, sales history, and other related factors 
to predict future sales. """)

st.sidebar.header("🔧 Model & Forecast Settings")
days_to_forecast = st.sidebar.number_input("Days to Forecast", min_value=1, max_value=365, value=30)
actual_color = st.sidebar.color_picker("Select Actual Sales Color", "#00FF00")
predicted_color = st.sidebar.color_picker("Select Predicted Sales Color", "#0000FF")

# Run Forecast button
if st.sidebar.button("Run Forecast"):
    y_test_display, y_pred_test_display, y_pred_future_display, test_indices, future_dates, y_pred_test_transformed, lower_bound, upper_bound = train_and_predict(
        X_train, y_train, X_test, y_test, days_to_forecast, boxcox_lambda)

    if y_test_display is not None and y_pred_test_display is not None:
        actual_vs_predicted = generate_forecast_chart(
            test_indices, y_test_display, y_pred_test_display, 
            'XGBoost Actual vs Predicted', actual_color=actual_color, predicted_color=predicted_color
        )
        st.plotly_chart(actual_vs_predicted)

        future_forecast_fig = generate_forecast_chart(
            future_dates, [None] * len(future_dates), y_pred_future_display,
            f"{days_to_forecast} Days Future Sales Forecast", lower_bound=lower_bound, upper_bound=upper_bound,
            actual_color=actual_color, predicted_color=predicted_color
        )
        st.plotly_chart(future_forecast_fig)

        # Display forecast table with AgGrid filtering, sorting, and searching
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecasted Sales': y_pred_future_display,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound
        })
        st.subheader('Forecasted Sales Table')
        st.dataframe(forecast_df)

        # Download option for forecast data
        csv = forecast_df.to_csv(index=False)
        st.download_button("Download Forecast Data", csv, "forecast_data.csv", "text/csv")

    # Show performance metrics
    metrics = calculate_errors(y_test.values.flatten(), y_pred_test_transformed)
    show_performance_metrics(metrics)
