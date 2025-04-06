import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

# Load dataset
df = pd.read_csv('domestic.csv')
df_1 = pd.read_csv('domestic_2.csv')
df = pd.merge(df,df_1,how='inner')


# Preprocessing the data
df = df.dropna()
df = df.drop_duplicates()

df.head()

# Feature engineering
df['Distance'] = df['nsmiles']
df['Arrival_City'] = df['citymarketid_2']
df['Departure_City'] = df['citymarketid_1']
df['Passenger_Count'] = df['passengers']
df['Quarter'] = df['quarter']
df['Carrier'] = df['carrier_lg']

# Store the original fare for comparison
df['original_fare'] = df['fare'].copy()
df['fare'] = np.log1p(df['fare'])
df['passengers'] = np.log1p(df['Distance'])
df['Distance'] = np.log1p(df['nsmiles'])

# Route popularity: count of flights for each route (Departure_City -> Arrival_City)
df['Route_Popularity'] = df.groupby(['Departure_City', 'Arrival_City'])['Distance'].transform('count')
df['Route_Popularity'] = df['Route_Popularity'] / df['Route_Popularity'].max()

# One-hot encoding the 'Carrier' column
df = pd.get_dummies(df, columns=['Carrier'], drop_first=True)

# Outlier removal using IQR (Interquartile Range)
numeric_cols = df.select_dtypes(include=["number"]).columns
Q1 = df[numeric_cols].quantile(0.25, numeric_only=True)
Q3 = df[numeric_cols].quantile(0.75, numeric_only=True)
IQR = Q3 - Q1
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df = df[~((df[numeric_cols] < (Q1 - 4 * IQR)) | (df[numeric_cols] > (Q3 + 4 * IQR))).any(axis=1)]

# Features (X) and target (y)
X = df[["Distance", "Quarter", "Year", "Route_Popularity"] + [col for col in df.columns if "Carrier_" in col]]
y = df["fare"]

# Fit the transformers on the training data
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)

# Train a linear regression model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_poly_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Save the model and transformers
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
    
with open("poly.pkl", "wb") as f:
    pickle.dump(poly, f)
    
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Load model & transformers (for prediction)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
    
with open("poly.pkl", "rb") as f:
    poly = pickle.load(f)
    
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("Welcome to our Flight Price Prediction Model!")
st.markdown("Made by Noah Pendo & Ian Beeck")

# Create tabs for prediction and model evaluation
tab1, tab2 = st.tabs(["Predict Prices", "Model Evaluation"])

with tab1:
    # Flight prediction widget
    flight_date = st.date_input("Select Flight Date")
    year = flight_date.year
    quarter = (flight_date.month - 1) // 3 + 1

    # Create a mapping between city names and city market IDs
    city_to_citymarketid = dict(zip(df['city1'], df['citymarketid_1']))
    city_to_citymarketid.update(dict(zip(df['city2'], df['citymarketid_2'])))

    citymarketid_to_city = {v: k for k, v in city_to_citymarketid.items()}

    # Get unique departure cities (using city1 column)
    departure_cities = sorted(df['city1'].unique())

    # User selects departure city by name
    selected_departure_city_name = st.selectbox("Select Departure City", departure_cities)

    # Get the city market ID for the selected departure city
    selected_departure_citymarketid = city_to_citymarketid.get(selected_departure_city_name)

    # Get available arrival cities based on the selected departure city
    available_routes = df[df['citymarketid_1'] == selected_departure_citymarketid]
    available_arrival_cities = sorted(available_routes['city2'].unique())

    # User selects arrival city by name from filtered options
    selected_arrival_city_name = st.selectbox("Select Arrival City", available_arrival_cities)

    # Get the city market ID for the selected arrival city
    selected_arrival_citymarketid = city_to_citymarketid.get(selected_arrival_city_name)

    # Get distance from the dataset for the selected route
    route_filter = (df["citymarketid_1"] == selected_departure_citymarketid) & (df["citymarketid_2"] == selected_arrival_citymarketid)
    if df[route_filter].empty:
        st.error("No available data for this route.")
        st.stop()

    distance = df[route_filter]["nsmiles"].median()  # Use median distance if multiple entries exist
    route_popularity = df[route_filter]["Route_Popularity"].median()

    # Get carrier information
    carriers = df[route_filter]["carrier_lg"].unique()
    carrier_info = ""
    if len(carriers) > 0:
        carrier_info = f"Airlines serving this route: {', '.join(carriers)}"
        st.info(carrier_info)

    # Allow user to select a carrier if available
    selected_carrier = None
    if len(carriers) > 0:
        selected_carrier = st.selectbox("Select Airline", carriers)

    # Create input DataFrame for prediction
    input_data = pd.DataFrame({
        "Distance": [np.log1p(distance)],
        "Quarter": [quarter],
        "Year": [year],
        "Route_Popularity": [route_popularity]
    })

    # Add carrier columns (all zeros initially)
    carrier_columns = [col for col in X.columns if "Carrier_" in col]
    for col in carrier_columns:
        input_data[col] = 0

    # Set the selected carrier to 1 if available
    if selected_carrier:
        carrier_col = f"Carrier_{selected_carrier}"
        if carrier_col in carrier_columns:
            input_data[carrier_col] = 1

    # Prediction
    if st.button("Predict Flight Price"):
        # Transform input for prediction
        X_poly_input = poly.transform(input_data)
        X_scaled_input = scaler.transform(X_poly_input)
        
        prediction = model.predict(X_scaled_input)
        predicted_fare = np.expm1(prediction[0])
        
        st.success(f"Predicted Log Fare: {prediction[0]:.2f}")
        st.metric(label="Estimated Fare ($)", value=f"${predicted_fare:.2f}")
        
        # Display route information
        st.write(f"Route: {selected_departure_city_name} to {selected_arrival_city_name}")
        st.write(f"Distance: {distance:.1f} miles")
        st.write(f"Quarter: {quarter}, Year: {year}")
        if selected_carrier:
            st.write(f"Airline: {selected_carrier}")
        
        # Compare with actual fares
        route_data = df[route_filter].copy()
        
        # Calculate actual fares from log-transformed values
        actual_fares = route_data['original_fare']
        
        # Create comparison statistics
        min_fare = actual_fares.min()
        max_fare = actual_fares.max()
        avg_fare = actual_fares.mean()
        median_fare = actual_fares.median()
        
        # Display comparison
        st.subheader("Fare Comparison")
        
        # Create columns for comparison metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Min Actual Fare", f"${min_fare:.2f}")
        col2.metric("Avg Actual Fare", f"${avg_fare:.2f}")
        col3.metric("Median Actual Fare", f"${median_fare:.2f}")
        col4.metric("Max Actual Fare", f"${max_fare:.2f}")
        
        # Calculate and display price difference
        price_diff = predicted_fare - avg_fare
        percent_diff = (price_diff / avg_fare) * 100
        
        st.subheader("Price Analysis")
        if price_diff > 0:
            st.warning(f"Your predicted fare is ${price_diff:.2f} higher than the average fare ({percent_diff:.1f}% higher)")
        elif price_diff < 0:
            st.success(f"Your predicted fare is ${abs(price_diff):.2f} lower than the average fare ({abs(percent_diff):.1f}% lower)")
        else:
            st.info("Your predicted fare matches the average fare exactly")
        
        # Create histogram of actual fares
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(actual_fares, bins=20, alpha=0.7)
        ax.axvline(predicted_fare, color='red', linestyle='dashed', linewidth=2, label='Predicted Fare')
        ax.axvline(avg_fare, color='green', linestyle='dashed', linewidth=2, label='Average Fare')
        ax.set_xlabel('Fare ($)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Fare Distribution: {selected_departure_city_name} to {selected_arrival_city_name}')
        ax.legend()
        
        st.pyplot(fig)
        
        # Show a sample of actual fares in a table with quarter info
        st.subheader("Sample of Actual Fares")
        fare_sample = route_data[['original_fare', 'quarter', 'Year']].sample(min(5, len(route_data)))
        fare_sample = fare_sample.rename(columns={'original_fare': 'Fare ($)', 'quarter': 'Quarter'})
        st.dataframe(fare_sample)
        
        # Add date sensitivity analysis
        st.subheader("Fare Sensitivity to Date")
        
        # Create test dates to show sensitivity
        test_quarters = [1, 2, 3, 4]
        test_years = [year - 1, year, year + 1]
        
        sensitivity_data = []
        
        for test_year in test_years:
            for test_quarter in test_quarters:
                # Create test input
                test_input = input_data.copy()
                test_input['Quarter'] = test_quarter
                test_input['Year'] = test_year
                
                # Predict fare
                test_poly = poly.transform(test_input)
                test_scaled = scaler.transform(test_poly)
                test_prediction = model.predict(test_scaled)
                test_fare = np.expm1(test_prediction[0])
                
                sensitivity_data.append({
                    'Quarter': test_quarter,
                    'Year': test_year,
                    'Predicted Fare': test_fare
                })
        
        # Create dataframe of sensitivity analysis
        sensitivity_df = pd.DataFrame(sensitivity_data)
        
        # Display sensitivity analysis
        st.write("How fares change across different quarters and years:")
        st.dataframe(sensitivity_df)
        
        # Create a line chart showing fare by quarter for each year
        sensitivity_pivot = sensitivity_df.pivot(index='Quarter', columns='Year', values='Predicted Fare')
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for year in test_years:
            ax2.plot(sensitivity_pivot.index, sensitivity_pivot[year], marker='o', linewidth=2, label=f'Year {year}')
        
        ax2.set_xlabel('Quarter')
        ax2.set_ylabel('Predicted Fare ($)')
        ax2.set_title('Fare Sensitivity to Quarter and Year')
        ax2.legend()
        ax2.grid(True)
        
        st.pyplot(fig2)

with tab2:
    st.header("Model Evaluation")
    st.write("This section evaluates the performance of the fare prediction model.")
    
    # Calculate predictions on the entire dataset
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Convert log predictions back to original scale
    y_train_orig = np.expm1(y_train)
    y_test_orig = np.expm1(y_test)
    y_pred_train_orig = np.expm1(y_pred_train)
    y_pred_test_orig = np.expm1(y_pred_test)
    
    # Calculate metrics on log scale
    train_mse = mean_squared_error(y_train, y_pred_train)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_pred_train)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Calculate metrics on original dollar scale
    train_mse_orig = mean_squared_error(y_train_orig, y_pred_train_orig)
    train_rmse_orig = np.sqrt(train_mse_orig)
    train_r2_orig = r2_score(y_train_orig, y_pred_train_orig)
    train_mae_orig = mean_absolute_error(y_train_orig, y_pred_train_orig)
    
    test_mse_orig = mean_squared_error(y_test_orig, y_pred_test_orig)
    test_rmse_orig = np.sqrt(test_mse_orig)
    test_r2_orig = r2_score(y_test_orig, y_pred_test_orig)
    test_mae_orig = mean_absolute_error(y_test_orig, y_pred_test_orig)
    
    # Display metrics
    st.subheader("Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Log Scale Metrics")
        st.markdown("#### Training Data")
        metrics_df_train = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
            'Value': [train_mse, train_rmse, train_mae, train_r2]
        })
        st.dataframe(metrics_df_train)
        
        st.markdown("#### Test Data")
        metrics_df_test = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
            'Value': [test_mse, test_rmse, test_mae, test_r2]
        })
        st.dataframe(metrics_df_test)
    
    with col2:
        st.markdown("### Dollar Scale Metrics")
        st.markdown("#### Training Data")
        metrics_df_train_orig = pd.DataFrame({
            'Metric': ['MSE ($)', 'RMSE ($)', 'MAE ($)', 'R²'],
            'Value': [train_mse_orig, train_rmse_orig, train_mae_orig, train_r2_orig]
        })
        st.dataframe(metrics_df_train_orig)
        
        st.markdown("#### Test Data")
        metrics_df_test_orig = pd.DataFrame({
            'Metric': ['MSE ($)', 'RMSE ($)', 'MAE ($)', 'R²'],
            'Value': [test_mse_orig, test_rmse_orig, test_mae_orig, test_r2_orig]
        })
        st.dataframe(metrics_df_test_orig)
    
    # Perform cross-validation
    st.subheader("Cross-Validation Results")
    cv_scores = cross_val_score(model, X_poly_scaled, y, cv=5, scoring='r2')
    
    cv_df = pd.DataFrame({
        'Fold': range(1, 6),
        'R² Score': cv_scores
    })
    
    st.dataframe(cv_df)
    st.write(f"Mean R² from cross-validation: {cv_scores.mean():.4f} (std: {cv_scores.std():.4f})")
    
    # Residual analysis
    st.subheader("Residual Analysis")
    
    # Calculate residuals
    residuals = y_test - y_pred_test
    
    # Plot residuals
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.scatter(y_pred_test, residuals, alpha=0.5)
    ax3.axhline(y=0, color='r', linestyle='-')
    ax3.set_xlabel('Predicted Values (Log Scale)')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residual Plot')
    ax3.grid(True)
    
    st.pyplot(fig3)
    
    # Actual vs Predicted Plot
    st.subheader("Actual vs Predicted Fares")
    
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.scatter(y_test, y_pred_test, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax4.set_xlabel('Actual Fare (Log Scale)')
    ax4.set_ylabel('Predicted Fare (Log Scale)')
    ax4.set_title('Actual vs Predicted Values')
    ax4.grid(True)
    
    st.pyplot(fig4)
    
    # Feature importance
    st.subheader("Feature Importance")
    
    # Get feature names from polynomial features
    poly_features = poly.get_feature_names_out(X.columns)
    
    # Get coefficients from the model
    coefficients = model.coef_
    
    # Create a dataframe of feature importance
    feature_importance = pd.DataFrame({
        'Feature': poly_features,
        'Coefficient': coefficients
    })
    
    # Sort by absolute coefficient value
    feature_importance['Absolute_Coefficient'] = np.abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values('Absolute_Coefficient', ascending=False).head(20)
    
    # Plot top features
    fig5, ax5 = plt.subplots(figsize=(12, 8))
    ax5.barh(feature_importance['Feature'], feature_importance['Coefficient'])
    ax5.set_xlabel('Coefficient')
    ax5.set_title('Top 20 Feature Importance')
    ax5.grid(True)
    
    st.pyplot(fig5)
    
    # Model summary
    st.subheader("Model Summary")
    
    # Calculate percentile errors
    errors = np.abs(y_test_orig - y_pred_test_orig)
    percentile_25 = np.percentile(errors, 25)
    percentile_50 = np.percentile(errors, 50)
    percentile_75 = np.percentile(errors, 75)
    percentile_95 = np.percentile(errors, 95)
    
    # Create a summary table
    st.markdown("### Error Distribution (Original Scale)")
    error_df = pd.DataFrame({
        'Percentile': ['25%', '50% (Median)', '75%', '95%'],
        'Error ($)': [percentile_25, percentile_50, percentile_75, percentile_95]
    })
    
    st.dataframe(error_df)
    
    st.markdown("### Final Assessment")
    st.write(f"""
    - The model explains {train_r2:.2%} of the variance in the training data and {test_r2:.2%} in the test data.
    - On average, predictions are off by ${test_mae_orig:.2f} from actual fares.
    - 75% of predictions are within ${percentile_75:.2f} of the actual fare.
    - Key factors influencing fare prices include distance, carrier, and route popularity.
    - The model shows {cv_scores.mean():.2%} average R² in cross-validation, indicating good generalization.
    """)
    
    # Download the evaluation report
    evaluation_text = f"""
    # Flight Price Prediction Model Evaluation
    
    ## Performance Metrics
    
    ### Log Scale Metrics
    - Training MSE: {train_mse:.4f}
    - Training RMSE: {train_rmse:.4f}
    - Training MAE: {train_mae:.4f}
    - Training R²: {train_r2:.4f}
    
    - Test MSE: {test_mse:.4f}
    - Test RMSE: {test_rmse:.4f}
    - Test MAE: {test_mae:.4f}
    - Test R²: {test_r2:.4f}
    
    ### Dollar Scale Metrics
    - Training MSE: ${train_mse_orig:.2f}
    - Training RMSE: ${train_rmse_orig:.2f}
    - Training MAE: ${train_mae_orig:.2f}
    - Training R²: {train_r2_orig:.4f}
    
    - Test MSE: ${test_mse_orig:.2f}
    - Test RMSE: ${test_rmse_orig:.2f}
    - Test MAE: ${test_mae_orig:.2f}
    - Test R²: {test_r2_orig:.4f}
    
    ## Cross-Validation Results
    - Mean R²: {cv_scores.mean():.4f}
    - R² Standard Deviation: {cv_scores.std():.4f}
    
    ## Error Distribution
    - 25% of errors are below ${percentile_25:.2f}
    - 50% of errors are below ${percentile_50:.2f}
    - 75% of errors are below ${percentile_75:.2f}
    - 95% of errors are below ${percentile_95:.2f}
    
    ## Final Assessment
    - The model explains {train_r2:.2%} of the variance in the training data and {test_r2:.2%} in the test data.
    - On average, predictions are off by ${test_mae_orig:.2f} from actual fares.
    - 75% of predictions are within ${percentile_75:.2f} of the actual fare.
    - Key factors influencing fare prices include distance, carrier, and route popularity.
    - The model shows {cv_scores.mean():.2%} average R² in cross-validation, indicating good generalization.
    """
    
    st.download_button(
        label="Download Evaluation Report",
        data=evaluation_text,
        file_name="model_evaluation_report.md",
        mime="text/markdown"
    )