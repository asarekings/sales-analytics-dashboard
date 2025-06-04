import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor
import shap
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.models import Sequential
from keras.layers import LSTM, Dense
import base64

st.set_page_config(
    page_title="Advanced Sales Analytics Dashboard",
    layout="wide",
    page_icon=":bar_chart:",
    initial_sidebar_state="expanded"
)

# --- Sidebar Branding & Filters ---
st.sidebar.image("https://i.imgur.com/5RHR6Ku.png", use_column_width=True)
st.sidebar.title("Sales Dashboard Filters")

@st.cache_data
def load_data():
    df = pd.read_csv('sales_data.csv', parse_dates=['OrderDate'])
    return df

df = load_data()
region = st.sidebar.multiselect("Region", options=sorted(df["Region"].unique()), default=list(df["Region"].unique()))
category = st.sidebar.multiselect("Category", options=sorted(df["Category"].unique()), default=list(df["Category"].unique()))
segment = st.sidebar.multiselect("Segment", options=sorted(df["Segment"].unique()), default=list(df["Segment"].unique()))
date_range = st.sidebar.date_input("Order Date Range", [df["OrderDate"].min(), df["OrderDate"].max()])

filtered = df[
    (df["Region"].isin(region)) &
    (df["Category"].isin(category)) &
    (df["Segment"].isin(segment)) &
    (df["OrderDate"] >= pd.to_datetime(date_range[0])) &
    (df["OrderDate"] <= pd.to_datetime(date_range[1]))
]

# --- KPI Cards ---
total_sales = filtered['Sales'].sum()
total_profit = filtered['Profit'].sum()
total_orders = filtered['OrderID'].nunique()
unique_customers = filtered['CustomerID'].nunique()
avg_order_value = total_sales / total_orders if total_orders else 0
repeat_rate = filtered.groupby('CustomerID')['OrderID'].nunique().gt(1).mean()

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total Sales", f"${total_sales:,.2f}")
kpi2.metric("Total Profit", f"${total_profit:,.2f}")
kpi3.metric("Total Orders", f"{total_orders:,}")
kpi4.metric("Unique Customers", f"{unique_customers:,}")
kpi5.metric("Repeat Purchase Rate", f"{repeat_rate:.1%}")

# --- Tabs for Navigation ---
tabs = st.tabs([
    "Overview", "Forecast & Decomposition", "Customer Segments", 
    "Feature Importance (Explainable AI)", "Anomalies", "Advanced", "Raw Data"
])
tab_overview, tab_forecast, tab_segments, tab_shap, tab_anomalies, tab_advanced, tab_raw = tabs

# --- Overview Tab ---
with tab_overview:
    st.markdown("### Overview: Sales Insights & Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        topcat = filtered.groupby('Category')['Sales'].sum().reset_index()
        fig1 = px.bar(topcat, x="Sales", y="Category", orientation='h', title="Sales by Category", color='Category',
                      color_discrete_sequence=px.colors.sequential.Blues_r, template="plotly_white")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        topreg = filtered.groupby('Region')['Sales'].sum().reset_index()
        fig2 = px.bar(topreg, x="Sales", y="Region", orientation='h', title="Sales by Region", color='Region',
                      color_discrete_sequence=px.colors.sequential.Greens_r, template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Sales Trend")
    monthly = filtered.set_index('OrderDate').resample('M').sum(numeric_only=True)['Sales'].reset_index()
    fig3 = px.line(monthly, x='OrderDate', y='Sales', title="Monthly Sales Trend", markers=True, template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### Top 10 Products")
    topprods = filtered.groupby('ProductName')['Sales'].sum().nlargest(10).reset_index()
    fig4 = px.bar(topprods, x="Sales", y="ProductName", orientation='h', color="Sales",
                  color_continuous_scale="mako", template="plotly_white", title="Top 10 Products by Sales")
    st.plotly_chart(fig4, use_container_width=True)

# --- Forecast & Decomposition Tab ---
with tab_forecast:
    st.markdown("### Forecasting & Seasonality Analysis")
    monthly_sales = filtered.set_index('OrderDate').resample('M').sum(numeric_only=True)['Sales']
    monthly_sales = monthly_sales.reset_index()
    if len(monthly_sales) >= 18:
        # SARIMA Forecast
        try:
            sarima_train = monthly_sales.set_index('OrderDate')['Sales']
            model = SARIMAX(sarima_train, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
            sarima_results = model.fit(disp=False)
            forecast = sarima_results.get_forecast(steps=6)
            forecast_index = pd.date_range(monthly_sales['OrderDate'].max() + pd.DateOffset(months=1), periods=6, freq="M")
            forecast_values = forecast.predicted_mean.values

            fig_sarima = go.Figure()
            fig_sarima.add_trace(go.Scatter(x=monthly_sales['OrderDate'], y=monthly_sales['Sales'], mode='lines+markers', name='Actual'))
            fig_sarima.add_trace(go.Scatter(x=forecast_index, y=forecast_values, mode='lines+markers', name='SARIMA Forecast', line=dict(dash='dash', color='orange')))
            fig_sarima.update_layout(title="SARIMA Forecast (Next 6 Months)", template="plotly_white")
            st.plotly_chart(fig_sarima, use_container_width=True)
        except Exception as e:
            st.warning(f"SARIMA forecast unavailable: {e}")
        # LSTM Forecast
        try:
            data = monthly_sales['Sales'].values.astype(np.float32)
            scaler_lstm = StandardScaler()
            data_scaled = scaler_lstm.fit_transform(data.reshape(-1, 1)).flatten()
            def create_sequences(series, seq_length=6):
                X, y = [], []
                for i in range(len(series)-seq_length):
                    X.append(series[i:i+seq_length])
                    y.append(series[i+seq_length])
                return np.array(X), np.array(y)
            seq_length = 6
            X_lstm, y_lstm = create_sequences(data_scaled, seq_length)
            if len(X_lstm) > 0:
                model_lstm = Sequential([
                    LSTM(32, input_shape=(seq_length, 1)),
                    Dense(1)
                ])
                model_lstm.compile(optimizer='adam', loss='mse')
                model_lstm.fit(X_lstm[..., np.newaxis], y_lstm, epochs=50, verbose=0)
                last_seq = data_scaled[-seq_length:]
                preds = []
                input_seq = last_seq.copy()
                for _ in range(6):
                    pred = model_lstm.predict(input_seq[np.newaxis, ..., np.newaxis], verbose=0)
                    preds.append(pred[0,0])
                    input_seq = np.append(input_seq[1:], pred[0,0])
                preds = scaler_lstm.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
                forecast_dates = pd.date_range(monthly_sales['OrderDate'].max() + pd.DateOffset(months=1), periods=6, freq="M")
                fig_lstm = go.Figure()
                fig_lstm.add_trace(go.Scatter(x=monthly_sales['OrderDate'], y=monthly_sales['Sales'], mode='lines+markers', name='Actual'))
                fig_lstm.add_trace(go.Scatter(x=forecast_dates, y=preds, mode='lines+markers', name='LSTM Forecast', line=dict(dash='dot', color='purple')))
                fig_lstm.update_layout(title="LSTM Forecast (Next 6 Months)", template="plotly_white")
                st.plotly_chart(fig_lstm, use_container_width=True)
        except Exception as e:
            st.warning(f"LSTM forecast unavailable: {e}")
        # Seasonality Decomposition
        try:
            decomp = seasonal_decompose(monthly_sales.set_index('OrderDate')['Sales'], model='multiplicative')
            fig_decomp = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                       subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
            fig_decomp.add_trace(go.Scatter(x=decomp.observed.index, y=decomp.observed, name="Observed"), row=1, col=1)
            fig_decomp.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend, name="Trend"), row=2, col=1)
            fig_decomp.add_trace(go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal, name="Seasonal"), row=3, col=1)
            fig_decomp.add_trace(go.Scatter(x=decomp.resid.index, y=decomp.resid, name="Residual"), row=4, col=1)
            fig_decomp.update_layout(height=800, title="Seasonal Decomposition of Monthly Sales", template="plotly_white")
            st.plotly_chart(fig_decomp, use_container_width=True)
        except Exception as e:
            st.warning(f"Seasonality Decomposition unavailable: {e}")
    else:
        st.info("Not enough data for time series analysis (need at least 18 months).")

# --- Customer Segments Tab ---
with tab_segments:
    st.markdown("### Customer Segmentation (RFM + KMeans Clustering)")
    snapshot_date = filtered['OrderDate'].max() + pd.Timedelta(days=1)
    rfm = filtered.groupby('CustomerID').agg({
        'OrderDate': lambda x: (snapshot_date - x.max()).days,
        'OrderID': 'nunique',
        'Sales': 'sum'
    }).rename(columns={'OrderDate': 'Recency', 'OrderID': 'Frequency', 'Sales': 'Monetary'})
    rfm = rfm.fillna(0)
    rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']
    rfm['AvgOrderValue'] = rfm['AvgOrderValue'].replace([np.inf, -np.inf], 0).fillna(0)
    features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue']
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[features])
    try:
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
        fig_km = px.scatter(rfm, x='Recency', y='Monetary', color='Cluster', title="Customer Segments",
                            color_continuous_scale='Set2', template="plotly_white",
                            hover_data=['Frequency', 'AvgOrderValue'])
        st.plotly_chart(fig_km, use_container_width=True)
        st.dataframe(rfm.sort_values("Cluster").head(15))
    except Exception as e:
        st.warning(f"KMeans segmentation unavailable: {e}")

# --- Feature Importance (Explainable AI) Tab ---
with tab_shap:
    st.markdown("### Feature Importance with XGBoost + SHAP (Explainable AI)")
    monthly_agg = filtered.copy()
    monthly_agg['Month'] = monthly_agg['OrderDate'].dt.to_period("M")
    features_df = monthly_agg.groupby(['Month', 'Region', 'Category', 'Segment']).agg(
        Sales=('Sales', 'sum'),
        Profit=('Profit', 'sum'),
        Quantity=('Quantity', 'sum'),
        Orders=('OrderID', 'nunique')
    ).reset_index()
    features_df = pd.get_dummies(features_df, columns=['Region', 'Category', 'Segment'])
    target = features_df['Sales'].values
    X = features_df.drop(['Month', 'Sales'], axis=1)
    X = X.astype(float)
    X = X.fillna(0)
    if X.shape[0] > 10:
        xgb = XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
        xgb.fit(X, target)
        explainer = shap.Explainer(xgb, X)
        shap_values = explainer(X)
        st.write("#### SHAP Summary Plot (Bar)")
        shap.summary_plot(shap_values, X, show=False, plot_type="bar")
        st.pyplot(bbox_inches='tight', use_container_width=True)
        st.write("#### SHAP Summary Plot (Bee Swarm)")
        shap.summary_plot(shap_values, X, show=False)
        st.pyplot(bbox_inches='tight', use_container_width=True)
    else:
        st.info("Not enough data for XGBoost/SHAP analysis.")

# --- Anomalies Tab ---
with tab_anomalies:
    st.markdown("### Advanced Anomaly Detection in Monthly Sales (Isolation Forest)")
    monthly_sales = filtered.set_index('OrderDate').resample('M').sum(numeric_only=True)['Sales'].reset_index()
    if len(monthly_sales) > 12:
        sales_for_anom = monthly_sales[['Sales']].copy()
        clf = IsolationForest(contamination=0.15, random_state=42)
        iforest_pred = clf.fit_predict(sales_for_anom)
        monthly_sales['Anomaly'] = iforest_pred == -1
        fig_anom = px.line(monthly_sales, x='OrderDate', y='Sales', title="Monthly Sales with Anomalies Detected",
                           template="plotly_white")
        fig_anom.add_scatter(x=monthly_sales[monthly_sales['Anomaly']]['OrderDate'],
                             y=monthly_sales[monthly_sales['Anomaly']]['Sales'],
                             mode='markers', marker=dict(color='red', size=12), name='Anomaly')
        st.plotly_chart(fig_anom, use_container_width=True)
    else:
        st.info("Not enough data for anomaly detection.")

# --- Advanced Tab ---
with tab_advanced:
    st.markdown("### Advanced Analytics Toolbox")
    st.write("- **Seasonality Heatmap:** See sales by month and year.")
    filtered['Year'] = filtered['OrderDate'].dt.year
    filtered['Month'] = filtered['OrderDate'].dt.month
    pivot = filtered.pivot_table(index='Month', columns='Year', values='Sales', aggfunc='sum')
    fig_heat = px.imshow(pivot, labels=dict(x="Year", y="Month", color="Sales"), 
                         color_continuous_scale='Viridis', title="Sales Seasonality Heatmap")
    st.plotly_chart(fig_heat, use_container_width=True)
    st.write("- **Cohort Analysis, Association Rules, or other advanced features can be integrated here.**")

    # Downloadable Reports
    st.markdown("#### Download Reports")
    csv = filtered.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown(f'<a href="data:file/csv;base64,{b64}" download="filtered_sales_data.csv">Download Filtered Data as CSV</a>', unsafe_allow_html=True)

# --- Raw Data Tab ---
with tab_raw:
    st.markdown("### Raw Data Preview")
    st.dataframe(filtered.head(100))

# --- Footer ---
st.markdown("""
<style>
footer {visibility: hidden;}
</style>
<hr>
<center>
<small>
Built with <a href="https://streamlit.io/" target="_blank">Streamlit</a> | By Asare K. Enock | Modern Design, Advanced Analytics, Explainable AI
</small>
</center>
""", unsafe_allow_html=True)