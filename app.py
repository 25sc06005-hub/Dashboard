import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Enterprise Call Management Analytics",
    page_icon="📞",
    layout="wide"
)

st.title("Enterprise Call Management Analytics & Forecasting Hub")
st.markdown("An end-to-end data science platform tracking operations, SLA compliance, workforce timing heatmaps, and ML-driven demand forecasting.")





# Custom Theme-Adaptive CSS for Prominent Navigation Tabs
st.markdown("""
    <style>
        /* 1. Global background wrapper for the tab row */
        div[data-testid="stTabs"] {
            background-color: var(--secondary-background-color) !important;
            padding: 8px 12px 0px 12px;
            border-radius: 8px;
            border: 1px solid rgba(128, 128, 128, 0.2);
            margin-bottom: 25px;
        }
        
        /* 2. Style for all Tab items */
        div[data-testid="stTabs"] button {
            font-size: 16px !important;
            font-weight: 600 !important;
            color: var(--text-color) !important;
            background-color: transparent !important;
            border: none !important;
            padding: 12px 24px !important;
            margin-right: 5px !important;
            border-radius: 6px 6px 0px 0px !important;
            opacity: 0.7; /* Slightly dims unselected tabs for clear hierarchy */
            transition: all 0.2s ease-in-out;
        }

        /* 3. Hover effects on the tabs */
        div[data-testid="stTabs"] button:hover {
            color: #007BFF !important;
            background-color: rgba(128, 128, 128, 0.1) !important;
            opacity: 1;
        }

        /* 4. Highlight styling for the ACTIVE selected tab */
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #FFFFFF !important; /* Explicitly white text for high contrast on blue button */
            background-color: #007BFF !important;
            font-weight: 700 !important;
            opacity: 1;
            box-shadow: 0px 4px 10px rgba(0, 123, 255, 0.25);
        }
        
        /* 5. Remove Streamlit's default bottom accent line */
        div[data-testid="stTabs"] [data-baseweb="tab-highlight-bar"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)




# File path configuration
EXCEL_FILE = "Call_Mgmt_data.xlsx"

if os.path.exists(EXCEL_FILE):
    @st.cache_data
    def load_data(file_path):
        try:
            df = pd.read_excel(file_path)
        except Exception:
            df = pd.read_csv(file_path)
            
        # Clean column spaces
        df.columns = df.columns.str.strip()
        
        # Enforce dayfirst=True to handle DD/MM/YYYY formats accurately
        if 'CREATED_DATE' in df.columns:
            df['CREATED_DATE'] = pd.to_datetime(df['CREATED_DATE'], dayfirst=True, format='mixed', errors='coerce')
        if 'ASSIGN_DATE' in df.columns:
            df['ASSIGN_DATE'] = pd.to_datetime(df['ASSIGN_DATE'], dayfirst=True, format='mixed', errors='coerce')
            
        # Drop rows where critical creation date is missing
        df = df.dropna(subset=['CREATED_DATE'])
            
        # --- Feature Engineering ---
        if 'ASSIGN_DATE' in df.columns:
            # Calculate resolution/handling time in minutes
            df['Resolution_Time_Mins'] = (df['ASSIGN_DATE'] - df['CREATED_DATE']).dt.total_seconds().abs() / 60
            
            # Calculate age of open tickets relative to the latest data point in the dataset
            max_dataset_date = df['CREATED_DATE'].max()
            df['Ticket_Age_Days'] = (max_dataset_date - df['CREATED_DATE']).dt.total_seconds() / (24 * 3600)
            df['Ticket_Age_Days'] = df['Ticket_Age_Days'].clip(lower=0).round(1)
        else:
            df['Resolution_Time_Mins'] = 0
            df['Ticket_Age_Days'] = 0
            
        # Time distribution parsing
        df['Hour_of_Day'] = df['CREATED_DATE'].dt.hour
        df['Day_of_Week'] = df['CREATED_DATE'].dt.day_name()
        df['Date_Only'] = df['CREATED_DATE'].dt.date
            
        return df

    try:
        df_raw = load_data(EXCEL_FILE)
        
        # --- 2. Sidebar Filters & Global Settings ---
        st.sidebar.header("🎯 Global Dashboard Filters")
        
        # Financial Year Filter (ACTAG)
        if 'ACTAG' in df_raw.columns:
            years = sorted(df_raw['ACTAG'].dropna().unique())
            selected_years = st.sidebar.multiselect("Select Financial Year", years, default=years)
            df = df_raw[df_raw['ACTAG'].isin(selected_years)]
        else:
            df = df_raw.copy()

        # Date Range Filter (CREATED_DATE)
        if not df['CREATED_DATE'].isnull().all():
            min_date = df['CREATED_DATE'].min().date()
            max_date = df['CREATED_DATE'].max().date()
            if min_date == max_date:
                date_range = (min_date, max_date)
            else:
                date_range = st.sidebar.date_input("Select Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            if len(date_range) == 2:
                df = df[(df['CREATED_DATE'].dt.date >= date_range[0]) & (df['CREATED_DATE'].dt.date <= date_range[1])]

        # Target Control Inputs
        st.sidebar.markdown("---")
        st.sidebar.header("⚙️ Threshold Configurations")
        sla_target = st.sidebar.slider("Target Resolution SLA (Mins)", min_value=5, max_value=120, value=30, step=5)
        anomaly_threshold = st.sidebar.slider("Anomaly Sensitivity (Std Dev)", min_value=1.5, max_value=3.0, value=2.0, step=0.1)

        # --- 3. Platform Tabs Setup ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Executive Overview", 
            "⏱️ SLA Tracking & Operations", 
            "🗓️ Time-Distribution Insights",
            "🔍 Advanced EDA & Search",
            "🤖 Predictive Analytics & ML"
        ])

        # Core global calculations used across tabs
        total_tickets = len(df)
        closed_mask = df['STATUS'].fillna('').str.strip() == 'CLOS' if 'STATUS' in df.columns else pd.Series([False]*total_tickets)
        closed_tickets = len(df[closed_mask])
        open_tickets = total_tickets - closed_tickets
        resolution_rate = (closed_tickets / total_tickets * 100) if total_tickets > 0 else 0

        # ==========================================
        # TAB 1: EXECUTIVE OVERVIEW
        # ==========================================
        with tab1:
            st.markdown("### 📊 Operational Key Metrics")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Logged Tickets", f"{total_tickets:,}")
            kpi2.metric("Closed Tickets", f"{closed_tickets:,}", delta=f"{closed_tickets} resolved")
            kpi3.metric("Open / Pending Tickets", f"{open_tickets:,}", delta=f"{open_tickets} active", delta_color="inverse")
            kpi4.metric("Resolution Rate (%)", f"{resolution_rate:.1f}%")

            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📈 Daily Ticket Generation Trend")
                trend_df = df.groupby('Date_Only').size().reset_index(name='Tickets Logged')
                fig_trend = px.line(trend_df, x='Date_Only', y='Tickets Logged', markers=True, template="plotly_white", color_discrete_sequence=["#007BFF"])
                st.plotly_chart(fig_trend, use_container_width=True)

            with col2:
                st.markdown("#### ⚙️ Infrastructure Breakdown (Hardware vs Software)")
                if 'CALL_TYPE' in df.columns and not df['CALL_TYPE'].isnull().all():
                    type_counts = df['CALL_TYPE'].value_counts().reset_index()
                    type_counts.columns = ['Call Type', 'Count']
                    fig_type = px.pie(type_counts, values='Count', names='Call Type', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig_type, use_container_width=True)

        # ==========================================
        # TAB 2: SLA TRACKING & OPERATIONS
        # ==========================================
        with tab2:
            st.markdown("### ⏱️ Service Level Agreement (SLA) Performance")
            closed_df = df[closed_mask].copy()
            open_df = df[~closed_mask].copy()

            if len(closed_df) > 0:
                closed_df['SLA_Status'] = closed_df['Resolution_Time_Mins'].apply(lambda x: 'Met' if x <= sla_target else 'Breached')
                sla_met_count = len(closed_df[closed_df['SLA_Status'] == 'Met'])
                sla_compliance_rate = (sla_met_count / len(closed_df)) * 100
                avg_resolution_time = closed_df['Resolution_Time_Mins'].mean()
            else:
                sla_compliance_rate, avg_resolution_time = 100.0, 0.0

            s_kpi1, s_kpi2, s_kpi3 = st.columns(3)
            s_kpi1.metric("SLA Compliance Rate (%)", f"{sla_compliance_rate:.1f}%")
            s_kpi2.metric("Avg. Resolution Time", f"{avg_resolution_time:.1f} Mins")
            s_kpi3.metric("Total Open Backlog", f"{open_tickets:,}")

            st.markdown("---")
            scol1, scol2 = st.columns(2)

            with scol1:
                st.markdown("#### 🚨 SLA Breach Analysis by Call Category")
                if len(closed_df) > 0 and 'CALL_TYPE' in closed_df.columns:
                    sla_breakdown = closed_df.groupby(['CALL_TYPE', 'SLA_Status']).size().reset_index(name='Count')
                    fig_sla = px.bar(sla_breakdown, x='CALL_TYPE', y='Count', color='SLA_Status', barmode='group',
                                     color_discrete_map={'Met': '#2ca02c', 'Breached': '#d62728'}, template="plotly_white")
                    st.plotly_chart(fig_sla, use_container_width=True)

            with scol2:
                st.markdown("#### Open Ticket Backlog Aging Bar Chart")
                if len(open_df) > 0:
                    bins = [-1, 1, 3, 7, 30, float('inf')]
                    labels = ['0-1 Day (New)', '1-3 Days', '3-7 Days', '7-30 Days', '30+ Days (Critical)']
                    open_df['Age_Bucket'] = pd.cut(open_df['Ticket_Age_Days'], bins=bins, labels=labels)
                    age_counts = open_df['Age_Bucket'].value_counts().reindex(labels).reset_index(name='Active Count')
                    fig_age = px.bar(age_counts, x='Age_Bucket', y='Active Count', template="plotly_white", color='Active Count', color_continuous_scale="Reds")
                    st.plotly_chart(fig_age, use_container_width=True)
                else:
                    st.success("🎉 Outstanding! There are no open or pending tickets in the backlog.")

        # ==========================================
        # TAB 3: INTERACTIVE TIME-DISTRIBUTION INSIGHTS
        # ==========================================
        with tab3:
            st.markdown("### 🗓️ Workforce Timing & Heatmap Analytics")
            st.markdown("Understand when incoming volumes peak to optimize staffing arrangements.")

            # Sort days of week standard order
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            # Pivot table calculation for density heatmap
            pivot_df = df.groupby(['Day_of_Week', 'Hour_of_Day']).size().reset_index(name='Ticket Volume')
            
            # Reindex to guarantee correct visual structure
            pivot_matrix = pivot_df.pivot(index='Day_of_Week', columns='Hour_of_Day', values='Ticket Volume').reindex(days_order).fillna(0)

            st.markdown("#### 🌡️ Operational Heatmap: Day of Week vs. Hour of Day")
            fig_heatmap = px.imshow(
                pivot_matrix,
                labels=dict(x="Hour of Day (24h)", y="Day of Week", color="Tickets Received"),
                x=pivot_matrix.columns,
                y=pivot_matrix.index,
                color_continuous_scale="YlOrRd",
                aspect="auto",
                template="plotly_white"
            )
            fig_heatmap.update_xaxes(side="bottom", dtick=1)
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # Secondary breakdown metrics
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.markdown("#### 📊 Aggregate Volume by Hour")
                hourly_volume = df.groupby('Hour_of_Day').size().reset_index(name='Count')
                fig_hour_bar = px.bar(hourly_volume, x='Hour_of_Day', y='Count', template="plotly_white", color_discrete_sequence=["#7F7F7F"])
                fig_hour_bar.update_xaxes(dtick=1)
                st.plotly_chart(fig_hour_bar, use_container_width=True)
            with t_col2:
                st.markdown("#### 📊 Aggregate Volume by Day of Week")
                day_volume = df.groupby('Day_of_Week').size().reindex(days_order).reset_index(name='Count')
                fig_day_bar = px.bar(day_volume, x='Day_of_Week', y='Count', template="plotly_white", color_discrete_sequence=["#17BECF"])
                st.plotly_chart(fig_day_bar, use_container_width=True)

        # ==========================================
        # TAB 4: ADVANCED EDA & SEARCH HUB
        # ==========================================
        with tab4:
            st.markdown("### 🔍 Custom Multi-Variable Explorer Engine")
            eda_col1, eda_col2, eda_col3 = st.columns(3)
            available_dims = [c for c in ['CALL_TYPE', 'CALL_SUB_TYPE', 'STATUS', 'ACTAG'] if c in df.columns]
            
            with eda_col1:
                groupby_col = st.selectbox("Dimension Target (X-Axis)", available_dims, index=0)
            with eda_col2:
                metric_col = st.selectbox("Metric Formula (Y-Axis)", ["Ticket Volume (Count)", "Average Resolution Time (Mins)"])
            with eda_col3:
                chart_style = st.radio("Chart Representation Mode", ["Bar Chart", "Pie Chart"], horizontal=True)

            if metric_col == "Ticket Volume (Count)":
                summary_df = df.groupby(groupby_col).size().reset_index(name='Value')
                ylabel = "Total Ticket Count"
            else:
                summary_df = df.groupby(groupby_col)['Resolution_Time_Mins'].mean().reset_index(name='Value')
                ylabel = "Average Resolution Speed (Minutes)"

            if chart_style == "Bar Chart":
                fig_custom = px.bar(summary_df.sort_values(by='Value', ascending=False), x=groupby_col, y='Value', 
                                    template="plotly_white", color='Value', color_continuous_scale="Viridis", labels={'Value': ylabel})
            else:
                fig_custom = px.pie(summary_df, values='Value', names=groupby_col, hole=0.3, color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig_custom, use_container_width=True)

            st.markdown("---")
            st.markdown("#### 🕵️ Phrase & Keyword Mining Intelligence")
            if 'CALL_DESC' in df.columns:
                search_query = st.text_input("Search descriptions (e.g., 'backup', 'printer', 'reset', 'oracle'):", "").strip()
                if search_query:
                    filtered_search_df = df[df['CALL_DESC'].fillna('').str.lower().str.contains(search_query.lower())]
                    st.success(f"Found **{len(filtered_search_df):,} matching entries** for term: '{search_query}'")
                    st.dataframe(filtered_search_df, use_container_width=True, hide_index=True)

        # ==========================================
        # TAB 5: PREDICTIVE ANALYTICS & MACHINE LEARNING
        # ==========================================
        with tab5:
            st.markdown("### 🤖 Predictive Intelligence & Statistical Modeling")
            
            # Prepare continuous daily dataset timeline
            daily_series = df.groupby('Date_Only').size().reset_index(name='Ticket_Count')
            daily_series = daily_series.sort_values('Date_Only').reset_index(drop=True)

            if len(daily_series) >= 5:
                # --- SECTION A: STATISTICAL ANOMALY DETECTION ---
                st.markdown("#### 🚨 Statistical Outage & Anomaly Detection")
                st.markdown("Identifies days where ticket volumes spiked unexpectedly beyond normal historical fluctuations.")
                
                mean_vol = daily_series['Ticket_Count'].mean()
                std_vol = daily_series['Ticket_Count'].std()
                upper_bound = mean_vol + (anomaly_threshold * std_vol)
                
                daily_series['Is_Anomaly'] = daily_series['Ticket_Count'] > upper_bound
                anomalies = daily_series[daily_series['Is_Anomaly']]
                
                # Visualizing Anomalies
                fig_anomaly = go.Figure()
                fig_anomaly.add_trace(go.Scatter(x=daily_series['Date_Only'], y=daily_series['Ticket_Count'], mode='lines+markers', name='Daily Ticket Count', line=dict(color='#A6ACAF')))
                fig_anomaly.add_trace(go.Scatter(x=anomalies['Date_Only'], y=anomalies['Ticket_Count'], mode='markers', name='Flagged Anomaly Outage', marker=dict(color='red', size=12, symbol='x')))
                fig_anomaly.add_shape(type="line", x0=daily_series['Date_Only'].min(), y0=upper_bound, x1=daily_series['Date_Only'].max(), y1=upper_bound, line=dict(color="red", width=2, dash="dash"), name="Anomaly Threshold")
                fig_anomaly.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Tickets Logged")
                st.plotly_chart(fig_anomaly, use_container_width=True)
                
                st.metric("Total Anomaly Days Detected", f"{len(anomalies)} Days", help="Days exceeding historical baseline standard deviations.")

                # --- SECTION B: MACHINE LEARNING TREND FORECASTING ---
                st.markdown("---")
                st.markdown("#### 🔮 Machine Learning 7-Day Demand Forecasting")
                st.markdown("Trains a Linear Regression model on historical trajectory sequences to project resource demand over the next week.")
                
                # Create ordinal feature variables for scikit-learn fitting
                daily_series['Day_Index'] = np.arange(len(daily_series))
                
                X = daily_series[['Day_Index']].values
                y = daily_series['Ticket_Count'].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Predict historical values to show trend fit line
                daily_series['Trend_Line'] = model.predict(X)
                
                # Generate future forecasting timeline variables
                future_days = 7
                last_index = daily_series['Day_Index'].max()
                future_indices = np.arange(last_index + 1, last_index + 1 + future_days).reshape(-1, 1)
                future_predictions = model.predict(future_indices).clip(min=0) # enforce no negative tickets
                
                last_date = daily_series['Date_Only'].max()
                future_dates = [last_date + pd.Timedelta(days=int(i)) for i in range(1, future_days + 1)]
                
                forecast_df = pd.DataFrame({
                    'Date_Only': future_dates,
                    'Forecast_Volume': future_predictions.round(1)
                })
                
                # Plot Forecast
                fig_forecast = go.Figure()
                # Historic Data
                fig_forecast.add_trace(go.Scatter(x=daily_series['Date_Only'], y=daily_series['Ticket_Count'], mode='lines+markers', name='Historical Data', line=dict(color='#007BFF')))
                # Fit Trend Line
                fig_forecast.add_trace(go.Scatter(x=daily_series['Date_Only'], y=daily_series['Trend_Line'], mode='lines', name='Historical Fit Trend', line=dict(color='#FFA500', dash='dash')))
                # Projected Line
                fig_forecast.add_trace(go.Scatter(x=forecast_df['Date_Only'], y=forecast_df['Forecast_Volume'], mode='lines+markers', name='7-Day Machine Learning Forecast', line=dict(color='#2CA02C', width=3)))
                
                fig_forecast.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Ticket Load Demand")
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Render forecast value breakdown array tabularly
                f_col1, f_col2 = st.columns([1, 2])
                with f_col1:
                    st.markdown("📋 **Expected Operational Load**")
                    st.dataframe(forecast_df, use_container_width=True, hide_index=True)
                with f_col2:
                    st.info(f"💡 **Model Evaluation Insights:** The statistical directional baseline shows a daily growth coefficient of **{model.coef_[0]:.3f} tickets per day**. Ensure staffing scales dynamically to support these projected parameters.")
            else:
                st.info("📊 Insufficient continuous timeline parameters on file. Minimum 5 distinct date groups required to feed predictive ML forecasting logic arrays.")

        # 5. Raw Comprehensive Data Table (Global)
        st.markdown("---")
        with st.expander("🔍 View Filtered Raw Ticketing Logs (All Data Columns)"):
            st.dataframe(df, use_container_width=True, hide_index=True)

    except PermissionError:
        st.error("⚠️ **Permission Error:** Please close the file `Call_Mgmt_data.xlsx` if you have it open in Microsoft Excel, then refresh this page.")
    except Exception as e:
        st.error(f"An unexpected data processing error occurred: {e}")

else:
    st.error(f"❌ **File Not Found:** Could not find `{EXCEL_FILE}` in your root directory. Ensure it sits right next to your `app.py` file.")