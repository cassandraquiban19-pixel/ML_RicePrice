import streamlit as st
import pandas as pd
import plotly.express as px
from rice_forecasting import load_prices, evaluate_models, forecast_future

st.set_page_config(page_title='RiceCast PH | Rice Price Forecasting', page_icon='🌾', layout='wide', initial_sidebar_state='expanded')

st.markdown('''
<style>
:root { --field:#174D35; --leaf:#63894B; --gold:#D6A62E; --cream:#F6F2E6; --ink:#173A2B; }
.stApp { background:linear-gradient(180deg,#FAF8F0 0%,#EFF1E3 100%); color:var(--ink); }
[data-testid="stHeader"] { background:rgba(250,248,240,.92); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#E6EEDC 0%,#F7F1E3 100%); border-right:1px solid #D6DEC9; }
[data-testid="stSidebar"] * { color:#173A2B; }
h1,h2,h3 { color:#174D35 !important; }
[data-testid="stMetric"] { background:rgba(255,255,255,.88); border:1px solid #E5E2D4; border-radius:15px; padding:15px; }
[data-testid="stMetricValue"] { color:#174D35; }
div[data-testid="stPlotlyChart"] { background:rgba(255,255,255,.7); border:1px solid #E5E2D4; border-radius:14px; padding:6px; }
div.stButton > button, div.stDownloadButton > button { background:#174D35; color:white; border:1px solid #174D35; border-radius:9px; font-weight:600; }
.hero { padding:25px 30px; border-radius:19px; color:#F9F6E9; background:linear-gradient(105deg,rgba(20,68,45,.98),rgba(76,117,55,.93)); margin-bottom:18px; }
.hero h1 { color:#FFF9E8 !important; margin:0; font-size:2.15rem; }
.hero p { color:#E6EEDB; margin:7px 0 0; }
.eyebrow { color:#E9C86D; text-transform:uppercase; font-size:.76rem; letter-spacing:.13em; font-weight:700; margin-bottom:7px; }
.note-card { padding:15px 18px; border-radius:13px; background:#E8EEDC; border:1px solid #D3DFC8; }
.gold-card { padding:17px; border-radius:15px; background:#FBF2D8; border:1px solid #EBD99A; }
</style>
''', unsafe_allow_html=True)

@st.cache_data
def load_default():
    return load_prices('rice_prices_cleaned.csv')


def forecast_selected_model(prices, region, rice_type, months, model_kind):
    forecast, _ = forecast_future(prices, region, rice_type, months=months, model_kind=model_kind)
    return forecast

# Load data
st.sidebar.markdown('# 🌾 RiceCast PH')
st.sidebar.caption('Data today. Food security tomorrow.')
st.sidebar.divider()
page = st.sidebar.radio('Navigate', ['Overview', 'Price Forecast', 'Model Comparison', 'About the Project'])
st.sidebar.divider()
st.sidebar.markdown('### 🌱 Forecast settings')
try:
    prices = load_default()
except Exception as exc:
    st.error(f'Could not read the dataset: {exc}')
    st.stop()
prices = prices.copy()
prices['Date'] = pd.to_datetime(prices['Date'])
regions = sorted(prices.Region.dropna().unique().tolist())
rice_types = sorted(prices.Rice_Type.dropna().unique().tolist())
region = st.sidebar.selectbox('Region', regions, index=0)
rice_type = st.sidebar.selectbox('Rice type', rice_types, index=0)
model_kind = st.sidebar.selectbox('Forecast model', ['Naive Forecast', 'Random Forest', 'Extra Trees', 'Ridge Regression', 'XGBoost'])
horizon = st.sidebar.slider('Forecast horizon (months)', 1, 12, 6)

selected = prices[(prices.Region == region) & (prices.Rice_Type == rice_type)].sort_values('Date')
if selected.empty:
    st.warning('No observations are available for this selection.')
    st.stop()
latest_date = prices.Date.max()
latest_price = float(selected.iloc[-1].Price_PHP_per_kg)
avg_price = float(selected.Price_PHP_per_kg.mean())
period_change = latest_price - float(selected.iloc[0].Price_PHP_per_kg)

st.markdown(f'''<div class="hero"><div class="eyebrow">🌾 Golden Palay • Agricultural Intelligence</div><h1>Philippine Rice Price Forecasting</h1><p>RiceCast PH · Monthly retail prices · {selected.Date.min():%B %Y}–{latest_date:%B %Y}</p></div>''', unsafe_allow_html=True)

if page == 'Overview':
    k1,k2,k3,k4 = st.columns(4)
    k1.metric('Latest recorded price', f'₱{latest_price:,.2f}/kg', selected.iloc[-1].Date.strftime('%b %Y'))
    k2.metric('Observations', f'{len(selected):,}', 'selected region and rice type')
    k3.metric('Historical average', f'₱{avg_price:,.2f}/kg')
    k4.metric('Price change across period', f'₱{period_change:+,.2f}/kg', 'first to latest observation')
    left,right = st.columns([1.25,.75])
    with left:
        st.subheader('🌱 Historical price trend')
        fig = px.line(selected, x='Date', y='Price_PHP_per_kg', markers=True, labels={'Date':'Month','Price_PHP_per_kg':'Price (PHP/kg)'}, title=f'{rice_type} · {region}')
        fig.update_traces(line=dict(color='#247343',width=3), marker=dict(color='#D6A62E',size=5))
        fig.update_layout(template='plotly_white',height=390,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(255,255,255,.35)',font=dict(color='#244334'),xaxis=dict(showgrid=False),yaxis=dict(gridcolor='#E5E8DC'))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.subheader('🌾 Dataset at a glance')
        st.markdown(f'''<div class="note-card"><b>Data coverage</b><br>{prices.Date.min():%B %Y} – {latest_date:%B %Y}<br><br><b>Reporting areas</b><br>{prices.Region.nunique()}<br><br><b>Rice types</b><br>{len(rice_types)}<br><br><b>Total price observations</b><br>{len(prices):,}<br><br><b>Selected forecast model</b><br>{model_kind}</div>''',unsafe_allow_html=True)
        st.info('Forecasts use historical prices and do not directly account for weather, inflation, imports, fuel costs, or policy changes.')

elif page == 'Price Forecast':
    st.subheader('🔮 Price Forecast')
    st.caption('Choose a baseline or regression model. Forecasts are estimates, not guaranteed market prices.')
    try:
        forecast = forecast_selected_model(prices,region,rice_type,horizon,model_kind)
        forecast['Date'] = pd.to_datetime(forecast.Date)
        hist = selected[['Date','Price_PHP_per_kg']].rename(columns={'Price_PHP_per_kg':'Price (PHP/kg)'}).copy(); hist['Series']='Historical price'
        future = forecast[['Date','Forecast_Price_PHP_per_kg']].rename(columns={'Forecast_Price_PHP_per_kg':'Price (PHP/kg)'}).copy(); future['Series']='Forecast'
        combined = pd.concat([hist,future],ignore_index=True)
        fig = px.line(combined,x='Date',y='Price (PHP/kg)',color='Series',markers=True,color_discrete_map={'Historical price':'#247343','Forecast':'#D6A62E'})
        fig.update_traces(line=dict(width=3)); fig.update_layout(template='plotly_white',height=440,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(255,255,255,.35)',font=dict(color='#244334'),xaxis=dict(showgrid=False),yaxis=dict(gridcolor='#E5E8DC'))
        st.plotly_chart(fig,use_container_width=True)
        last_forecast=float(forecast.Forecast_Price_PHP_per_kg.iloc[-1]); delta=last_forecast-latest_price
        a,b,c=st.columns(3); a.metric('Price at forecast horizon',f'₱{last_forecast:,.2f}/kg',forecast.Date.iloc[-1].strftime('%B %Y')); b.metric('Change vs latest price',f'₱{delta:+,.2f}/kg'); c.metric('Forecast model',model_kind)
        display=forecast.copy(); display['Date']=display.Date.dt.strftime('%B %Y'); display['Forecast_Price_PHP_per_kg']=display.Forecast_Price_PHP_per_kg.round(2)
        st.dataframe(display,use_container_width=True,hide_index=True)
        st.download_button('⬇ Download forecast CSV',forecast.to_csv(index=False).encode('utf-8'),file_name='rice_price_forecast.csv',mime='text/csv')
    except Exception as exc:
        st.error(f'Forecast could not be generated for {model_kind}: {exc}')

elif page == 'Model Comparison':
    st.subheader('📊 Model comparison')
    st.write('Compare a naive baseline and four regression models using a chronological holdout. The latest 12 target months are reserved for testing.')
    if st.button('🌾 Evaluate all models',type='primary'):
        with st.spinner('Training and evaluating models on a time-based split...'):
            try:
                scores,preds,details=evaluate_models(prices,test_months=12,include_xgboost=True)
                scores=scores.copy()
                if 'Model' in scores.columns:
                    scores['Model']=scores['Model'].replace({'Naive (previous month)':'Naive Forecast'})
                st.session_state['rice_model_scores']=scores
                st.session_state['rice_model_preds']=preds
            except Exception as exc:
                st.error(f'Model evaluation failed: {exc}')
    if 'rice_model_scores' in st.session_state:
        scores=st.session_state['rice_model_scores']
        st.dataframe(scores.round(3),use_container_width=True,hide_index=True)
        if not scores.empty and 'MAE' in scores.columns:
            best=scores.sort_values('MAE').iloc[0]
            st.markdown(f'''<div class="gold-card"><b>🏆 Best result by MAE</b><h3 style="margin:5px 0">{best['Model']}</h3>MAE: <b>₱{best['MAE']:.2f}/kg</b>. Lower MAE means smaller average absolute errors on this test period.</div>''',unsafe_allow_html=True)
        metrics=[c for c in ['MAE','RMSE'] if c in scores.columns]
        if metrics and 'Model' in scores.columns:
            metric=st.selectbox('Metric to compare',metrics)
            bar=px.bar(scores.sort_values(metric),x='Model',y=metric,color='Model',labels={metric:f'{metric} (PHP/kg)'})
            bar.update_layout(template='plotly_white',height=350,showlegend=False,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(255,255,255,.35)',font=dict(color='#244334'),yaxis=dict(gridcolor='#E5E8DC'))
            bar.update_traces(marker_color='#6F9656'); st.plotly_chart(bar,use_container_width=True)
        st.download_button('⬇ Download model metrics CSV',scores.to_csv(index=False).encode('utf-8'),file_name='rice_model_metrics.csv',mime='text/csv')
        preds=st.session_state.get('rice_model_preds')
        if isinstance(preds,pd.DataFrame):
            st.download_button('⬇ Download test predictions CSV',preds.to_csv(index=False).encode('utf-8'),file_name='rice_test_predictions.csv',mime='text/csv')
            with st.expander('View test predictions'): st.dataframe(preds,use_container_width=True,hide_index=True)

else:
    st.subheader('ℹ️ About the Project')
    st.write('RiceCast PH explores monthly retail rice prices across Philippine reporting areas and rice types. The cleaned long-format dataset powers historical charts, filtering, and forecasting.')
    st.markdown('### Models included')
    st.markdown('- **Naive Forecast:** repeats the latest observed price as a simple benchmark.\n- **Random Forest:** learns nonlinear patterns using many decision trees.\n- **Extra Trees:** uses randomized decision trees to learn price patterns.\n- **Ridge Regression:** a regularized linear regression model.\n- **XGBoost:** uses gradient-boosted decision trees for prediction.')
    st.markdown('### Dataset note')
    st.write('The separate supervised dataset can support feature-based training with lagged prices and rolling statistics. It is not loaded directly by this dashboard version. Features must be constructed without using future information.')
    st.markdown('### Limitations')
    st.write('Forecasts use historical prices only and do not directly account for weather, inflation, fuel costs, imports, or policy changes. Treat forecasts as estimates, not guaranteed prices.')

st.divider()
st.caption('RiceCast PH • Golden Palay agricultural theme • For educational and analytical use.')
