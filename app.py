import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertForSequenceClassification, BertTokenizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Streamlit page configuration
st.set_page_config(
    page_title="Economic Narratives Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for modern, clean design and vibrant backgrounds
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Poppins:wght@400;500;600;700&display=swap');
    body {
        font-family: 'Roboto', sans-serif;
        background: linear-gradient(120deg, #e0eafc 0%, #cfdef3 100%);
        color: #2c3e50;
    }
    .stApp {
        background: #f5f7fa;
    }
    .main-header, .footer {
        background: linear-gradient(90deg, #3a7bd5, #00d2ff);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 16px rgba(58, 123, 213, 0.15);
        text-align: center;
    }
    .main-header h2, .footer h2 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.95;
        max-width: 800px;
        margin: 0 auto;
    }
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
    }
    .logo-container span {
        margin-right: 0.75rem;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #3a7bd5 0%, #00d2ff 100%);
        color: #fff;
    }
    .sidebar-header {
        font-family: 'Poppins', sans-serif;
        color: #fff;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        background: #000000;
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        width: 100%;
        transition: all 0.3s ease;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.4);
        transform: translateY(-2px);
    }
    .date-input label {
        font-weight: 500;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .footer {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(90deg, #3a7bd5, #00d2ff);
        border-radius: 12px;
        margin-top: 2rem;
        color: #fff;
        font-size: 0.8rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
    }
    .stSelectbox label {
        font-weight: 500 !important;
    }
    .main .block-container {
        background-image: none;
        background-color: #121212;
        padding-right: 2rem;
        color: #ffffff;
    }
    .stApp {
        background: #121212;
        color: #ffffff;
    }
    h1, h2, h3, p, label {
        color: #ffffff !important;
    }
    .plotly .bg {
        fill: #1e1e1e !important;
    }
    .js-plotly-plot .plotly .gridlayer path {
        stroke: rgba(255, 255, 255, 0.1) !important;
    }
    /* Style for the date range text */
    .date-range-text {
        font-size: 0.7rem;
        color: #ffffff;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
    }
    /* Hover effect for About this Dashboard expander */
    div[data-testid="stExpander"] > div > div > div {
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    div[data-testid="stExpander"] > div > div > div:hover {
        background-color: rgba(58, 123, 213, 0.1);
        box-shadow: 0 2px 8px rgba(58, 123, 213, 0.2);
        transform: translateY(-2px);
    }
    /* Style for the table to match plot backgrounds */
    div[data-testid="stDataFrame"] {
        background-color: #2C3E50 !important;
        border-radius: 8px;
        padding: 10px;
    }
    div[data-testid="stDataFrame"] table {
        background-color: #2C3E50 !important;
        color: #ffffff !important;
    }
    div[data-testid="stDataFrame"] th, div[data-testid="stDataFrame"] td {
        background-color: #2C3E50 !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    /* Decrease frequency text size on bar chart */
    .plotly .text {
        font-size: 10px !important;
    }
    /* Ensure chart titles are white */
    .plotly .gtitle {
        fill: #FFFFFF !important;
    }
    @media (max-width: 768px) {
        .main-header h2, .footer h2 {
            font-size: 1.2rem;
        }
        .main .block-container {
            background-size: 0;
        }
    }
</style>
""", unsafe_allow_html=True)

def create_header():
    st.markdown("""
    <div class="main-header">
        <div class="logo-container">
            <span style="font-size: 2.5rem;">📈</span>
            <h2>Economic Narratives Dashboard</h2>
        </div>
        <p>Transforming complex economic indicators into narratives</p>
    </div>
    """, unsafe_allow_html=True)

def create_footer():
    st.markdown("""
    <div class="footer">
        <p>© 2025 Economic Narratives Dashboard. Built with Streamlit | Data from FRED, Yahoo Finance | Powered by GPT-2</p>
    </div>
    """, unsafe_allow_html=True)

def about_section():
    with st.expander("About this Dashboard", expanded=False):
        st.markdown("""
        This dashboard transforms complex economic data into clear, concise narratives using advanced AI models (GPT-2).

        - **Economic Indicators:**
            - **Unemployment:** % of labor force without work but seeking employment.
            - **S&P 500:** Index tracking 500 large U.S. companies.
            - **CPI:** Measures price changes in consumer goods/services.
            - **Yield Spread:** Difference between long- and short-term Treasury yields.
            - **Recession Probability:** Likelihood of economic recession.

        - **Data Sources:** 
            - FRED (Federal Reserve Economic Data) 
            - Yahoo Finance
        - **Features:**
            - Visualize trends and highlight recession periods
            - AI-generated daily economic summaries
            - Sentiment analysis using ensemble methods
        """)

@st.cache_data
def load_data():
    try:
        data = pd.read_csv('./data/economic_indicators.csv', parse_dates=['DATE'])
        data.set_index('DATE', inplace=True)
        return data
    except FileNotFoundError:
        st.error("Error: './data/economic_indicators.csv' not found. Please run the data collection script first.")
        return None

@st.cache_resource
def load_gpt2_model():
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading GPT-2 model: {e}")
        return None, None, None

@st.cache_resource
def load_finbert_model():
    try:
        finbert_tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        finbert_model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
        finbert_model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        finbert_model = finbert_model.to(device)
        return finbert_model, finbert_tokenizer, device
    except Exception as e:
        st.error(f"Error loading FinBERT model: {e}")
        return None, None, None

def assign_sentiment(row):
    if row['SP500_change'] > 0 and row['recession_probabilities_change'] <= 0:
        return 'positive'
    elif row['SP500_change'] < 0 or row['recession_probabilities_change'] > 5:
        return 'negative'
    else:
        return 'neutral'

@st.cache_resource
def train_rf_model(data):
    key_columns = ['unemployment', 'yield_spread', 'SP500', 'CPI', 'recession_probabilities']
    daily_changes = data[key_columns].diff().dropna()
    daily_data = data[key_columns].loc[daily_changes.index]
    combined_data = pd.concat([daily_data, daily_changes.add_suffix('_change')], axis=1)
    combined_data['sentiment'] = combined_data.apply(assign_sentiment, axis=1)
    features = [col for col in combined_data if '_change' in col]
    X = combined_data[features]
    y = combined_data['sentiment']
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    return rf_model

def create_sentiment_chart(data):
    sentiment_counts = data['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    colors = {
        'positive': '#1b9e77',
        'neutral': '#7570b3',
        'negative': '#d95f02'
    }
    fig = px.bar(
        sentiment_counts, 
        x='Sentiment', 
        y='Count',
        color='Sentiment',
        color_discrete_map=colors,
        title='Sentiment Distribution',
        labels={'Count': 'Frequency', 'Sentiment': 'Economic Sentiment'},
        text='Count'
    )
    fig.update_layout(
        xaxis_title="Sentiment",
        yaxis_title="Count",
        legend_title="Sentiment",
        font=dict(family="Roboto, sans-serif", size=14, color='#FFFFFF'),
        title_font=dict(color='#FFFFFF'),
        plot_bgcolor='#2C3E50',
        paper_bgcolor='#34495E',
        showlegend=True,
        legend=dict(
            bgcolor='rgba(255,255,255,0.1)',
            bordercolor='rgba(255,255,255,0.3)',
            borderwidth=1
        ),
        margin=dict(t=45),  
        xaxis=dict(
            title_font=dict(color='#FFFFFF'),
            tickfont=dict(color='#FFFFFF')
        ),
        yaxis=dict(
            title_font=dict(color='#FFFFFF'),
            tickfont=dict(color='#FFFFFF')
        )
    )
    fig.update_traces(
        textposition='outside',
        textfont=dict(size=12)  # Reduced font size for the frequency labels
    )
    return fig

def generate_narrative_gpt2(date, data, rf_model, finbert_model, finbert_tokenizer, model, tokenizer, max_length=150):
    if date not in data.index:
        nearest_idx = data.index.get_indexer([date], method='nearest')[0]
        nearest_date = data.index[nearest_idx]
        return (f'No data available for {date.strftime("%Y-%m-%d")}. '
                f'Nearest available date is {nearest_date.strftime("%Y-%m-%d")}.')
    
    row = data.loc[date]
    date_str = date.strftime('%B %d, %Y')
    rule_sentiment = assign_sentiment(row)
    change_cols = [col for col in data.columns if '_change' in col]
    change_data = pd.DataFrame([row[change_cols].values], columns=change_cols)
    rf_sentiment = rf_model.predict(change_data)[0]
    summary = (f"On {date_str}, unemployment was {row['unemployment']:.1f}%, "
               f"S&P 500 at {row['SP500']:.2f} ({row['SP500_change']:+.2f} points), "
               f"recession probability at {row['recession_probabilities']:.1f}%, "
               f"yield spread at {row['yield_spread']:.2f}, CPI at {row['CPI']:.1f}.")
    inputs = finbert_tokenizer(summary, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(finbert_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = finbert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_score = torch.argmax(probs, dim=-1).item()
    finbert_sentiment = ["positive", "neutral", "negative"][sentiment_score]
    sentiments = [rule_sentiment, rf_sentiment, finbert_sentiment]
    positive_count = sentiments.count("positive")
    negative_count = sentiments.count("negative")
    neutral_count = sentiments.count("neutral")
    if positive_count > max(negative_count, neutral_count):
        final_sentiment = "positive"
    elif negative_count > max(positive_count, neutral_count):
        final_sentiment = "negative"
    else:
        final_sentiment = "neutral"
    
    sp500_change = row['SP500_change']
    sp500 = row['SP500']
    unemployment = row['unemployment']
    recession_prob = row['recession_probabilities']
    yield_spread = row['yield_spread']
    if final_sentiment == "positive":
        narrative = (f"Economic data for {date_str} points to a positive outlook. The S&P 500 rose by "
                     f"{sp500_change:.2f} points to {sp500:.2f}, reflecting market strength. Unemployment held steady "
                     f"at {unemployment:.1f}%, indicating a robust labor market. With a recession probability of "
                     f"{recession_prob:.1f}% and a yield spread of {yield_spread:.2f}, conditions suggest continued economic stability.")
    elif final_sentiment == "negative":
        narrative = (f"Economic indicators for {date_str} signal caution. The S&P 500 moved by {sp500_change:.2f} "
                     f"points to {sp500:.2f}, with unemployment at {unemployment:.1f}%. A recession probability of {recession_prob:.1f}% "
                     f"and yield spread of {yield_spread:.2f} raise concerns about economic uncertainty.")
    else:
        narrative = (f"On {date_str}, economic conditions appear mixed. The S&P 500 shifted by {sp500_change:.2f} "
                     f"points to {sp500:.2f}, while unemployment remained at {unemployment:.1f}%. A recession probability of "
                     f"{recession_prob:.1f}% and yield spread of {yield_spread:.2f} suggest a wait-and-see approach.")
    
    sp500_change_str = f"{row['SP500_change']:+.2f}"
    unemp_change_str = f"{row['unemployment_change']:+.2f}"
    recession_change_str = f"{row['recession_probabilities_change']:+.2f}"
    prompt = (
        f'On {date_str}, economic conditions showed a {final_sentiment} sentiment based on key indicators: '
        f'unemployment stood at {row["unemployment"]:.1f}% (change: {unemp_change_str}%), '
        f'the yield spread was {row["yield_spread"]:.2f} (change: {row["yield_spread_change"]:.2f}), '
        f'the S&P 500 was at {row["SP500"]:.2f} (change: {sp500_change_str} points), '
        f'CPI stood at {row["CPI"]:.1f} (change: {row["CPI_change"]:.2f}), and '
        f'recession probability was {row["recession_probabilities"]:.1f}% '
        f'(change: {recession_change_str}%). '
        'Generate a concise, coherent narrative summarizing these economic conditions.'
    )
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length + len(inputs['input_ids'][0]),
                num_return_sequences=1,
                do_sample=True,
                top_k=30,
                top_p=0.85,
                temperature=0.5,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        gpt_narrative = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gpt_narrative = gpt_narrative[len(prompt):].strip()
        formatted_narrative = f"""### Economic Narrative for {date_str}

**Overall Sentiment:** {final_sentiment.upper()}

**Key Indicators:**
- Unemployment: {row['unemployment']:.1f}% ({unemp_change_str}%)
- S&P 500: {row['SP500']:.2f} ({sp500_change_str} points)
- Recession Probability: {row['recession_probabilities']:.1f}% ({recession_change_str}%)
- Yield Spread: {row['yield_spread']:.2f} ({row['yield_spread_change']:.2f})
- CPI: {row['CPI']:.1f} ({row['CPI_change']:.2f})

**Narrative:**
{narrative}

**GPT-2 Enhanced Narrative:**
{gpt_narrative}"""
        return formatted_narrative
    except Exception as e:
        return f'Error generating narrative: {e}'

def generate_fallback_narrative(date, data, sentiment):
    unemployment = data.get('unemployment', 0)
    sp500 = data.get('SP500', 0)
    cpi = data.get('CPI', 0)
    recession_prob = data.get('recession_probabilities', 0)
    yield_spread = data.get('yield_spread', 0)
    date_str = date.strftime('%B %d, %Y')
    narrative = f"## Economic Narrative for {date_str}\n\n**Overall Sentiment:** {sentiment.upper()}\n\n"
    narrative += "**Key Indicators:**\n"
    narrative += f"- Unemployment: {unemployment:.1f}%\n"
    narrative += f"- S&P 500: {sp500:.2f}\n"
    narrative += f"- Recession Probability: {recession_prob:.1f}%\n"
    narrative += f"- Yield Spread: {yield_spread:.2f}\n"
    narrative += f"- CPI: {cpi:.1f}\n\n"
    narrative += "**Narrative:**\n"
    if sentiment == 'positive':
        narrative += (
            f"The economy is looking strong. With unemployment at {unemployment:.1f}%, jobs are plentiful. "
            f"The S&P 500 at {sp500:.2f} shows investor confidence. Inflation (CPI at {cpi:.1f}) is stable, "
            f"and recession risks are low at {recession_prob:.1f}%. This could be a good time to invest or make major purchases."
        )
    elif sentiment == 'negative':
        narrative += (
            f"The economy faces challenges. Unemployment at {unemployment:.1f}% suggests job market struggles. "
            f"The S&P 500 at {sp500:.2f} reflects market unease. Inflation is at {cpi:.1f}, and recession risks "
            f"are at {recession_prob:.1f}%. Consider saving and avoiding risky financial moves for now."
        )
    else:
        narrative += (
            f"The economy is stable but mixed. Unemployment at {unemployment:.1f}% is steady, and the S&P 500 "
            f"at {sp500:.2f} shows cautious markets. Inflation is at {cpi:.1f}, with recession risks at "
            f"{recession_prob:.1f}%. Maintain a balanced approach to financial decisions."
        )
    return narrative

def main():
    data = load_data()
    tokenizer, model, device = load_gpt2_model()
    finbert_model, finbert_tokenizer, finbert_device = load_finbert_model()
    if data is not None and tokenizer is not None and model is not None and finbert_model is not None:
        key_columns = ['unemployment', 'yield_spread', 'SP500', 'CPI', 'recession_probabilities']
        daily_changes = data[key_columns].diff()
        combined_data = pd.concat([data[key_columns], daily_changes.add_suffix('_change')], axis=1)
        combined_data['sentiment'] = combined_data.apply(assign_sentiment, axis=1)
        rf_model = train_rf_model(data)
        if 'startup_complete' not in st.session_state:
            st.session_state.startup_complete = False
            st.session_state.load_progress = 0
            st.session_state.show_data = False
        if 'selected_date' not in st.session_state:
            st.session_state.selected_date = data.index.min()
        create_header()
         
        if not st.session_state.startup_complete:
            progress_bar = st.progress(st.session_state.load_progress)
            load_status = st.empty()
            load_status.text("Loading data...")
        if data is not None:
            if not st.session_state.startup_complete:
                st.session_state.load_progress = 33
                progress_bar.progress(st.session_state.load_progress)
                load_status.text("Setting up analysis components...")
            if not st.session_state.startup_complete:
                st.session_state.load_progress = 67
                progress_bar.progress(st.session_state.load_progress)
                load_status.text("Preparing dashboard...")
            if not st.session_state.startup_complete:
                st.session_state.load_progress = 100
                progress_bar.progress(st.session_state.load_progress)
                st.session_state.startup_complete = True
                load_status.empty()
                progress_bar.empty()
            about_section()
            st.sidebar.markdown('<div class="sidebar-header">Select Date Range</div>', unsafe_allow_html=True)
            min_date = data.index.min()
            max_date = data.index.max()
            if 'start_date' not in st.session_state:
                st.session_state.start_date = min_date
            if 'end_date' not in st.session_state:
                st.session_state.end_date = max_date
            start_date = st.sidebar.date_input("From", value=st.session_state.start_date,
                                               min_value=min_date, max_value=max_date)
            end_date = st.sidebar.date_input("To", value=st.session_state.end_date,
                                             min_value=min_date, max_value=max_date)
            st.sidebar.markdown('<div class="date-range-text">Available date range: January 1, 1990 – April 14, 2025</div>', unsafe_allow_html=True)
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            auto_update = st.sidebar.checkbox("Auto-update on date change", value=True)
            if not auto_update:
                update_btn = st.sidebar.button("Update Dashboard")
            else:
                update_btn = True
            show_data_btn = st.sidebar.checkbox("Show Data Table", value=st.session_state.show_data)
            if (end_date - start_date).days > 1825:
                st.warning("Selected range exceeds 5 years. Consider narrowing for faster loading.")
            if update_btn:
                st.session_state.start_date = start_date
                st.session_state.end_date = end_date
                st.session_state.show_data = show_data_btn
                filtered_data = combined_data.loc[st.session_state.start_date:st.session_state.end_date].copy()
                   # ✅ INSERTED KPI CODE HERE
                 # Dynamically update KPI metrics based on selected date range
                latest_row = filtered_data.dropna().iloc[-1]
                latest_sentiment = latest_row['sentiment']
                latest_date = latest_row.name.strftime('%Y-%m-%d')
                recession_prob = round(latest_row['recession_probabilities'], 1)

                col1, col2, col3 = st.columns(3)
                col1.metric("📊 Sentiment", latest_sentiment.title())
                col2.metric("🕒 Date", latest_date)
                col3.metric("📉 Recession Probability", f"{recession_prob}%")
                if filtered_data.empty:
                    st.error(f"No data available for the selected date range ({st.session_state.start_date.strftime('%Y-%m-%d')} to {st.session_state.end_date.strftime('%Y-%m-%d')}). Please select a different range.")
                else:
                    # Show data table at the top for the FROM to TO range
                    if st.session_state.show_data:
                        st.subheader(f"Economic Data from {st.session_state.start_date.strftime('%Y-%m-%d')} to {st.session_state.end_date.strftime('%Y-%m-%d')}")
                        data_to_show = combined_data.loc[st.session_state.start_date:st.session_state.end_date].copy()
                        numeric_cols = data_to_show.select_dtypes(include=[np.number]).columns
                        formatter = {col: "{:.2f}" for col in numeric_cols}
                        st.dataframe(data_to_show.style.format(formatter))
                    chart_tab, narrative_tab = st.tabs(["Economic Charts", "Economic Narrative"])
                    with chart_tab:
                        st.subheader("Key Economic Indicators")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=filtered_data.index, y=filtered_data['SP500'],
                            mode='lines', name='S&P 500', line=dict(color='#3a7bd5', width=2)
                        ))
                        
                        fig.add_vrect(
                            x0="2001-03-01", x1="2001-11-30",
                            fillcolor="gray", opacity=0.15, line_width=0
                        )
                        if 'recession_probabilities' in filtered_data.columns:
                            rec_thresh = 50
                            in_recession = filtered_data['recession_probabilities'] > rec_thresh
                            start = None
                            for i, (date, rec) in enumerate(in_recession.items()):
                                if rec and start is None:
                                    start = date
                                elif not rec and start is not None:
                                    fig.add_vrect(
                                        x0=start, x1=date,
                                        fillcolor="gray", opacity=0.15, line_width=0
                                    )
                                    start = None
                            if start is not None:
                                fig.add_vrect(
                                    x0=start, x1=filtered_data.index[-1],
                                    fillcolor="gray", opacity=0.15, line_width=0
                                )
                        fig.add_trace(go.Scatter(
                            x=filtered_data.index, y=filtered_data['unemployment'],
                            mode='lines', name='Unemployment', line=dict(color='#e41a1c', width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=filtered_data.index, y=filtered_data['CPI'],
                            mode='lines', name='CPI', line=dict(color='#377eb8', width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=filtered_data.index, y=filtered_data['yield_spread'],
                            mode='lines', name='Yield Spread', line=dict(color='#4daf4a', width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=filtered_data.index, y=filtered_data['recession_probabilities'],
                            mode='lines', name='Recession Probabilities', line=dict(color='#984ea3', width=2)
                        ))
                        fig.update_layout(
                            title="Economic Indicators Over Time",
                            yaxis_title="Value",
                            xaxis_title="Date",
                            hovermode="x unified",
                            plot_bgcolor='#2C3E50',
                            paper_bgcolor='#34495E',
                            font=dict(color='#FFFFFF'),
                            margin=dict(t=45),
                            title_font=dict(color='#FFFFFF'),
                            showlegend=True,
                            legend=dict(
                                bgcolor='rgba(255,255,255,0.1)',
                                bordercolor='rgba(255,255,255,0.3)',
                                borderwidth=1
                            ),
                            xaxis=dict(
                                title_font=dict(color='#FFFFFF'),
                                tickfont=dict(color='#FFFFFF')
                            ),
                            yaxis=dict(
                                title_font=dict(color='#FFFFFF'),
                                tickfont=dict(color='#FFFFFF')
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.subheader("Market Sentiment Analysis")
                        sentiment_chart = create_sentiment_chart(filtered_data)
                        st.plotly_chart(sentiment_chart, use_container_width=True)
                    with narrative_tab:
                        st.subheader("Select Date for Economic Narrative")
                        default_narrative_date = st.session_state.start_date
                        available_dates = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
                        selected_date = st.date_input(
                            "Choose a date to view narrative", 
                            value=pd.to_datetime(default_narrative_date).date(),
                            min_value=pd.to_datetime(data.index.min()).date(),
                            max_value=pd.to_datetime(data.index.max()).date(),
                            key="narrative_date"
                        )
                        if selected_date:
                            selected_date = pd.to_datetime(selected_date)
                            st.session_state.start_date = selected_date
                            st.session_state.selected_date = selected_date
                            with st.spinner("Generating narrative..."):
                                narrative = generate_narrative_gpt2(
                                    selected_date,
                                    filtered_data,
                                    rf_model,
                                    finbert_model,
                                    finbert_tokenizer,
                                    model,
                                    tokenizer
                                )
                            if narrative.startswith("Error") or "No data available" in narrative:
                                if selected_date in filtered_data.index:
                                    row = filtered_data.loc[selected_date]
                                    sentiment = assign_sentiment(row)
                                    narrative = generate_fallback_narrative(selected_date, row, sentiment)
                                else:
                                    nearest_idx = filtered_data.index.get_indexer([selected_date], method='nearest')[0]
                                    nearest_date = filtered_data.index[nearest_idx]
                                    row = filtered_data.loc[nearest_date]
                                    sentiment = assign_sentiment(row)
                                    narrative = generate_fallback_narrative(nearest_date, row, sentiment)
                                    narrative = f'No data available for {selected_date.strftime("%Y-%m-%d")}. Using nearest date: {nearest_date.strftime("%Y-%m-%d")}.\n\n' + narrative
                            st.markdown(narrative)
    else:
        st.error("Failed to load required components. Please check the console for errors.")
    create_footer()

if __name__ == "__main__":
    main()