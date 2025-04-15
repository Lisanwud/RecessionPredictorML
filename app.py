import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertForSequenceClassification, BertTokenizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Streamlit page configuration
st.set_page_config(page_title="Economic Narratives Dashboard", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #e0eafc, #cfdef3);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.stApp {
    background: transparent;
}
h1 {
    color: #2c3e50;
    text-align: center;
    font-size: 2.5em;
    margin-bottom: 0.5em;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}
h2 {
    color: #34495e;
    font-size: 1.8em;
    border-bottom: 2px solid #3498db;
    padding-bottom: 0.3em;
    margin-top: 1em;
}
.stSidebar {
    background: #ffffff;
    border-right: 2px solid #3498db;
    padding: 1em;
    box-shadow: 2px 0 5px rgba(0,0,0,0.1);
}
.stSidebar h2 {
    color: #2c3e50;
    font-size: 1.5em;
}
.card {
    background: #ffffff;
    border-radius: 10px;
    padding: 1.5em;
    margin: 1em 0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    border-left: 5px solid #3498db;
}
.stButton>button {
    background: linear-gradient(to right, #3498db, #2980b9);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 0.7em 2em;
    font-size: 1em;
    font-weight: bold;
    transition: all 0.3s ease;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}
.stButton>button:hover {
    background: linear-gradient(to right, #2980b9, #1f618d);
    transform: translateY(-2px);
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
}
.stSpinner {
    color: #3498db;
}
.footer {
    text-align: center;
    color: #7f8c8d;
    font-size: 0.9em;
    margin-top: 2em;
    padding: 1em;
    background: #ecf0f1;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📈 Economic Narratives Dashboard")
st.markdown("""
<div class="card">
This dashboard transforms complex economic indicators into clear, actionable insights for the general public. 
Select a date range to explore economic trends, view sentiment analysis, and read AI-generated narratives about the economy.
</div>
""", unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('./data/economic_indicators.csv', parse_dates=['DATE'])
        data.set_index('DATE', inplace=True)
        return data
    except FileNotFoundError:
        st.error("Error: './data/economic_indicators.csv' not found. Please run the data collection script first.")
        return None

# Load and cache GPT-2 model and tokenizer
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

# Load and cache FinBERT model and tokenizer
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

# Load and cache Random Forest model
@st.cache_resource
def train_rf_model(data):
    key_columns = ['unemployment', 'yield_spread', 'SP500', 'CPI', 'recession_probabilities']
    daily_changes = data[key_columns].diff().dropna()
    daily_data = data[key_columns].loc[daily_changes.index]
    combined_data = pd.concat([daily_data, daily_changes.add_suffix('_change')], axis=1)
    
    # Apply sentiment classification
    def assign_sentiment(row):
        if row['SP500_change'] > 0 and row['recession_probabilities_change'] <= 0:
            return 'positive'
        elif row['SP500_change'] < 0 or row['recession_probabilities_change'] > 5:
            return 'negative'
        else:
            return 'neutral'
    
    combined_data['sentiment'] = combined_data.apply(assign_sentiment, axis=1)
    
    # Train RF model
    features = [col for col in combined_data if '_change' in col]
    X = combined_data[features]
    y = combined_data['sentiment']
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    return rf_model

# New generate_narrative_gpt2 function as provided
def generate_narrative_gpt2(date, data, rf_model, finbert_model, finbert_tokenizer, model, tokenizer, max_length=150):
    """
    Generate an economic narrative for a specific date using GPT-2 with combined sentiment voting.
    
    Args:
        date: Target date for narrative
        data: DataFrame containing economic indicators
        rf_model: Trained Random Forest classifier
        finbert_model: FinBERT model for sentiment analysis
        finbert_tokenizer: FinBERT tokenizer
        model: GPT-2 language model
        tokenizer: GPT-2 tokenizer
        max_length: Maximum length of generated text
        
    Returns:
        String containing the economic narrative
    """
    if date not in data.index:
        nearest_idx = data.index.get_indexer([date], method='nearest')[0]
        nearest_date = data.index[nearest_idx]
        return (f'No data available for {date.strftime("%Y-%m-%d")}. '
                f'Nearest available date is {nearest_date.strftime("%Y-%m-%d")}.')
    
    row = data.loc[date]
    date_str = date.strftime('%B %d, %Y')
    
    # Rule-based sentiment
    def assign_sentiment(row):
        if row['SP500_change'] > 0 and row['recession_probabilities_change'] <= 0:
            return 'positive'
        elif row['SP500_change'] < 0 or row['recession_probabilities_change'] > 5:
            return 'negative'
        else:
            return 'neutral'
    rule_sentiment = assign_sentiment(row)
    
    # Random Forest sentiment
    change_cols = [col for col in data.columns if '_change' in col]
    change_data = pd.DataFrame([row[change_cols].values], columns=change_cols)
    rf_sentiment = rf_model.predict(change_data)[0]
    
    # FinBERT sentiment
    summary = (f"On {date_str}, unemployment was {row['unemployment']:.1f}%, "
               f"S&P 500 at {row['SP500']:.2f} ({row['SP500_change']:+.2f} points), "
               f"recession probability at {row['recession_probabilities']:.1f}%, "
               f"yield spread at {row['yield_spread']:.2f}, CPI at {row['CPI']:.1f}.")
    inputs = finbert_tokenizer(summary, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(finbert_model.device) for k, v in inputs.items()}  # Move inputs to the correct device
    with torch.no_grad():
        outputs = finbert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_score = torch.argmax(probs, dim=-1).item()
    finbert_sentiment = ["positive", "neutral", "negative"][sentiment_score]
    
    # Combine sentiments with a voting system
    sentiments = [rule_sentiment, rf_sentiment, finbert_sentiment]
    positive_count = sentiments.count("positive")
    negative_count = sentiments.count("negative")
    neutral_count = sentiments.count("neutral")

    if positive_count > max(negative_count, neutral_count):
        final_sentiment = "positive"
    elif negative_count > max(positive_count, neutral_count):
        final_sentiment = "negative"
    else:
        final_sentiment = "neutral"  # Default to neutral for ties or mixed signals
    
    # Generate narrative based on final sentiment
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
        narrative = (f"On {date_str}, economic conditions appear mixed. The S&P 500 shifted by {sp50_change:.2f} "
                     f"points to {sp500:.2f}, while unemployment remained at {unemployment:.1f}%. A recession probability of "
                     f"{recession_prob:.1f}% and yield spread of {yield_spread:.2f} suggest a wait-and-see approach.")
    
    # Format change values for readability
    sp500_change_str = f"{row['SP500_change']:+.2f}"
    unemp_change_str = f"{row['unemployment_change']:+.2f}"
    recession_change_str = f"{row['recession_probabilities_change']:+.2f}"
    
    # Create prompt for GPT-2
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
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Move inputs to the correct device
    
    # Generate narrative with GPT-2
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
        
        # Extract only the generated part (not the prompt)
        gpt_narrative = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gpt_narrative = gpt_narrative[len(prompt):].strip()
        
        # Format the final narrative with clear indicator highlights
        formatted_narrative = f"""## Economic Narrative for {date_str}

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

# Fallback template-based narrative
def generate_fallback_narrative(date, data, sentiment):
    """
    Generate a template-based narrative if GPT-2 fails.
    """
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

# Load data
data = load_data()

# Load models
tokenizer, model, device = load_gpt2_model()
finbert_model, finbert_tokenizer, finbert_device = load_finbert_model()

if data is not None and tokenizer is not None and model is not None and finbert_model is not None:
    # Prepare data with change columns
    key_columns = ['unemployment', 'yield_spread', 'SP500', 'CPI', 'recession_probabilities']
    daily_changes = data[key_columns].diff()
    combined_data = pd.concat([data[key_columns], daily_changes.add_suffix('_change')], axis=1)
    
    # Train RF model
    rf_model = train_rf_model(data)
    
    # Sidebar for date range selection
    st.sidebar.header("Filter Options")
    min_date = data.index.min()
    max_date = data.index.max()
    
    # Use session state to store date inputs and control rendering
    if 'start_date' not in st.session_state:
        st.session_state.start_date = min_date
    if 'end_date' not in st.session_state:
        st.session_state.end_date = max_date
    if 'show_data' not in st.session_state:
        st.session_state.show_data = False

    start_date = st.sidebar.date_input("Start Date", st.session_state.start_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", st.session_state.end_date, min_value=min_date, max_value=max_date)

    # Convert date inputs to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Fancy Submit Button
    if st.sidebar.button("Explore Data"):
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
        st.session_state.show_data = True

    # Only render the data and visualizations if the "Explore Data" button is clicked
    if st.session_state.show_data:
        # Filter data based on date range
        filtered_data = combined_data.loc[st.session_state.start_date:st.session_state.end_date].copy()

        if filtered_data.empty:
            st.error(f"No data available for the selected date range ({st.session_state.start_date.strftime('%Y-%m-%d')} to {st.session_state.end_date.strftime('%Y-%m-%d')}). Please select a different range.")
        else:
            # Function to assign sentiment (for visualization)
            def assign_sentiment(row):
                sp500_change = row['SP500_change'] if 'SP500_change' in row else 0
                recession_prob_change = row['recession_probabilities_change'] if 'recession_probabilities_change' in row else 0
                if sp500_change > 0 and recession_prob_change <= 0:
                    return 'positive'
                elif sp500_change < 0 or recession_prob_change > 5:
                    return 'negative'
                else:
                    return 'neutral'

            # Apply sentiment classification for visualization
            filtered_data['sentiment'] = filtered_data.apply(assign_sentiment, axis=1)

            # Convert sentiment to numeric for plotting
            sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
            filtered_data['sentiment_numeric'] = filtered_data['sentiment'].map(sentiment_map)

            # Main content
            st.header("Economic Indicators Overview")

            # Plot key indicators
            st.subheader("Key Economic Indicators Over Time")
            indicators = ['unemployment', 'SP500', 'CPI', 'recession_probabilities', 'yield_spread']
            fig = go.Figure()
            for indicator in indicators:
                if filtered_data[indicator].notna().any():
                    fig.add_trace(go.Scatter(
                        x=filtered_data.index,
                        y=filtered_data[indicator],
                        mode='lines',
                        name=indicator
                    ))
            fig.update_layout(
                title="Economic Indicators",
                xaxis_title="Date",
                yaxis_title="Value",
                template="plotly_white",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Sentiment Timeline
            st.subheader("Economic Sentiment Timeline")
            fig_sentiment = px.scatter(
                filtered_data,
                x=filtered_data.index,
                y='sentiment_numeric',
                color='sentiment',
                color_discrete_map={'positive': '#009E73', 'neutral': '#0072B2', 'negative': '#CC79A7'},
                labels={'sentiment_numeric': 'Sentiment', 'DATE': 'Date'},
                title="Economic Sentiment Over Time"
            )
            fig_sentiment.update_yaxes(tickvals=[-1, 0, 1], ticktext=['Negative', 'Neutral', 'Positive'])
            fig_sentiment.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.plotly_chart(fig_sentiment, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Sentiment Distribution
            st.subheader("Sentiment Distribution")
            sentiment_counts = filtered_data['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            fig_dist = px.bar(
                sentiment_counts,
                x='Sentiment',
                y='Count',
                title="Sentiment Distribution",
                color='Sentiment',
                color_discrete_map={'positive': '#009E73', 'neutral': '#0072B2', 'negative': '#CC79A7'}
            )
            fig_dist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.plotly_chart(fig_dist, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Narrative Generation
            st.header("Economic Narrative")
            selected_date = st.selectbox("Select a Date for Narrative", filtered_data.index.strftime('%Y-%m-%d'))
            selected_date = pd.to_datetime(selected_date)

            if selected_date in filtered_data.index:
                # Generate narrative
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
                
                # Use fallback if GPT-2 fails
                if narrative.startswith("Error") or "No data available" in narrative:
                    row = filtered_data.loc[selected_date]
                    sentiment = assign_sentiment(row)
                    narrative = generate_fallback_narrative(selected_date, row, sentiment)
                    st.warning("Using fallback narrative due to GPT-2 unavailability or missing data.")

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(narrative)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No data available for the selected date.")

    # Footer
    st.markdown("""
    <div class="footer">
    Built with Streamlit | Data sourced from FRED and Yahoo Finance | Powered by GPT-2 and FinBERT
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("Data or model loading failed. Please check the dataset and try again.")