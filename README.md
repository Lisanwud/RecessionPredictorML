# Project 5: Enabling Wealth Managers to Safeguard Assets with an Economic Narratives Generator
*Abirami, Ahmad, Clarissa, Dat, Ted*

---

## Problem Statement  
When you're a wealth manager, timing is critical whether you're deploying a winning investment strategy or safeguarding assets against recessions.  
  
This project transforms complex economic indicators into approachable natural language narratives for adult wealth managers of all skill levels, using machine learning to classify economic sentiment and GPT-2 to generate readable insights that help individuals make timely decisions that can grow or save their clients' wealth. 

## The Question
How might we explain economic indicators in a way that even wealth managers without an economics background can understand and make decisions accordingly?

## The Audience

Wealth Managers of all skill levels (from professionals to amateurs managing personal finances) over age 18 who are managing US-based assets.

---

## Repository Structure

**code:**  
This folder contains several notebooks dedicated to creating a model to address the problem statement.

- Notebook [01](./code/1_DataCollection.ipynb) is where we collected data.
- Notebook [02](./code/2_DataCleaning_And_EDA.ipynb) is where we cleaned the collected data and saved the cleaned data as [economic_indicators.csv](./data/economic_indicators.csv).
- Notebook [03](./code/3_ModelTraining.ipynb) is where we created a random forest model with sentiment analysis to generate economic narratives.

**data:**  
This folder contains the datasets used to develop our model.

-  [Economic Indicators](./data/economic_indicators.csv)

**images:**  
This folder contains data visualizations created during our data exploration, which can explain our findings to our audience.

- [Correlation Heatmap](./images/correlation_heatmap.png)
- [Correlation Matrix of Economic Indicators](./images/correlation_matrix_economic_indicators.png)
- [Distribution of Key Indicators](./images/distribution_of_key_indicators.png)
- [Economic Indicators with Recession Periods](./images/economic_indicators_with_recession_periods.png) 
- [Indicators Over Time](./images/indicators_over_time.png)
- [Recession vs Non-Recession KDE](./images/recession_vs_non_recession_kde.png)
- [Sentiment Distribution](./images/sentiment_distribution.png)
- [Sentiment Timeline](./images/sentiment_timeline.png)
- [External: Federal Reserve of Economic Data (FRED) Graph of recession indicators from January 1, 1948 to March 1, 2025](./images/fredgraph.png)

**other:**  

- [Slides: Economic Narratives Generator](./slide/Economic\Narratives\Generator.pdf)
- [Our Streamlit Minimum Viable Product (MVP)](economic-narratives-dashboard.streamlit.app) and the [Prototype Online](https://economic-narratives-dashboard.streamlit.app/)
- README.md - The file you are reading right now. It contains the summaries, techniques used, data visualizations, conclusions, recommended next steps, and appendix of our project.

---

## Data Dictionary  
This data dictionary refers to [economic_indicators.csv](./data/economic_indicators.csv), which can be found in the [data](./data/) folder of this repository.  

| Variable Name  |   Role  |   Full Variable Name   | Description |
|----------------|---------|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DATE              | Feature |         Date      | From January 1, 1990 to April 8, 2025.    |
| unemployment      | Feature | Unemployment Rate | The percentage of the labor force that is unemployed and actively seeking employment. |
| yield_spread      | Feature |    Yield Spread   | The difference in yields between two different debt instruments, often of varying credit qualities or maturities. The yield spread can provide insights into investor sentiment, economic expectations, and potential risks in the financial markets.|
| industrial_prod   | Feature | Industrial Productivity | The output of the industrial sector of the economy, which includes manufacturing, mining, and utilities.   |
| consumer_conf  | Feature | Consumer Confidence | The degree of optimism that consumers feel about the overall state of the economy and their personal financial situation. When consumers are confident, they are more likely to spend money, which can stimulate economic growth.     |
| LEI | Feature | Leading Economic Index | A composite index that is designed to predict future economic activity. A rising LEI suggests an economical upswing; a declining LEI may indicate a potential downturn. |
| CPI     | Feature | Consumer Price Index | A key economic indicator that measures the average change over time in the prices paid by consumers for a basket of goods and services. |
| GDP_Growth   | Feature | Gross Domestic Product (GDP) Growth | The increase in a country's Gross Domestic Product (GDP) over a specific period. It reflects how much more value is being produced in the economy compared to a previous period.|
| recession_probabilities  | Feature | Recession Probabilities | The likelihood or chance that an economy will enter a recession within a specified time frame. |
| fed_funds_rate | Feature | Federal Funds Rate | The interest rate at which depository institutions (such as banks and credit unions) lend reserve balances to other depository institutions overnight on an uncollateralized basis. |
| currency_strength   | Feature |    Currency Strength (USD)   | Here, it is the value of the US Dollar relative to other currencies.|
| housing_starts     | Feature |     Housing Starts            |        The number of new residential construction projects that have begun during a specific period. |
| personal_consumption_expenses | Feature | Personal Consumption Expenditures (PCE)| Personal Consumption Expenditures (PCE) refer to the measure of the value of goods and services consumed by households.  |
| PPI | Feature | Producer Price Index    | The average change over time in the selling prices received by domestic producers for their output. |
| SP500         | Feature  | S&P 500      | S&P 500, or Standard & Poor's 500, is a stock market index that measures the performance of 500 of the largest publicly traded companies in the United States.                                                            

---

## Model Performance  
We used a Random Forest model and measured the performance of its sentiment classifier on accuracy.  
|Metric|Metric Value|Interpretation|
|---|---|---|
|Accuracy|1.00|Our model accurately predicted 100% of sentiments to generate its narratives.|

### Models We Considered to Answer the Problem  
Ultimately, we chose Random Forest because we are tackling a classification problem, and it's pretty fast.  

The sentiment of the economy is determined through a combination of three methods:
- FinBERT Sentiment Analysis
- Rule-Based Sentiment
- Random Forest (RF) Model Sentiment
Final Sentiment Decision:
The three sentiments (FinBERT, Rule-Based, RF) are combined using a voting system:
The counts of "positive," "neutral," and "negative" votes are compared.
The final sentiment is set to the majority vote:
- If "positive" has the highest count, the overall sentiment is "positive."
- If "negative" has the highest count, it is "negative."
- In case of ties or mixed signals, the default sentiment is "neutral."  
  
We considered using Recurrent Neural Networks as it is time based, but it usually needs lots of data, reshaping, the input of one layer should match input of the next - in other words, RNN requires a lot more work to generate narratives.

---

## Conclusions and Recommendations  
We conducted sentiment analysis to identify sentiments of the economy. Then we used the sentiment analysis as one of the inputs for GPT-2 to generate the narration, to explain the results in a way that wealth managers without an economics background can understand.  

We conclude that our model answered the problem. Our model achieved an accuracy of 1.00 for sentiment classification, despite the phrasing in our generated narrations being imperfect, likely because we used an older, free version of GPT (GPT-2).

### Recommendations  
None of us who worked on this project are licensed financial advisors, but we would recommend looking to what licensed experts have recommended in the past in similar economic upswings and downturns.  

Generally speaking, if our model's narrative sentiment is negative, take measures to safeguard wealth. If the narrative sentiment is positive, consider shifting into wealth-growing strategies.

---

# Appendix  
[U.S. Department of Commerce. (2024, May). Fact Sheet: President Biden Takes Action to Protect American Workers and Businesses from Unfair Chinese Trade Practices.](https://www.commerce.gov/news/fact-sheets/2024/05/fact-sheet-president-biden-takes-action-protect-american-workers-and)  
  
[Partington, R. (2025, April 3). Global markets in turmoil as Trump tariffs wipe $2.5tn off Wall Street. The Guardian.](https://www.theguardian.com/business/2025/apr/03/global-markets-turmoil-trump-tariffs-wall-street-downturn)  
  
[The Guardian. (2025, April 5). Trump tariffs come into effect in 'seismic' shift to global trade.](https://www.theguardian.com/us-news/2025/apr/05/trump-tariffs-come-into-effect-in-seismic-shift-to-global-trade)  
  
[The Guardian. (2025, April 5). Ted Cruz warns of midterm 'bloodbath' if Trump tariffs cause a recession.](https://www.theguardian.com/us-news/2025/apr/05/ted-cruz-midterm-trump-tariffs-recession). 
  
[The Guardian. (2025, April 5). How will the world respond to Donald Trump's tariffs?](https://www.theguardian.com/business/2025/apr/05/fundamentally-wrong-brutal-and-paranoid-how-will-the-world-respond-to-donald-trumps-tariffs)  
  
[The Guardian. (2025, April 5). Jaguar Land Rover pauses shipments to US as Trump says impact of tariffs 'won't be easy'.](https://www.theguardian.com/us-news/live/2025/apr/05/trump-tariffs-global-economy-markets-latest-news-updates)  
  
[The Guardian. (2025, April 3). US stock markets see worst day since Covid pandemic after investors shaken by Trump tariffs.](https://www.theguardian.com/us-news/2025/apr/03/trump-tariffs-stock-market)  
  
[The Guardian. (2025, April 2). Trump goes full gameshow host to push his tariff plan - and nobody's a winner.](https://www.theguardian.com/us-news/2025/apr/02/trump-tariffs-white-house-sketch)

[APIs - Yahoo Developer Network](https://developer.yahoo.com/api/)  

[Federal Reserve Economic Data (FRED) Graph](https://fred.stlouisfed.org/graph/?g=1HjmI)

