# Project 5: Enabling Wealth Managers to Safeguard Assets with an Economic Narratives Generator
*Abirami, Ahmad, Clarissa, Dat, Ted*

---

## Problem Statement  
When you're a wealth manager, timing is critical whether you're deploying a winning investment strategy or safeguarding assets against recessions. This project attempts to predict the timing of the next recession to help wealth managers of all skill levels make timely decisions that can grow or save their clients' wealth. 

## The Question
How might we confidently predict the timing of the next recession so that wealth managers can make better decisions to grow or safeguard their clients' finances?

## The Audience

Wealth Managers of all skill levels (from professionals to amateurs managing personal finances) over age 18 who are managing US-based assets.

---

## Repository Structure

**code:**  
This folder contains several notebooks dedicated to creating a model to address the problem statement.

- Notebook [01](./code/01_recession-prediction_data_collection.ipynb) is where we collected data and saved the cleaned data as [economic_indicators.csv](./data/economic_indicators.csv).
- Notebook [02](./code/02_income_predictor.ipynb) is where we created a random forest model with sentiment analysis to generate economic narratives.

**data:**
This folder contains the datasets used to develop our model.

-  [Economic Indicators](./data/economic_indicators.csv)

**images:**
This folder contains data visualizations created during our data exploration, which can explain our findings to our audience.

- [Correlation Heatmap](./images/correlation_heatmap.png)
- [Indicators Over Time](./images/indicators_over_time.png)
- [Sentiment Distribution](./images/sentiment_distribution.png)
- [Sentiment Timeline](./images/sentiment_timeline.png)
- [External: Federal Reserve of Economic Data (FRED) Graph of recession indicators from January 1, 1948 to March 1, 2025](./images/fredgraph.png)

**other:**
README.md - The file you are reading right now. It contains the summaries, techniques used, data visualizations, conclusions, and recommended next steps of our project.

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

We considered using Recurrent Neural Networks as it is time based, but it usually needs lots of data, reshaping, the input of one layer should match input of the next - in other words, RNN requires a lot more work to generate narratives.

---

## Conclusions and Recommendations
We conducted sentiment analysis to identify sentiments of the economy. Then we used the sentiment analysis as one of the inputs for GPT-2 to generate the narration, to explain the results in a way that wealth managers without an economics background can understand.  

We conclude that our model answered the problem. Our model achieved an accuracy of 1.00 for sentiment classification, despite the phrasing in our generated narrations being imperfect, likely because we used an older, free version of GPT (GPT-2).

### Recommendations
None of us who worked on this project are licensed financial advisors, but we would recommend looking to what licensed experts have recommended in the past in similar economic upswings and downturns.

Generally speaking, if our model's narrative sentiment is negative, take measures to safeguard wealth. If the narrative sentiment is positive, consider shifting into wealth-growing strategies.

---

