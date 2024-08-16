# Financial-Market-News-Sentiment-Analysis



To create a Financial Market News Sentiment Analysis using Python and pandas, I'll outline the steps and provide the code. Here's how we can approach it:

### Project Outline
1. **Problem Definition**
2. **Data Collection**
3. **Data Preprocessing**
4. **Exploratory Data Analysis (EDA)**
5. **Sentiment Analysis**
6. **Model Building (if required)**
7. **Results and Conclusion**

### Code Implementation

Let's start by implementing the code:

```python
# Step 1: Importing Libraries
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the Dataset
url = "https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Financial%20Market%20News.csv"
df = pd.read_csv(url)

# Step 3: Data Preprocessing
# Drop any rows with missing values
df.dropna(inplace=True)

# Step 4: Sentiment Analysis
# Function to get the subjectivity
def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# Function to get the polarity
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

# Adding columns for Subjectivity and Polarity
df['Subjectivity'] = df['News'].apply(get_subjectivity)
df['Polarity'] = df['News'].apply(get_polarity)

# Function to classify sentiment
def get_sentiment(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

# Adding a column for sentiment
df['Sentiment'] = df['Polarity'].apply(get_sentiment)

# Step 5: Exploratory Data Analysis (EDA)
# Plotting the sentiment distribution
plt.figure(figsize=(10, 5))
sns.countplot(x='Sentiment', data=df, palette='viridis')
plt.title('Sentiment Distribution')
plt.show()

# Display the first few rows of the dataframe
df.head()

# Step 6: Sentiment Analysis Results
# Aggregating by date to see the sentiment over time
df['Date'] = pd.to_datetime(df['Date'])
sentiment_over_time = df.groupby(df['Date'].dt.date)['Sentiment'].value_counts().unstack().fillna(0)

# Plotting sentiment over time
plt.figure(figsize=(15, 7))
sentiment_over_time.plot(kind='line', marker='o')
plt.title('Sentiment Over Time')
plt.xlabel('Date')
plt.ylabel('Frequency')
plt.show()

# Step 7: Conclusion
# Summary statistics
summary = df['Sentiment'].value_counts(normalize=True) * 100
print(summary)
```

### Explanation of Each Step:

1. **Importing Libraries:** The necessary libraries such as `pandas` for data manipulation, `TextBlob` for sentiment analysis, and `matplotlib` & `seaborn` for visualization are imported.

2. **Load the Dataset:** The dataset is loaded from the provided URL into a pandas DataFrame.

3. **Data Preprocessing:** We drop any missing values to clean the data.

4. **Sentiment Analysis:** 
   - We define functions to calculate `Subjectivity` and `Polarity` using TextBlob.
   - Based on the polarity score, sentiment is classified as 'Positive', 'Neutral', or 'Negative'.

5. **Exploratory Data Analysis (EDA):** 
   - We visualize the distribution of sentiments using a count plot.
   - Further, we examine sentiment trends over time.

6. **Results and Conclusion:**
   - Sentiment distribution is displayed.
   - Sentiment trends over time are plotted.

### Final Notes:
- This code uses the `TextBlob` library for sentiment analysis, which is a basic approach and might not be as sophisticated as models trained on financial text specifically.
- You can expand this project by including more advanced models like VADER, or BERT for financial text, depending on the complexity required. 

## CONCLUSION 
   The sentiment analysis reveals a dominant sentiment trend in financial news, indicating overall market optimism or caution during the analyzed period. Positive sentiment suggests a bullish outlook, while negative sentiment could signal market concerns. Significant sentiment shifts often correlate with key financial events, providing insights into market behavior. However, the basic sentiment analysis used here may not fully capture financial text nuances. Future work could involve more advanced models and real-time data for more precise predictions, aiding investors in making informed decisions based on market sentiment trends.
