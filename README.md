# 📊 Flipkart Home Appliances Sentiment Analysis

<p align="center">
  <img src="https://github.com/Nandaniipriya/Flipkart-Reviews-Sentiment-Analysis/raw/main/assests/Sentiment-Analysis.png" alt="Banner Image" width="80%">
</p>

## 📝 Overview
This project analyzes customer sentiments from Flipkart's Home Appliances category reviews using Python and various data analytics techniques. The analysis provides insights into customer satisfaction levels and overall product perception through sentiment analysis of reviews and ratings.

<p align="center">
  <img src="https://github.com/Nandaniipriya/Flipkart-Reviews-Sentiment-Analysis/raw/main/assests/sample.png" alt="Sample" width="80%">
</p>

## 🎯 Key Objectives
- Analyze customer rating patterns on Flipkart
- Identify key sentiment patterns in customer reviews
- Determine the predominant sentiment across the platform
- Draw comprehensive conclusions about customer perceptions

## 🔧 Technologies Used
- Python 3.12
- Pandas (Data manipulation)
- NLTK (Natural Language Processing)
- Matplotlib & Seaborn (Visualization)
- WordCloud (Text visualization)
- scikit-learn (Machine Learning)

## 📈 Key Findings

### Rating Distribution
- Nearly 50% of buyers gave 5-star ratings
- Over 20% gave 4-star ratings
- Overall 70% of customers showed high satisfaction

<p align="center">
  <img src="https://github.com/Nandaniipriya/Flipkart-Reviews-Sentiment-Analysis/raw/main/assests/accuracy.png" alt="Result-1" width="80%">
</p>

### Sentiment Analysis Results
- Positive Sentiment: 47.47%
- Neutral Sentiment: 46.73%
- Negative Sentiment: 5.80%

<p align="center">
  <img src="https://github.com/Nandaniipriya/Flipkart-Reviews-Sentiment-Analysis/raw/main/assests/confusion-matrix.png" alt="Result-2" width="80%">
</p>


## 🚀 Features
1. **Data Cleaning & Preprocessing**
   - Removal of duplicates
   - Text cleaning and normalization
   - Handling missing values

2. **Sentiment Analysis**
   - VADER sentiment scoring
   - Positive/Negative/Neutral classification
   - Sentiment intensity analysis

3. **Visualizations**
   - Word clouds of review text
   - Rating distribution charts
   - Sentiment distribution pie charts

## 💻 Code Structure
```python
# Key libraries
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
```

## 📊 Sample Analysis Code
```python
# Sentiment Analysis
sentiments = SentimentIntensityAnalyzer()
df['Positive'] = [sentiments.polarity_scores(i)['pos'] for i in df['Summary']]
df['Neutral'] = [sentiments.polarity_scores(i)['neu'] for i in df['Summary']]
df['Negative'] = [sentiments.polarity_scores(i)['neg'] for i in df['Summary']]
```

## 📋 Requirements
- Python 3.12
- pandas
- nltk
- matplotlib
- seaborn
- wordcloud
- scikit-learn

## 🚀 Getting Started

1. Clone the repository
```bash
git clone https://github.com/yourusername/flipkart-sentiment-analysis.git
```

2. Download NLTK data
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
```

## 📊 Results
The analysis reveals high customer satisfaction with Flipkart's home appliances:
- Most reviews express positive or neutral sentiments
- Very low percentage of negative reviews (5.80%)
- Strong correlation between high ratings and positive sentiment scores

## 🔄 Future Improvements
- Implement deep learning models for sentiment analysis
- Add real-time analysis capabilities
- Include price analysis correlation with sentiments
- Expand to other product categories

## 👥 Contributors
- Nandani Priya

## 📬 Contact
For any queries or suggestions, please reach out to:
- 📧 Email: nandani15p@gmail.com
- 💻 GitHub: https://github.com/Nandaniipriya

---
⭐ Don't forget to star this repo if you found it helpful!