import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the FNSPID dataset - adjust the path as needed
df = pd.read_csv('../data/fnspid.csv', parse_dates=['date'])
print(df.shape)        # Check dataset dimensions
df.head()              # Preview first few rows
df.info()              # Get data types and missing values
# Calculate headline character length
df['headline_len'] = df['headline'].astype(str).str.len()

# Display descriptive statistics
print(df['headline_len'].describe())

# Visualize the distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['headline_len'], bins=50, kde=True)
plt.title('Distribution of Headline Lengths')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.show()
# Top 10 most active publishers
publisher_counts = df['publisher'].value_counts().head(10)
print("Top 10 Publishers by Article Count:")
print(publisher_counts)

# Horizontal bar chart
plt.figure(figsize=(10, 6))
publisher_counts.plot(kind='barh')
plt.title('Top 10 Most Active Publishers')
plt.xlabel('Number of Articles')
plt.show()

# Extract domains from email-like publisher names
df['publisher_domain'] = df['publisher'].str.extract(r'@([\w\-\.]+)')
domains = df['publisher_domain'].dropna().value_counts().head(10)
print("Most Common Publisher Domains:")
print(domains)
# Set date as index for time series analysis
df.set_index('date', inplace=True)

# Resample to daily frequency and count articles
daily_volume = df.resample('D').size()

# Plot daily news volume
plt.figure(figsize=(12, 6))
daily_volume.plot()
plt.title('Daily News Publication Volume')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.show()

# Identify spike days (> 2 standard deviations above mean)
mean_volume = daily_volume.mean()
std_volume = daily_volume.std()
spike_days = daily_volume[daily_volume > mean_volume + 2*std_volume]
print("Potential spike days with unusually high news volume:")
print(spike_days)

# Analyze publication hour patterns
df['hour'] = df.index.hour
hourly_pattern = df.groupby('hour').size()

plt.figure(figsize=(12, 6))
hourly_pattern.plot(kind='bar')
plt.title('News Publication Pattern by Hour of Day (UTC-4)')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Articles')
plt.show()
# Reset index to access headline column easily
df_reset = df.reset_index()

# Create document-term matrix with unigrams and bigrams
vectorizer = CountVectorizer(stop_words=stop_words, max_features=30, ngram_range=(1,2))
X = vectorizer.fit_transform(df_reset['headline'].fillna(''))

# Get top keywords and phrases
sum_words = X.sum(axis=0)
words_freq = [(word, sum_words[0, i]) for word, i in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:20]

print("Top 20 Keywords and Phrases in Headlines:")
for word, freq in words_freq:
    print(f"{word}: {freq}")

# Optional: LDA Topic Modeling
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=5, random_state=42, max_iter=100)
lda.fit(X)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx+1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words-1:-1]]))

feature_names = vectorizer.get_feature_names_out()
display_topics(lda, feature_names, 10)
