import pandas as pd
import matplotlib.pyplot as plt
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK stopwords (if not already downloaded)
import nltk
nltk.download('stopwords')

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv('record.csv')

# Assuming 'reviews' column contains text data
# If not, you may need to replace 'reviews' with the appropriate column name

# Combine all reviews into a single string
text = ' '.join(df['review'].dropna())

# Tokenize the text into words
words = word_tokenize(text)

# Remove stopwords (common words like 'the', 'and', etc.)
stop_words = set(stopwords.words('english'))
filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

# Calculate word frequencies
word_freq = FreqDist(filtered_words)

# Select the top N words to categorize
top_words = word_freq.most_common(20)  # Change 20 to the desired number of top words

# Extract words and frequencies for plotting
words, frequencies = zip(*top_words)

# Calculate dynamic thresholds based on the size of the text
total_words = len(filtered_words)
most_use_threshold = total_words * 0.02  # 2% of total words
low_use_threshold = total_words * 0.005  # 0.5% of total words

# Categorize words into three groups based on their frequency
categories = ['most use' if freq > most_use_threshold else 'median use' if freq > low_use_threshold else 'low use' for freq in frequencies]

# Define colors for each category
colors = {'most use': 'green', 'median use': 'orange', 'low use': 'red'}

# Plot the bar chart with colored bars
plt.figure(figsize=(12, 6))
bars = plt.bar(words, frequencies, color=[colors[category] for category in categories])

# Add legend
legend_labels = [plt.Rectangle((0, 0), 1, 1, color=colors[category]) for category in colors]
plt.legend(legend_labels, colors.keys())

# Add title and labels
plt.title('Categorized Top Words in Reviews')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility

# Show the plot
plt.show()
