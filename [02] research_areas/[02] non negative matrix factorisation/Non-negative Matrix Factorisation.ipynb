# Databricks notebook source
# MAGIC %md
# MAGIC ## Non-negative Matrix Factorisation (NMF) with Python
# MAGIC **Author:** [Vijay Choubey](https://medium.com/voice-tech-podcast/topic-modelling-using-nmf-2f510d962b6e) <br>
# MAGIC **Modified by:** [Lehlohonolo Makoti](https://github.com/lmakoti) <br>
# MAGIC **Notebook/Application:** [Databricks Notebooks](https://docs.databricks.com/en/notebooks/index.html)
# MAGIC
# MAGIC ## What is Non-Negative Matrix Factorisation
# MAGIC **Non-Negative Matrix Factorisation** is a statistical method to reduce the dimensions of the input corpora. It uses factor analysis method to provide comparatively less weightage to the words with less coherence.
# MAGIC
# MAGIC **Translation:** Non-Negative Matrix Factorisation (NMF) is a technique used to simplify large sets of data. It works by breaking down the data into smaller parts and focuses more on the parts that make the most sense together, while giving less importance to less relevant parts. This method helps in understanding and analysing large and complex data more easily.
# MAGIC
# MAGIC **For the math behind NMF visit:** https://medium.com/voice-tech-podcast/topic-modelling-using-nmf-2f510d962b6e
# MAGIC
# MAGIC The techniques discussed include:
# MAGIC - Generalized Kullback–Leibler divergence
# MAGIC - Frobenius Form (Euclidean Norm)
# MAGIC
# MAGIC Optimisation is required to achieve high accuracy in finding relation between the topics, this is pre-packaged in `scikit-learn`.
# MAGIC
# MAGIC ## Packages Required
# MAGIC
# MAGIC This walkthrough uses the following Python packages:
# MAGIC * [Numpy](https://pypi.org/project/numpy/), is the fundamental package for scientific computing with Python.<br>
# MAGIC For Mac/Unix with pip: `$ sudo pip install -U numpy`
# MAGIC * [Scikit-Learn](pip install scikit-learn), is a Python module for machine learning built on top of SciPy and is distributed under the 3-Clause BSD license.<br>
# MAGIC For Mac/Unix with pip: `$ sudo pip install scikit-learn`
# MAGIC
# MAGIC **Databricks:** `%pip install <package_name>`
# MAGIC
# MAGIC

# COMMAND ----------

# Importing Necessary packages
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Importing the Documents

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Loading text from the Scikit-Learn 20 NewsGroups dataset
# MAGIC The [20 newsgroups dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) comprises around 18000 newsgroups posts on 20 topics split in two subsets: one for training (or development) and the other one for testing (or for performance evaluation). The split between the train and test set is based upon a messages posted before and after a specific date.
# MAGIC
# MAGIC This dataset is a collection newsgroup documents. The 20 newsgroups collection has become a popular data set for experiments in text applications of machine learning techniques, such as text classification and text clustering.
# MAGIC
# MAGIC **Alternative Source:** [Kaggle 20 Newsgroups](https://www.kaggle.com/datasets/crawford/20-newsgroups)

# COMMAND ----------

# Importing Data
text_data= fetch_20newsgroups(remove=('headers', 'footers', 'quotes')).data
text_data[:2] # importing the first three articles

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Text Preprocessing
# MAGIC Preprocessing text is essential for Latent Dirichlet Allocation (LDA) and other natural language processing tasks for several reasons:
# MAGIC
# MAGIC * **Tokenisation:** This involves breaking down text into individual words or terms (tokens). It's a fundamental step for the LDA model to understand and process the text data.
# MAGIC
# MAGIC * **Reducing Dimensionality (Stopwords removal):** Text data can be very high-dimensional due to the vast number of unique words. Preprocessing steps like removing common stopwords (e.g., "the", "is", "and") reduces the dimensionality, making the LDA model more efficient and effective.
# MAGIC
# MAGIC * **Stemming/Lemmatization:** These processes reduce words to their root form. For example, "running", "ran", and "runs" would all be reduced to "run". This helps in consolidating the variations of a word into a single term, improving the model's ability to identify relevant topics.
# MAGIC
# MAGIC * **Removing Noise:** Text data often contains elements like special characters, punctuation, and numbers that may not be relevant for topic modeling. Removing these helps focus on meaningful words.
# MAGIC
# MAGIC * **Standardisation:** Converting all text to a standard format, typically lower case, ensures that the algorithm treats words like "Apple" and "apple" as the same word.
# MAGIC
# MAGIC * **Removing Rare Words:** Words that appear very infrequently may not be useful in identifying common themes and topics. Removing these can help improve the model's focus and performance.
# MAGIC
# MAGIC By preprocessing text, you essentially clean and refine the data, making it more suitable for the LDA model to analyze and draw meaningful conclusions about the underlying topics in the text.
# MAGIC
# MAGIC **NB:** for NMF while not strictly necessary, applying lemmatisation or stemming before NMF can be beneficial. It can improve the efficiency of the matrix factorisation and potentially lead to more coherent and interpretable results, especially when dealing with large and diverse text corpora. The choice between stemming and lemmatisation should be guided by the specific requirements of your analysis and the nature of your text data.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Tokenisation | Lowercasing
# MAGIC This `tokenizer` is often used in text processing to break text into words while filtering out punctuation and other non-word characters.

# COMMAND ----------

from nltk.tokenize import RegexpTokenizer
import re

tokenizer = RegexpTokenizer(r'\w+')
# standardisation/lowering the case
raw = str(text_data).lower() # convert to string because 'list' object has no attribute 'lower'
tokens = str(tokenizer.tokenize(raw))

# COMMAND ----------

# handling special text and characters
# Remove special characters
tokens = re.sub(r'[^\w\s]', '', tokens)

# Remove HTML/XML tags
tokens = re.sub(r'<[^>]+>', '', tokens)

# Remove phone numbers (US format)
tokens = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', tokens)

# Example of removing simple JSON/XML structures (not recommended for complex structures)
tokens = re.sub(r'\{.*?\}', '', tokens) # for simple JSON
tokens = re.sub(r'<.*?>', '', tokens) # for simple XML

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Constructing a Document-Term Matrix
# MAGIC Now we convert a collection of text documents into a numeric form that machine learning algorithms can work with, focusing on the importance of words in the documents while filtering out common words and limiting the total number of words to manage.

# COMMAND ----------

# Converting the given text term-document matrix
vectorizer = TfidfVectorizer(max_features=1500, min_df=10, stop_words='english')
X = vectorizer.fit_transform(text_data)
words = np.array(vectorizer.get_feature_names())

print(X)
print("X = ", words)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Applying Non-Negative Matrix Factorization
# MAGIC **Theory:** https://github.com/lmakoti/topic-modelling
# MAGIC
# MAGIC For a general case, consider we have an input matrix V of shape m x n. This method factorizes V into two matrices W and H, such that the dimension of W is m x k and that of H is n x k. For our situation, V represent the term document matrix, each row of matrix H is a word embedding and each column of the matrix W represent the weightage of each word get in each sentences ( semantic relation of words with each sentence). You can find a practical application with example below.
# MAGIC
# MAGIC But the assumption here is that all the entries of W and H is positive given that all the entries of V is positive.
# MAGIC <br><br>
# MAGIC <p align="center"><img src="https://miro.medium.com/v2/resize:fit:640/format:webp/0*uz3OkHMgjAH2Yc40.png"/></p>

# COMMAND ----------

nmf = NMF(n_components=10, solver="mu")
W = nmf.fit_transform(X)
H = nmf.components_

for i, topic in enumerate(H):
     print("Topic {}: {}".format(i + 1, ",".join([str(x) for x in words[topic.argsort()[-10:]]])))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Examining the Results
# MAGIC When can we use this approach:
# MAGIC - NMF by default produces sparse representations. This mean that most of the entries are close to zero and only very few parameters have significant values. This can be used when we strictly require fewer topics.
# MAGIC - NMF produces more coherent topics compared to LDA.

# COMMAND ----------

# W matrix is given below.
print(W[:10,:10])

# COMMAND ----------

# H matrix can be printed as shown below.
print(H[:10,:10])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Putting Everything Together

# COMMAND ----------

# Importing Data
text_data= fetch_20newsgroups(remove=('headers', 'footers', 'quotes')).data
text_data[:2] # importing the first three articles

# Importing Data
text_data= fetch_20newsgroups(remove=('headers', 'footers', 'quotes')).data
text_data[:3]

# converting the given text term-document matrix
vectorizer = TfidfVectorizer(max_features=1500, min_df=10, stop_words='english')
X = vectorizer.fit_transform(text_data)
words = np.array(vectorizer.get_feature_names())

# print(X)
# print("X = ", words)

# Applying Non-Negative Matrix Factorization
nmf = NMF(n_components=10, solver="mu")
W = nmf.fit_transform(X)
H = nmf.components_

for i, topic in enumerate(H):
     print("Topic {}: {}".format(i + 1, ",".join([str(x) for x in words[topic.argsort()[-10:]]])))

# COMMAND ----------

print(W[:10,:10])

# COMMAND ----------

print(H[:10,:10])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Visualise the Results

# COMMAND ----------

import matplotlib.pyplot as plt

# Assuming 'H' is your topic matrix and 'words' is your array of words
for i, topic in enumerate(H):
    top_words_indices = topic.argsort()[-10:]  # Indices of top 10 words in this topic
    top_words = [words[j] for j in top_words_indices]  # Top 10 words
    top_words_weights = [topic[j] for j in top_words_indices]  # Weights of top 10 words

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.barh(range(10), top_words_weights, align='center')
    plt.yticks(range(10), top_words)
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest value on top
    plt.title(f'Topic {i + 1}')
    plt.xlabel('Weights')
    plt.ylabel('Words')
    plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import random

# Assuming 'H' is your topic matrix and 'words' is your array of words
for i, topic in enumerate(H):
    # Generate a dictionary of word frequencies for this topic
    word_freq = {words[j]: topic[j] for j in topic.argsort()[-10:]}

    # Normalise frequencies for better visualisation
    max_freq = max(word_freq.values())
    word_freq = {word: freq / max_freq for word, freq in word_freq.items()}

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.title(f'Topic {i + 1}')
    plt.axis('off')
    for word, freq in word_freq.items():
        plt.text(random.uniform(0, 1), random.uniform(0, 1), word, 
                 ha='center', va='center',
                 fontsize=freq * 40)  # Adjust font size based on frequency

    plt.show()

