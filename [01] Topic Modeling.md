

<img src="G:\My Drive\Colab Notebooks\[01] Research Papers\assets\notes_logo.png" alt="notes_logo" style="zoom:43%;" />

## What is Topic Modelling

**Topic modelling** is a type of statistical model for discovering the abstract "topics" that occur in a collection of documents. It's a common tool in the field of natural language processing (NLP) and text mining. Here's a more detailed look at what topic modelling involves:

1. **Purpose**: Topic modelling is used to uncover hidden thematic structures in a large corpus of text. Essentially, it helps in organizing and understanding large collections of textual information.
2. **How it Works**: It identifies patterns of word usage across documents and groups words that frequently occur together. By doing so, it can highlight the recurring themes or topics across the documents.
3. **Topics**: In this context, a "topic" is a recurring pattern of words. For example, in a collection of news articles, you might find topics related to politics, sports, economics, etc. Each topic is represented as a collection of words with certain probabilities of occurring in that topic.
4. **Common Models**: One of the most popular topic modelling techniques is Latent Dirichlet Allocation (LDA). Other methods include Non-Negative Matrix Factorization (NMF) and Latent Semantic Analysis (LSA).
5. **Applications**: Topic modelling is used in various fields such as in digital humanities for analysing large archives of texts, in customer service to categorize complaints and queries, in content recommendation systems to understand user preferences, and in academic research for literature review and analysis.
6. **Limitations**: While powerful, topic modelling has limitations. It doesn't capture the meaning of words (semantics), the order of words (syntax), or understand context beyond co-occurrence patterns. It also requires a good amount of domain knowledge to interpret the topics correctly.
7. **Interpretation**: The interpretation of the topics and their relevance is subjective and depends on the user's knowledge and the context of the dataset. Fine-tuning parameters and understanding the domain are crucial for effective topic modelling.

In summary, topic modelling is a method for automatically organizing, understanding, and summarizing large collections of textual data by discovering patterns of word usage that suggest themes or topics within the data.



## Term Frequency - Inverse Document Frequency

**TF-IDF, which stands for Term Frequency-Inverse Document Frequency**, is a statistical measure used to evaluate how important a word is to a document in a collection or corpus of documents. It's often used in text mining and information retrieval. Here's a simple breakdown of what TF-IDF means and how it works:

1. **Term Frequency (TF)**: This is simply how often a word appears in a document. The idea is that the more often a word appears in a document, the more important it is for that document. For example, if the word "apple" appears five times in a document, its term frequency is higher compared to a word that appears only once.

2. **Inverse Document Frequency (IDF)**: This measures how common or rare a word is across all documents in the corpus. It helps to identify if a word is common or unique in the entire corpus. The IDF increases when the word is rare and decreases when the word is common. For example, the word "the" might appear in almost every document, so its IDF is low, whereas a specific term like "photosynthesis" might appear in fewer documents, giving it a higher IDF.

3. **Combining TF and IDF**: TF-IDF is calculated by multiplying the TF of a word by its IDF. This gives a higher score to words that are more relevant (i.e., appear frequently in a particular document but not too frequently across all documents). This way, TF-IDF tends to filter out common words like "is", "are", "and", etc., and gives higher importance to words that are more specific to a particular document.

In summary, TF-IDF allows us to evaluate words in documents not just by how often they appear, but also by considering their importance across a larger set of documents, helping to highlight the most relevant words in each document.

### Term Frequency (TF) - Document Focused

Term Frequency measures how frequently a term occurs in a document. The simplest formula for TF is:

$$
\text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in a document } d}{\text{Total number of terms in the document } d}
$$
This formula calculates the TF of a term \( t \) in a document \( d \) as the frequency of \( t \) divided by the total number of terms in \( d \).

### Inverse Document Frequency (IDF) - Corpus Focused

Inverse Document Frequency measures how important a term is within the whole corpus of documents. The formula for IDF is:

$$
 \text{IDF}(t, D) = \log \left( \frac{\text{Total number of documents in the corpus } D}{\text{Number of documents with term } t \text{ in them}} \right) \
$$
Here, \( D \) is the total number of documents in the corpus, and the denominator is the number of documents where the term \( t \) appears. The logarithm of the quotient is used to dampen the effect of IDF.

### TF-IDF - Combination (Scoring)

TF-IDF is simply the product of TF and IDF:

$$
\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
$$
This formula combines the frequency of a term in a specific document (TF) with the inverse of its frequency across all documents (IDF), to calculate a weight that represents the importance of the term \( t \) in a specific document \( d \) within the context of a document corpus \( D \).

In summary, TF-IDF increases with the number of times a word appears in a document but is offset by the frequency of the word in the corpus, which helps to control for the fact that some words are generally more common than others.

**Example**:

**Assumptions:** d represents documents and associated document number `n` and `D` is the corpus of all documents `d`.

d1 = `[Thabo is a key player in our team he is the best we have]`
d2 = `[He is the best player in the entire world]`
d3 = `[is it really ideal to label players by skill rank]`

D = `[d1,d2,d3]`

t = `[is]`

TF(t,d) = `count(t)/count(d)` = 2/15 = 0.133

IDF(t, D) = `log(count(D)/count(t))` = log(3/4) = 0.602

### 1. Latent Dirichlet Allocation - Model

**Latent Dirichlet Allocation (LDA)** is a statistical model commonly used for topic modelling in natural language processing (NLP). It helps in discovering abstract topics within a collection of documents. Here's a simple way to understand LDA:

1. **Starting Assumption**: LDA starts with a basic assumption: every document is a mixture of different topics, and each topic is a collection of various words.
2. **Hidden Topics**: The 'latent' part of LDA refers to the fact that these topics are hidden or not directly observable. The model aims to uncover these hidden topics.
3. **Document as a Mixture of Topics**: Imagine you have a bunch of documents. LDA assumes that each document is like a bag of topics. For instance, a news article might be 30% politics, 20% economics, and 50% international relations.
4. **Topics as a Mixture of Words**: Each topic is also considered a bag of words. For example, the 'politics' topic might be made up of words like 'election', 'government', 'policy', etc., each with a certain probability of appearing in that topic.
5. **The LDA Process**: When you apply LDA to a set of documents, it goes through each document and randomly assigns each word in the document to one of the K topics (where K is a predetermined number of topics). These random assignments give both initial topic representations for all the documents and word distributions for all the topics.
6. **Iterative Refinement**: LDA then iteratively updates these assignments to make them more reasonable. It looks at how frequently each word appears in each topic (across all documents) and how often each topic appears in each document. Based on this, it reassigns words to topics in a way that words are more likely to be in topics where they are common, and documents are more likely to contain topics that they already have a lot of.
7. **Final Output**: After many iterations, the model reaches a steady state where the topic assignments are relatively stable. The end result is that each document in your collection is described as a mixture of topics, and each topic is characterized by a set of words.

In essence, LDA provides a way to automatically discover topics from a large collection of documents, making it a powerful tool for understanding large text datasets.



### 2. Non-Negative Matrix Factorization (NMF) 

**Non-Negative Matrix Factorization (NMF)** is a mathematical method used for data analysis, often in the context of pattern recognition and machine learning. Here's a simple way to understand it:

1. **Matrix Factorization**: Imagine you have a large matrix (a table of numbers) where you have data about something, like people and their preferences for different movies. Matrix factorization is about breaking this large matrix into smaller, more manageable matrices that, when multiplied together, approximately recreate the original matrix.

2. **Non-Negative**: The "non-negative" part means that all the numbers in these matrices are zero or positive. This constraint makes NMF particularly useful for data where negative numbers don’t make sense, like counts of words in a document or the intensity of pixels in an image.

3. **Discovering Patterns**: NMF is good at finding patterns in the data. In the case of movies and people, one of the smaller matrices might represent how much each person likes different genres (like action, romance, comedy), and the other matrix might show how much of each genre is in each movie.

4. **Applications**: NMF is widely used in text mining (like topic modeling), image processing, and recommendation systems. For instance, in topic modeling, NMF can help discover topics (patterns of words) in a collection of documents.

5. **Working Mechanism**: The idea is to start with initial guesses for the smaller matrices and then iteratively adjust them to get as close as possible to the original matrix, under the constraint that all values must be non-negative.

6. **Interpretation**: The results of NMF are often easier to interpret than other matrix factorization methods because of the non-negativity constraint. This makes the components more representative of actual features in the data, like topics in documents or genres in movies.

In essence, NMF is a way to uncover hidden patterns in complex data sets by breaking down a large matrix into smaller, meaningful components, all while ensuring that the data stays in the realm of the non-negative.



## 3. Latent Semantic Model (LSM)

Latent Semantic Analysis (LSA), also known as Latent Semantic Indexing, is a technique in natural language processing and information retrieval that helps to discover hidden (or "latent") relationships between words in large collections of text. Here's a simplified explanation:

1. **Starting Point - Term-Document Matrix**: LSA begins with the creation of a term-document matrix. In this matrix, each row represents a unique word in the corpus (the entire set of documents), and each column represents a document. The values in the matrix indicate the frequency or occurrence of the words in each document.

2. **The Problem of Synonyms and Polysemy**: In language, the same idea can be expressed with different words (synonyms), and the same word can have multiple meanings (polysemy). This makes it hard for a computer to understand text based on word counts alone.

3. **Using Singular Value Decomposition (SVD)**: LSA applies a mathematical technique called Singular Value Decomposition to the term-document matrix. SVD reduces the number of rows (words) while preserving the similarity structure among columns (documents). Essentially, it compresses the matrix into a smaller one that captures the most important relationships.

4. **Discovering Concepts**: This compressed matrix doesn't just represent individual words; it represents concepts or topics. Words that often appear together in documents (like "river" and "water") will be closer in this conceptual space, even if they don't always appear in the same documents.

5. **Applications**: LSA is used for various tasks like document classification, clustering, information retrieval, and even in developing chatbots and recommendation systems. It helps computers to process and understand large volumes of text by identifying patterns and relationships in the words.

6. **Limitations**: While powerful, LSA has its limitations. It doesn't account for word order and can sometimes mix topics or meanings if the synonym and polysemy issues are strong.

In summary, Latent Semantic Analysis is a method of analysing relationships between a set of documents and the terms they contain by producing a set of concepts related to the documents and terms. It's a way of extracting the underlying meaning or concept from a collection of text, helping in better understanding and processing the information.



| Feature/Model         | LDA (Latent Dirichlet Allocation)                            | NMF (Non-Negative Matrix Factorization)                      | LSA (Latent Semantic Analysis)                               |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Basic Concept         | Probabilistic, generative model                              | Matrix factorization technique with non-negativity constraints | Singular Value Decomposition (SVD) of term-document matrix   |
| Approach              | Assumes documents are mixtures of topics, which are mixtures of words | Factorizes matrices into two non-negative matrices (W and H) | Decomposes the term-document matrix to discover latent concepts |
| Key Characteristics   | Topics have a Dirichlet prior, providing a richer probabilistic model | Emphasizes part-based representation, better for parts-of-whole data | Efficient in capturing the variance in data, good for synonyms |
| Output Interpretation | Topics are probability distributions over words              | Learns the parts of a whole, each feature is only additive   | Topics are singular vectors, not probability distributions   |
| Handling of Data      | Better for discrete, count-based data (like word counts)     | Works well with count data but also adaptable for other data types | Suitable for continuous data, uses orthogonal linear algebra |
| Limitations           | Requires pre-setting the number of topics, sensitive to hyperparameters | Cannot model negative data, sensitive to the choice of features | Does not capture polysemy, affected by term frequency        |



| Feature/Model         | LDA (Latent Dirichlet Allocation)          | NMF (Non-Negative Matrix Factorization)           | LSA (Latent Semantic Analysis)                               |
| --------------------- | ------------------------------------------ | ------------------------------------------------- | ------------------------------------------------------------ |
| Basic Concept         | Probabilistic model for finding topics     | Breaks down data into parts without negatives     | Uses math to find hidden themes                              |
| Approach              | Mixes topics and words to create documents | Separates data into positive factors only         | Breaks down words and documents                              |
| Key Characteristics   | Uses probabilities for topics and words    | Only adds, never subtracts parts of data          | Good at finding similar words                                |
| Output Interpretation | Topics are mixes of words                  | Shows parts making up the data                    | Shows main themes in data                                    |
| Handling of Data      | Best for word counts                       | Good for counts and other data types              | Works well with varied data types                            |
| Limitations           | Fixed number of topics, sensitive settings | Can't handle negative numbers, sensitive settings | Struggles with words having multiple meanings, influenced by word counts |















































































































