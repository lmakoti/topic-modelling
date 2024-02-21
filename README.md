## What is Topic Modelling

**Topic modelling** is a type of statistical model for discovering the abstract "topics" that occur in a collection of documents, a common tool in the field of natural language processing (NLP) and text mining. Its purpose is to uncover hidden thematic structures in a large text corpus, aiding in the organisation and understanding of vast collections of textual information. By identifying patterns of word usage across documents and grouping words that frequently occur together, topic modelling highlights recurring themes or topics within the documents. In this context, a "topic" represents a pattern of words, such as those related to politics, sports, or economics in a collection of news articles, each depicted as a collection of words with certain probabilities of occurrence within that topic.

One of the most popular topic modelling techniques is [Latent Dirichlet Allocation (LDA), introduced by Blei, Ng, and Jordan (2003)](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf), alongside other methods like Non-Negative Matrix Factorization (NMF) and [Latent Semantic Analysis (LSA), which were developed by Lee and Seung (1999) and Deerwester et al. (1990)](https://napier.primo.exlibrisgroup.com/permalink/44NAP_INST/19n0mho/cdi_openaire_primary_doi_dedup_e07b9809ac7d8f096992f8362d7a8a7b), respectively. Topic modelling has diverse applications across digital humanities, customer service, content recommendation systems, and academic research, facilitating the analysis of large text archives, categorisation of complaints and queries, understanding user preferences, and literature review and analysis.

However, topic modelling has limitations. **It does not capture the semantics or syntax of words, nor does it understand context beyond co-occurrence patterns, requiring substantial domain knowledge for accurate interpretation**. The interpretation of topics and their relevance is subjective, depending on the user's knowledge and dataset context, with parameter tuning and domain understanding being crucial for effective application.



## Term Frequency - Inverse Document Frequency

**TF-IDF, which stands for Term Frequency-Inverse Document Frequency** is another statistical measure used in text mining and information retrieval to evaluate the importance of a word to a document within a corpus. Term Frequency (TF) reflects how often a word appears in a document, suggesting that more frequent words carry more significance. Inverse Document Frequency (IDF) measures a word's commonality across all documents, with rare words receiving higher scores. The combination of TF and IDF, as discussed in foundational texts by Salton & McGill (1983) and further elaborated by Manning, Raghavan, & Schütze (2008), highlights words that are not only prevalent in a particular document but also rare across the entire corpus, thus filtering out common words and emphasizing document-specific terms.

1. **Term Frequency (TF)**: This is simply how often a word appears in a document. The idea is that the more often a word appears in a document, the more important it is for that document. For example, if the word "apple" appears five times in a document, its term frequency is higher compared to a word that appears only once.

2. **Inverse Document Frequency (IDF)**: This measures how common or rare a word is across all documents in the corpus. It helps identify whether a word is common or unique in the entire corpus. The IDF increases when the word is rare and decreases when the word is common. For example, the word "the" might appear in almost every document, so its IDF is low. In contrast, a specific term like "photosynthesis" might appear in fewer documents, giving it a higher IDF.

3. **Combining TF and IDF**: TF-IDF is calculated by multiplying the TF of a word by its IDF. This gives a higher score to more relevant words (i.e., appear frequently in a particular document but not too frequently across all documents). This way, TF-IDF tends to filter out common words like "is", "are", "and", etc., and gives higher importance to words that are more specific to a particular document.

In summary, TF-IDF allows us to evaluate words in documents by how often they appear and by considering their importance across a larger set of documents, helping to highlight the most relevant words in each document.

### Term Frequency (TF) - Document Focused

Term Frequency measures how frequently a term occurs in a document. The simplest formula for TF is:

$$\text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in a document } d}{\text{Total number of terms in the document } d}$$
This formula calculates the TF of a term \( t \) in a document \( d \) as the frequency of \( t \) divided by the total number of terms in \( d \).

### Inverse Document Frequency (IDF) - Corpus Focused

Inverse Document Frequency measures how important a term is within the whole corpus of documents. The formula for IDF is:

$$\text{IDF}(t, D) = \log \left( \frac{\text{Total number of documents in the corpus } D}{\text{Number of documents with term } t \text{ in them}} \right) \$$
Here, \( D \) is the total number of documents in the corpus, and the denominator is the number of documents where the term \( t \) appears. The logarithm of the quotient is used to dampen the effect of IDF.

### TF-IDF - Combination (Scoring)

TF-IDF is simply the product of TF and IDF:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
This formula combines the frequency of a term in a specific document (TF) with the inverse of its frequency across all documents (IDF), to calculate a weight that represents the importance of the term \( t \) in a specific document \( d \) within the context of a document corpus \( D \).

In summary, TF-IDF increases with the number of times a word appears in a document but is offset by the frequency of the word in the corpus, which helps to control for the fact that some words are generally more common than others.

**Example**:

**Assumptions:** d represents documents and associated document number `n` , and `D` is the corpus of all documents `d`.

d1 = `[Thabo is a key player in our team he is the best we have]`<br/>
d2 = `[He is the best player in the entire world]`<br/>
d3 = `[is it really ideal to label players by skill rank]`<br/>

D = `[d1,d2,d3]`

t = `[is]`

TF(t,d) = `count(t)/count(d)` = 2/15 = 0.133

IDF(t, D) = `log(count(D)/count(t))` = log(3/4) = 0.602

### 1. Latent Dirichlet Allocation - Model

**Latent Dirichlet Allocation (LDA)** is a statistical model commonly used for topic modelling in natural language processing (NLP). It helps in discovering abstract topics within a collection of documents. Here's a simple way to understand LDA:

1. **Starting Assumption**: LDA starts with a basic assumption: every document is a mixture of different topics, and each topic is a collection of various words.
2. **Hidden Topics**: The 'latent' part of LDA refers to whether these topics are hidden or not directly observable. The model aims to uncover these hidden topics.
3. **Document as a Mixture of Topics**: Imagine you have many documents. LDA assumes that each document is like a bag of topics. For instance, a news article might be 30% politics, 20% economics, and 50% international relations.
4. **Topics as a Mixture of Words**: Each topic is also considered a bag of words. For example, the topic of 'politics' might be comprised of words like 'election', 'government', 'policy', etc., each with a certain probability of appearing in that topic.
5. **The LDA Process**: When you apply LDA to a set of documents, it goes through each document and randomly assigns each word in the document to one of the K topics (where K is a predetermined number of topics). These random assignments give both initial topic representations for all the documents and word distributions for all the topics.
6. **Iterative Refinement**: LDA then iteratively updates these assignments to make them more reasonable. It looks at how frequently each word appears in each topic (across all documents) and how often each topic appears in each document. Based on this, it reassigns words to topics so that words are more likely to be in topics where they are common, and documents are more likely to contain topics that they already have a lot of.
7. **Final Output**: After many iterations, the model reaches a steady state where the topic assignments are relatively stable. The result is that each document in your collection is described as a mixture of topics, and a set of words characterises each topic.

LDA provides a way to automatically discover topics from a large collection of documents, making it a powerful tool for understanding large text datasets.

**Python Implementation:** [https://github.com/lmakoti/topic-modelling/lda-nltk-python](https://github.com/lmakoti/topic-modelling/blob/main/%5B02%5D%20research_areas/%5B01%5D%20latent_dirichlet_allocation/Latent%20Dirichlet%20Allocation%20(Python).ipynb)

### 2. Non-Negative Matrix Factorization (NMF) 

**Non-Negative Matrix Factorization (NMF)** is a mathematical method used for data analysis, often in the context of pattern recognition and machine learning. Here's a simple way to understand it:

1. **Matrix Factorization**: Imagine you have a large matrix (a table of numbers) with data about something, like people and their preferences for different movies. Matrix factorisation is about breaking this large matrix into smaller, more manageable matrices that, when multiplied together, approximately recreate the original matrix.

2. **Non-Negative**: The "non-negative" part means that all the numbers in these matrices are zero or positive. This constraint makes NMF particularly useful for data where negative numbers don’t make sense, like counts of words in a document or the intensity of pixels in an image.

3. **Discovering Patterns**: NMF is good at finding patterns in the data. In the case of movies and people, one of the smaller matrices might represent how much each person likes different genres (like action, romance, comedy), and the other matrix might show how much of each genre is in each movie.

4. **Applications**: NMF is widely used in text mining (like topic modelling), image processing, and recommendation systems. For instance, in topic modelling, NMF can help discover topics (patterns of words) in a collection of documents.

5. **Working Mechanism**: The idea is to start with initial guesses for the smaller matrices and then iteratively adjust them to get as close as possible to the original matrix, under the constraint that all values must be non-negative.

6. **Interpretation**: The results of NMF are often easier to interpret than other matrix factorisation methods because of the non-negativity constraint. This makes the components more representative of actual features in the data, like topics in documents or genres in movies.

In essence, NMF is a way to uncover hidden patterns in complex data sets by breaking down a large matrix into smaller, meaningful components, all while ensuring that the data stays in the realm of the non-negative.

**Python Implementation:** [https://github.com/lmakoti/topic-modelling/nmf-scikit-learn-python](https://github.com/lmakoti/topic-modelling/blob/main/%5B02%5D%20research_areas/%5B02%5D%20non-negative%20matrix%20factorisation/Non-negative%20Matrix%20Factorisation.ipynb)

## 3. Latent Semantic Model (LSM)

Latent semantic analysis (LSA), also known as latent semantic indexing, is a technique in natural language processing and information retrieval that helps discover hidden (or "latent") relationships between words in large collections of text. Here's a simplified explanation:

1. **Starting Point - Term-Document Matrix**: LSA begins with creating a term-document matrix. In this matrix, each row represents a unique word in the corpus (the entire set of documents), and each column represents a document. The values in the matrix indicate the frequency or occurrence of the words in each document.

2. **The Problem of Synonyms and Polysemy**: The same idea can be expressed with different words (synonyms), and the same word can have multiple meanings (polysemy). This makes it hard for a computer to understand text based on word counts alone.

3. **Using Singular Value Decomposition (SVD)**: LSA applies a mathematical technique called Singular Value Decomposition to the term-document matrix. SVD reduces the number of rows (words) while preserving the similarity structure among columns (documents). It compresses the matrix into a smaller one that captures the most important relationships.

4. **Discovering Concepts**: This compressed matrix represents not just individual words but concepts or topics. Words that often appear together in documents (like "river" and "water") will be closer in this conceptual space, even if they don't always appear in the same documents.

5. **Applications**: LSA is used for various tasks like document classification, clustering, information retrieval, and even in developing chatbots and recommendation systems. It helps computers process and understand large volumes of text by identifying word patterns and relationships.

6. **Limitations**: While powerful, LSA has its limitations. It doesn't account for word order and can sometimes mix topics or meanings if the synonym and polysemy issues are strong.

In summary, Latent Semantic Analysis is a method of analysing relationships between a set of documents and the terms they contain by producing a set of concepts related to the documents and terms. It's a way of extracting the underlying meaning or concept from a text collection, helping better understand and process the information.



## 4. Top2Vec







## 5. BERTopic







### Comparison of LDA, NMF and LSM (Complex)

| Feature/Model         | LDA (Latent Dirichlet Allocation)                            | NMF (Non-Negative Matrix Factorization)                      | LSA (Latent Semantic Analysis)                               | Top2Vec | BERTopic |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------- | -------- |
| Basic Concept         | Probabilistic, generative model                              | Matrix factorization technique with non-negativity constraints | Singular Value Decomposition (SVD) of term-document matrix   |         |          |
| Approach              | Assumes documents are mixtures of topics, which are mixtures of words | Factorizes matrices into two non-negative matrices (W and H) | Decomposes the term-document matrix to discover latent concepts |         |          |
| Key Characteristics   | Topics have a Dirichlet prior, providing a richer probabilistic model | Emphasises part-based representation, better for parts-of-whole data | Efficient in capturing the variance in data, good for synonyms |         |          |
| Output Interpretation | Topics are probability distributions over words              | Learns the parts of a whole, each feature is only additive   | Topics are singular vectors, not probability distributions   |         |          |
| Handling of Data      | Better for discrete, count-based data (like word counts)     | Works well with count data but also adaptable for other data types | Suitable for continuous data, uses orthogonal linear algebra |         |          |
| Limitations           | Requires pre-setting the number of topics, sensitive to hyperparameters | Cannot model negative data, sensitive to the choice of features | Does not capture polysemy, affected by term frequency        |         |          |

### Comparison of LDA, NMF and LSM (Simplified)

| Feature/Model         | LDA (Latent Dirichlet Allocation)          | NMF (Non-Negative Matrix Factorization)           | LSA (Latent Semantic Analysis)                               | Top2Vec | BERTopic |
| --------------------- | ------------------------------------------ | ------------------------------------------------- | ------------------------------------------------------------ | ------- | -------- |
| Basic Concept         | Probabilistic model for finding topics     | Breaks down data into parts without negatives     | Uses math to find hidden themes                              |         |          |
| Approach              | Mixes topics and words to create documents | Separates data into positive factors only         | Breaks down words and documents                              |         |          |
| Key Characteristics   | Uses probabilities for topics and words    | Only adds, never subtracts parts of data          | Good at finding similar words                                |         |          |
| Output Interpretation | Topics are mixes of words                  | Shows parts making up the data                    | Shows main themes in data                                    |         |          |
| Handling of Data      | Best for word counts                       | Good for counts and other data types              | Works well with varied data types                            |         |          |
| Limitations           | Fixed number of topics, sensitive settings | Can't handle negative numbers, sensitive settings | Struggles with words having multiple meanings, influenced by word counts |         |          |

### Reference Papers

| Authors                | Link to Paper                                                | Publication Date |
| ---------------------- | ------------------------------------------------------------ | ---------------- |
| Roman Egger, Joanne Yu | [A Topic Modeling Comparison Between LDA, NMF, Top2Vec, and BERTopic to Demystify Twitter Posts](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9120935/) | 06-May-2022      |
| Maarten Grootendorst   | [BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) | 11-Mar-2022      |

- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. *Journal of Machine Learning Research, 3*, 993-1022.
- Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R. (1990). Indexing by Latent Semantic Analysis. *Journal of the American Society for Information Science, 41*(6), 391-407.
- Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature, 401*(6755), 788-791.
- Blei, D. M. (2012). Probabilistic topic models. *Communications of the ACM, 55*(4), 77-84.
- Salton, G., & McGill, M. J. (1983). *Introduction to Modern Information Retrieval.* McGraw-Hill.
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press.
- Ramos, J. (2003). Using TF-IDF to Determine Word Relevance in Document Queries. *Proceedings of the First Instructional Conference on Machine Learning*.
