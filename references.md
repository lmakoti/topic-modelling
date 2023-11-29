## References

| Ref   | Author                                         | Title                                                        | URL                                                          |
| ----- | ---------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| tm_01 | Maarten Grootendorst                           | BERTopic: Neural topic modelling with a class-based TF-IDF procedure | https://arxiv.org/abs/2203.05794                             |
| tm_02 | David M. Blei, Andrew Y. Ng, Michael I. Jordan | Latent Dirichlet Allocation                                  | https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf?ref=https://githubhelp.com |
| tm_03 | Luis Serrano                                   | Latent Dirichlet Allocation                                  | https://www.youtube.com/watch?v=T05t-SqKArY                  |
|       |                                                |                                                              |                                                              |
|       |                                                |                                                              |                                                              |
|       |                                                |                                                              |                                                              |
|       |                                                |                                                              |                                                              |
|       |                                                |                                                              |                                                              |
|       |                                                |                                                              |                                                              |



## Abstracts

**tm_01:** Topic models can be useful tools to discover latent topics in  collections of documents. Recent studies have shown the feasibility of  approach topic modelling as a clustering task. We present BERTopic, a  topic model that extends this process by extracting coherent topic  representation through the development of a class-based variation of  TF-IDF. More specifically, BERTopic generates document embedding with  pre-trained transformer-based language models, clusters these  embeddings, and finally, generates topic representations with the  class-based TF-IDF procedure. BERTopic generates coherent topics and  remains competitive across a variety of benchmarks involving classical  models and those that follow the more recent clustering approach of  topic modelling.    

**tm_02:** We describe latent Dirichlet allocation (LDA), a generative probabilistic model for collections of
discrete data such as text corpora. LDA is a three-level hierarchical Bayesian model, in which each
item of a collection is modelled as a finite mixture over an underlying set of topics. Each topic is, in
turn, modelled as an infinite mixture over an underlying set of topic probabilities. In the context of
text modelling, the topic probabilities provide an explicit representation of a document. We present
efficient approximate inference techniques based on variational methods and an EM algorithm for
empirical Bayes parameter estimation. We report results in document modelling, text classification,
and collaborative filtering, comparing to a mixture of unigrams model and the probabilistic LSI
model.

**tm_03:** Latent Dirichlet Allocation is a powerful machine learning technique used to sort documents by topic.
