## Deep Information Retrieval Hybrid
This project explores using embeddings from a large language model (LLM) to support standard document retrieval in the context of scientific literature. Specifically, it leverages the SPECTER2 transformer model to generate dense document and query embeddings, which are then compared with the traditional vector space retrieval (VSR) approach. A hybrid retrieval method is proposed, combining the dense retrieval and traditional VSR techniques to evaluate the potential improvements in retrieval performance.

# Project Overview
The goal of this project is to evaluate the effectiveness of deep learning-based embeddings (SPECTER2) for scientific document retrieval and compare them with a traditional vector space retrieval system. The project involves the following main steps:

1. Document and Query Embeddings:
Using the pre-trained SPECTER2 model, embeddings are generated for both scientific documents and queries. These embeddings are used to represent documents and queries as dense vectors.
Two variants of SPECTER2 embeddings are explored:
  Base Model: SPECTER2 model trained on a range of scientific document tasks.
  Adapter Model: SPECTER2 model with task-specific adapter modules attached for better 
  encoding of documents and queries.

2. Deep Retriever:
A custom class (DeepRetriever) is used to retrieve documents based on their dense vector embeddings. The retrieval is done using cosine similarity or Euclidean distance.

3. Hybrid Retrieval Approach:
A hybrid retrieval method is implemented to combine the traditional VSR system with the deep-learning-based retrieval. This method ranks retrieved documents using a weighted combination of the dense cosine similarity from the deep retriever and the sparse cosine similarity from the traditional VSR approach.

4. Evaluation:
The effectiveness of the models is evaluated using precision-recall (PR) and normalized discounted cumulative gain (NDCG) plots. Various configurations, such as the use of different similarity measures (cosine vs. Euclidean) and different values of the hybrid weighting parameter (λ), are tested.

The results from the experiments, including the precision-recall and NDCG plots for the different models, are summarized in the results file.



This code supplies "miniature" pedagogical Java implementations of
information retrieval, spidering, and other IR and text-processing
software.  It is being released for educational and research purposes only under
the GNU General Public License (see http://www.gnu.org/copyleft/gpl.html).

It was developed for an introductory course on "Intelligent Information
Retrieval and Web Search".  See:

http://www.cs.utexas.edu/users/mooney/ir-course/ 

for more information and introductory documentation (especially see the Project
assignment descriptions).

Copyleft: Raymond J. Mooney, 2001
