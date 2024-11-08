### Conclusion : https://arxiv.org/pdf/2004.12832

ColBERT: Eicient and Eective Passage Search via
Contextualized Late Interaction over BERT
Omar Khaab
Stanford University
okhaab@stanford.edu
Matei Zaharia
Stanford University
matei@cs.stanford.edu

In this paper, we introduced ColBERT, a novel ranking model that leverages contextualized late interaction over deep language models (in particular, BERT) for efficient retrieval. By independently encoding queries and documents into fine-grained representations that interact via cheap and pruning-friendly computations, ColBERT can utilize the expressiveness of deep LMs while significantly speeding up query processing. Additionally, this approach allows using ColBERT for end-to-end neural retrieval directly from a large document collection.

Our results demonstrate that ColBERT is more than 170× faster and requires 14,000× fewer FLOPs/query compared to existing BERT-based models, all while minimally impacting quality. Moreover, ColBERT outperforms every non-BERT baseline on the evaluation metrics used.

The key contributions of this work include:

1. **Proposing Late Interaction**: We propose late interaction as a paradigm for efficient and effective neural ranking.
2. **Presenting ColBERT**: We present ColBERT, a highly-effective model that employs novel BERT-based query and document encoders within the late interaction paradigm.
3. **Leveraging ColBERT**: We show how to leverage ColBERT both for re-ranking on top of a term-based retrieval model and for searching a full collection using vector similarity indexes.
4. **Evaluation**: We evaluate ColBERT on MS MARCO and TREC CAR, two recent passage search collections.

Overall, ColBERT represents a significant advancement in the field of Information Retrieval (IR), offering a scalable and efficient solution that combines the strengths of deep learning models with practical retrieval performance.

Here's a simplified version of the key algorithms described in the paper using pseudo-code written in C++-style syntax:

### ColBERT Algorithm

```cpp
// Define BERT-based encoders for queries and documents
class QueryEncoder {
public:
    vector<vector<float>> encodeQuery(string query) {
        // Tokenize query, prepend [Q] token, pass through BERT, linear layer, etc.
        // Return bag of fixed-size embeddings Eq
    }
};

class DocumentEncoder {
public:
    vector<vector<float>> encodeDocument(string document) {
        // Segment document, prepend [CLS], [D] tokens, pass through BERT, linear layer, etc.
        // Return bag of fixed-size embeddings Ed
    }
};

// Define the MaxSim operator for late interaction
float maxSim(vector<vector<float>>& Eq, vector<vector<float>>& Ed) {
    float maxSim = 0.0;
    for (auto& eq : Eq) {
        for (auto& ed : Ed) {
            // Compute cosine similarity and update maxSim if higher
            float sim = dotProduct(eq, ed);
            if (sim > maxSim) {
                maxSim = sim;
            }
        }
    }
    return maxSim;
}

// Define the scoring mechanism using MaxSim operators
float scoreQueryDocument(string query, string document) {
    QueryEncoder qe;
    DocumentEncoder de;
    vector<vector<float>> Eq = qe.encodeQuery(query);
    vector<vector<float>> Ed = de.encodeDocument(document);
    
    float relevanceScore = 0.0;
    for (auto& eq : Eq) {
        relevanceScore += maxSim({eq}, Ed); // Simplified to a single document for illustration
    }
    return relevanceScore;
}
```

### Offline Indexing Algorithm

```cpp
// Define the offline indexing procedure
void offlineIndex(vector<Document>& documents, string indexPath) {
    DocumentEncoder de;
    
    faiss::IndexIVFPQ index(faiss::Metric_L2, 128, 1000, 4);
    vector<float> documentEmbeddings;
    
    for (auto& doc : documents) {
        vector<vector<float>> Ed = de.encodeDocument(doc.content);
        
        // Concatenate embeddings and store
        documentEmbeddings.insert(documentEmbeddings.end(), Ed.begin(), Ed.end());
    }
    
    faiss::IndexIVFPQ* indexPtr = dynamic_cast<faiss::IndexIVFPQ*>(&index);
    if (indexPtr) {
        indexPtr->add(documentEmbeddings.size() / 128, documentEmbeddings.data());
    }
    
    // Save the index to file
    faiss::write_index(&index, indexPath.c_str());
}
```

### Top-k Re-ranking Algorithm

```cpp
// Define the top-k re-ranking algorithm
vector<Document> topKReRank(vector<Document>& documents, string query) {
    QueryEncoder qe;
    vector<vector<float>> Eq = qe.encodeQuery(query);
    
    faiss::IndexIVFPQ index(faiss::Metric_L2, 128, 1000, 4);
    // Load precomputed document embeddings from index file
    
    vector<Document> topKDocuments;
    for (auto& doc : documents) {
        vector<vector<float>> Ed = de.encodeDocument(doc.content);
        
        // Compute similarity using Faiss
        float similarity = faiss::search(&index, queryEmbeddings.data(), 1).first[0][0];
        
        // Store document with similarity score
        topKDocuments.push_back({doc.content, similarity});
    }
    
    // Sort documents by similarity and return top-k
    sort(topKDocuments.begin(), topKDocuments.end(), [](const Document& a, const Document& b) {
        return a.similarity > b.similarity;
    });
    
    return vector<Document>(topKDocuments.begin(), topKDocuments.begin() + k);
}
```

### End-to-end Top-k Retrieval Algorithm

```cpp
// Define the end-to-end top-k retrieval algorithm
vector<Document> endToEndTopKRetrieval(vector<Document>& documents, string query) {
    QueryEncoder qe;
    vector<vector<float>> Eq = qe.encodeQuery(query);
    
    faiss::IndexIVFPQ index(faiss::Metric_L2, 128, 1000, 4);
    // Load precomputed document embeddings from index file
    
    vector<Document> topKDocuments;
    for (auto& doc : documents) {
        vector<vector<float>> Ed = de.encodeDocument(doc.content);
        
        // Compute similarity using Faiss
        float similarity = faiss::search(&index, queryEmbeddings.data(), 1).first[0][0];
        
        // Store document with similarity score
        topKDocuments.push_back({doc.content, similarity});
    }
    
    // Sort documents by similarity and return top-k
    sort(topKDocuments.begin(), topKDocuments.end(), [](const Document& a, const Document& b) {
        return a.similarity > b.similarity;
    });
    
    return vector<Document>(topKDocuments.begin(), topKDocuments.begin() + k);
}
```

This pseudo-code provides a high-level overview of the key algorithms described in the paper. Note that this is a simplified version and does not include all the details and optimizations mentioned in the original paper.
