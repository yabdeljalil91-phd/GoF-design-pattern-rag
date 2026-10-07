# RAG-Based Design Pattern Detection Using CodeBERT, FAISS, and PlantUML

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system to improve the ability of Large Language Models (LLMs) to **detect and reason about software design patterns**. The system combines semantic retrieval over a curated knowledge base of design patterns with generative reasoning from an LLM.

The knowledge base is composed of **PlantUML representations of design patterns**, embedded using **CodeBERT** and indexed with **FAISS** for efficient similarity search. At inference time, the most relevant design pattern descriptions are retrieved and injected into the LLM prompt to guide and constrain its reasoning.

## Objectives

* Enhance LLM accuracy in identifying software design patterns
* Ground LLM responses in formal, structured design pattern knowledge
* Enable semantic search over UML-based representations
* Reduce hallucinations by augmenting prompts with retrieved context

## Architecture Summary

The system follows a standard RAG pipeline:

1. **Knowledge Base Construction**
   Design patterns are encoded as PlantUML diagrams and stored as text documents.

2. **Embedding Generation**
   Each document is transformed into a dense vector using CodeBERT.

3. **Vector Indexing**
   Embeddings are stored in a FAISS vector index for fast similarity search.

4. **Query Processing**
   Input source code or UML-related queries are embedded using the same model.

5. **Retrieval**
   FAISS retrieves the top-k most semantically similar design pattern documents.

6. **Augmented Generation**
   Retrieved documents are injected into the LLM prompt to support design pattern detection.

## Key Technologies

### Retrieval-Augmented Generation (RAG)

RAG enhances generative models by incorporating external knowledge retrieved at runtime. Instead of relying solely on model parameters, the LLM is provided with contextually relevant documents, improving factual accuracy and domain specificity.

In this project, RAG is used to:

* Ground pattern detection in formal UML descriptions
* Provide explicit structural cues to the LLM
* Improve explainability of detected patterns

### FAISS (Facebook AI Similarity Search)

FAISS is used as the vector database for efficient similarity search over high-dimensional embeddings.

Key reasons for choosing FAISS:

* High-performance nearest neighbor search
* Scalable to large knowledge bases
* Seamless integration with Python-based ML pipelines

The FAISS index stores embeddings of all PlantUML design pattern documents and enables fast top-k retrieval during inference.

### Embedding Model: CodeBERT

The system uses:

```
HuggingFaceEmbeddings(model_name="microsoft/codebert-base")
```

CodeBERT is a transformer model pretrained on both source code and natural language. It is particularly well-suited for this use case because:

* It captures semantic relationships in code and structured text
* It performs well on software engineering tasks
* It generalizes across programming languages and modeling artifacts

Using CodeBERT allows PlantUML diagrams and code snippets to be embedded into the same semantic space.

### Knowledge Base: PlantUML Design Pattern Dataset

The knowledge base consists of **design patterns expressed in PlantUML format**.

Each document typically includes:

* Participants (classes, interfaces)
* Relationships (inheritance, composition, dependencies)
* Structural constraints characteristic of a design pattern

Benefits of using PlantUML:

* Human-readable and machine-processable
* Encodes structural information explicitly
* Aligns closely with UML-based design pattern definitions

## Data Preparation Pipeline

1. **Collect Design Patterns**
   Each design pattern is represented as a PlantUML diagram and stored as a text file.

2. **Preprocessing**

   * Remove non-essential comments (optional)
   * Normalize formatting if required
   * Split large diagrams into logical chunks if necessary

3. **Embedding Generation**
   Each PlantUML document is embedded using CodeBERT via Hugging Face embeddings.

4. **Index Construction**
   Embeddings are stored in a FAISS index along with document metadata (pattern name, category, etc.).

## Inference Workflow

1. **Input**
   The user provides source code, UML snippets, or a natural language query.

2. **Query Embedding**
   The input is embedded using the same CodeBERT model.

3. **Similarity Search**
   FAISS retrieves the top-k most similar PlantUML design pattern documents.

4. **Prompt Augmentation**
   Retrieved documents are appended to the LLM prompt as contextual knowledge.

5. **Design Pattern Detection**
   The LLM analyzes the augmented prompt and identifies candidate design patterns, often with justification.

## Benefits of the Approach

* Improves precision and recall in design pattern detection
* Reduces hallucinated or incorrect pattern classifications
* Enables explainable results through retrieved UML context
* Decouples knowledge updates from model retraining

## Limitations and Considerations

* Retrieval quality depends on embedding quality and dataset coverage
* Structural nuances may be lost if PlantUML diagrams are overly simplified
* Chunking strategy can significantly affect retrieval performance

## Possible Extensions

* Support for behavioral diagrams (sequence, activity diagrams)
* Hybrid retrieval (dense + keyword-based)
* Pattern classification confidence scoring
* Visualization of retrieved UML diagrams

## Conclusion

This project demonstrates how a RAG-based architecture, combined with CodeBERT embeddings, FAISS vector indexing, and a PlantUML-based knowledge base, can significantly enhance LLM performance in detecting software design patterns. By grounding generative reasoning in structured UML representations, the system achieves more accurate, explainable, and reliable results.
