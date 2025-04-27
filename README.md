# RAG-RL: Reinforcement Learning Enhanced Retrieval-Augmented Generation

This project implements a Reinforcement Learning (RL) enhanced Retrieval-Augmented Generation (RAG) system that optimizes document retrieval for question answering tasks.

## Overview

The system combines:
- RAG (Retrieval-Augmented Generation) for context-aware responses
- Reinforcement Learning for optimizing document retrieval
- BERT-based reward modeling for semantic similarity
- Policy gradient methods for learning optimal retrieval strategies

## Features

- Custom PolicyNetwork for document re-ranking
- FAISS vector store for efficient document retrieval
- HuggingFace embeddings (all-MiniLM-L6-v2)
- Groq LLM integration (llama-3.3-70b-versatile)
- Reward calculation based on semantic similarity and document diversity
- Policy gradient training with experience replay

## Requirements

```
torch
numpy
langchain
faiss-cpu
transformers
langchain_groq
```

## Installation

1. Clone the repository
2. Install the required packages:
```bash
pip install torch numpy langchain faiss-cpu transformers langchain_groq
```
3. Set up your Groq API key as an environment variable:
```bash
export GROQ_API_KEY='your-api-key'
```

## Usage

1. Initialize the RAG-RL system:
```python
rag_system = RLRAGSystem("your_data.txt")
```

2. Prepare your training data (queries and corresponding answers)

3. Train the system:
```python
train_rag(rag_system, queries, answers, epochs=50)
```

4. Use the trained system for inference:
```python
test_query = "Your question here?"
documents = rag_system.retriever.get_relevant_documents(test_query)
```

## Architecture

- **PolicyNetwork**: Neural network that learns to score and rank documents
- **RLRAGSystem**: Main class that integrates:
  - Document loading and chunking
  - Embedding generation
  - Document retrieval
  - Policy-based learning
  - Reward calculation
  - LLM integration

## Training Process

The system learns through:
1. Initial document retrieval using FAISS
2. Document scoring using the policy network
3. Sampling document order based on scores
4. Generating answers using the LLM
5. Computing rewards based on answer quality and diversity
6. Updating the policy network using policy gradients

## License

MIT License

## Contributing

Feel free to open issues and pull requests for improvements!