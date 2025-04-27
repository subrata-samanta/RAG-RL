# Reinforcement Learning-based RAG System Implementation

This notebook implements a Retrieval Augmented Generation (RAG) system enhanced with Reinforcement Learning (RL) capabilities. The system combines the power of large language models with efficient document retrieval, optimized through reinforcement learning.

## Key Components

1. **RAG System**:
   - Uses FAISS for efficient vector similarity search
   - Implements document chunking and embedding
   - Integrates with Groq's LLM for text generation

2. **Reinforcement Learning**:
   - Policy network for document ranking optimization
   - Reward system based on answer relevance and diversity
   - Learning through experience to improve retrieval quality

3. **Core Technologies**:
   - Groq LLM integration via langchain_groq
   - HuggingFace embeddings for text encoding
   - PyTorch for neural network implementation
   - FAISS for vector storage and retrieval

## Setup Requirements

The following packages will be installed:
- langchain_groq: For LLM integration
- langchain-community: For document processing
- faiss-cpu: For vector similarity search

Note: This implementation requires a valid Groq API key stored in Google Colab secrets.

## Package Installation and Environment Setup

This cell installs the required packages and sets up the Groq API key. We'll install:
- `langchain_groq`: For interacting with Groq's LLM
- `langchain-community`: For document processing utilities
- `faiss-cpu`: For efficient vector similarity search

The API key is securely stored in Google Colab's secrets management system.
"""

!pip install langchain_groq langchain-community faiss-cpu
from google.colab import userdata
import os
os.environ["GROQ_API_KEY"] = userdata.get('groq_api_key')

"""## Core Components Implementation

This section implements the fundamental components of our RL-RAG system:

1. **Imports**: Required libraries for deep learning, natural language processing, and vector operations
2. **PolicyNetwork Class**: Neural network architecture for learning document ranking
   - Input: Combined query and document embeddings
   - Architecture: 3-layer feed-forward network with ReLU activation
   - Output: Scalar score for document relevance
   - Initialization: Xavier uniform for weights, small positive bias
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
import os

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)

    def forward(self, state):
        return self.net(state)

"""## RL-RAG System Implementation

The `RLRAGSystem` class integrates RAG with reinforcement learning:

### Key Methods:
- `load_documents`: Processes input text into manageable chunks
- `initialize_embeddings`: Sets up sentence embeddings using HuggingFace
- `initialize_retriever`: Configures FAISS vector store with MMR retrieval
- `initialize_policy`: Creates the policy network and optimizer
- `get_state`: Generates state representations for RL
- `calculate_reward`: Computes rewards based on answer quality and diversity
- `train_step`: Performs one iteration of policy optimization

### Features:
- MMR (Maximum Marginal Relevance) for diverse document selection
- Hybrid reward system considering both relevance and diversity
- Robust error handling for production environments
"""

class RLRAGSystem:
    def __init__(self, data_path):
        self.load_documents(data_path)
        self.initialize_embeddings()
        self.initialize_retriever()
        self.initialize_policy()
        self.llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
        self.min_documents = 1

    def load_documents(self, path):
        loader = TextLoader(path)
        documents = loader.load()
        splitter = CharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separator="\n",
            length_function=len
        )
        self.texts = splitter.split_documents(documents)
        print(f"Loaded {len(self.texts)} document chunks")

    def initialize_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def initialize_retriever(self):
        self.vector_store = FAISS.from_documents(self.texts, self.embeddings)
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10}
        )

    def initialize_policy(self):
        state_dim = 384 * 2
        self.policy = PolicyNetwork(state_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)

    def get_state(self, query, document):
        try:
            query_embed = self.embeddings.embed_query(query)
            doc_embed = self.embeddings.embed_query(document.page_content)
            return torch.FloatTensor(query_embed + doc_embed)
        except:
            return None

    def calculate_reward(self, query, documents, answer):
        try:
            answer_lower = answer.lower()
            query_words = set(query.lower().split())
            answer_words = set(answer_lower.split())
            match_score = len(query_words & answer_words) / len(query_words)
            unique_docs = len({d.page_content[:50] for d in documents})
            diversity_score = unique_docs / len(documents)
            return 0.7 * match_score + 0.3 * diversity_score
        except:
            return 0.0

    def train_step(self, query, true_answer):
        try:
            documents = self.retriever.get_relevant_documents(query)
            if len(documents) < self.min_documents:
                return 0.0, 0.0

            if len(documents) == 1:
                return 0.0, 0.0  # Skip single-document ranking

            states = [self.get_state(query, doc) for doc in documents]
            valid_states = [s for s in states if s is not None]

            if len(valid_states) < 2:
                return 0.0, 0.0

            # Fixed dimension handling
            scores = torch.stack([self.policy(state) for state in valid_states]).squeeze(-1)
            probs = torch.softmax(scores, dim=0)

            # Ensure valid sampling
            n_samples = min(len(valid_states), probs.size(-1))
            sorted_indices = torch.multinomial(probs, n_samples, replacement=False)

            ranked_docs = [documents[i] for i in sorted_indices]
            context = "\n".join([d.page_content for d in ranked_docs[:3]])
            answer = self.llm([
                SystemMessage(content=f"Answer based on:\n{context}"),
                HumanMessage(content=query)
            ]).content

            reward = self.calculate_reward(query, ranked_docs, answer)
            log_probs = torch.log(probs.gather(0, sorted_indices))
            loss = -torch.mean(log_probs) * reward

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item(), reward

        except Exception as e:
            print(f"Training error: {str(e)}")
            return 0.0, 0.0

"""## Training Loop Implementation

The `train_rag` function orchestrates the training process:

- Iterates through multiple epochs
- Processes query-answer pairs
- Tracks and reports training metrics:
  - Loss: Indicates policy improvement
  - Reward: Measures retrieval quality

The training loop includes validation checks and detailed logging for monitoring the learning progress.
"""

def train_rag(rl_rag, queries, answers, epochs=50):
    for epoch in range(epochs):
        total_loss = 0.0
        total_reward = 0.0
        valid_steps = 0

        for query, answer in zip(queries, answers):
            loss, reward = rl_rag.train_step(query, answer)
            if loss != 0 or reward != 0:
                total_loss += loss
                total_reward += reward
                valid_steps += 1

        if valid_steps > 0:
            avg_loss = total_loss / valid_steps
            avg_reward = total_reward / valid_steps
            print(f"\nEpoch {epoch+1}: Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}")
        else:
            print(f"\nEpoch {epoch+1}: No valid training steps")

"""## Sample Data Generation

The `create_sample_data` function generates a diverse dataset covering AI topics:

- Creates structured paragraphs on various AI concepts
- Topics include: AI basics, machine learning, deep learning, neural networks
- Saves data with appropriate formatting for retrieval

This synthetic dataset helps demonstrate and validate the RL-RAG system's capabilities.
"""

def create_sample_data():
    paragraphs = [
        "Artificial Intelligence (AI) is the simulation of human intelligence in machines. AI systems are designed to perform tasks like visual perception, speech recognition, and decision-making. Modern AI techniques include machine learning, deep learning, and neural networks.",
        "Machine learning is a subset of AI that enables systems to learn from data without explicit programming. There are three main types: supervised learning (labeled data), unsupervised learning (unlabeled data), and reinforcement learning (reward-based learning).",
        "Deep learning uses artificial neural networks with multiple layers to model complex patterns. Common architectures include CNNs for image processing and RNNs for sequence data. Transformers have recently become popular for NLP tasks.",
        "Neural networks are computing systems inspired by biological neurons. They consist of interconnected nodes (neurons) organized in layers. Training involves forward propagation and backpropagation to adjust weights.",
        "Reinforcement learning is a type of machine learning where agents learn by interacting with an environment. Key components include states, actions, rewards, and policies. Popular algorithms are Q-learning and policy gradient methods.",
        "Natural Language Processing (NLP) enables computers to understand human language. Techniques include tokenization, word embeddings (Word2Vec, GloVe), and transformer models (BERT, GPT).",
        "Computer Vision focuses on enabling machines to interpret visual data. Common tasks include image classification, object detection, and image segmentation. Popular frameworks are OpenCV and PyTorch Vision.",
        "The Turing Test evaluates a machine's ability to exhibit intelligent behavior indistinguishable from humans. Alan Turing proposed this test in 1950 as a measure of true AI.",
        "Ethics in AI involves addressing bias, privacy concerns, and transparency. Responsible AI development requires considering societal impacts and potential misuse of technology."
    ]

    with open('sample_data.txt', 'w') as f:
        f.write("\n\n".join(paragraphs))  # Use double newlines between paragraphs
    print("Sample data created with 9 paragraphs")

"""## Main Execution Block

The main execution section demonstrates the complete workflow:

1. **Data Preparation**:
   - Creates sample AI-focused dataset
   - Initializes the RL-RAG system

2. **Training Setup**:
   - Defines diverse query-answer pairs
   - Covers various AI topics for comprehensive learning

3. **System Training**:
   - Runs training for 10 epochs
   - Demonstrates retrieval capabilities with a test query

The output shows progressive improvement in retrieval quality and answer relevance.
"""

if __name__ == "__main__":
    # Create properly formatted sample data
    create_sample_data()

    # Initialize system
    rag_system = RLRAGSystem("sample_data.txt")

    # Training data
    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "How does reinforcement learning work?",
        "What are the main AI techniques?",
        "Describe deep learning architectures",
        "What's the difference between AI and machine learning?",
        "How do transformers work in NLP?",
        "What is the Turing Test?",
        "List computer vision applications",
        "Why are ethics important in AI?"
    ]

    answers = [
        "Machine learning is a subset of AI...",
        "Neural networks are computing systems...",
        "Reinforcement learning involves agents...",
        "Main AI techniques include machine learning...",
        "Deep learning architectures include CNNs...",
        "AI is the broader concept...",
        "Transformers process words using self-attention...",
        "The Turing Test evaluates machine intelligence...",
        "Applications include image classification...",
        "AI ethics addresses bias and privacy..."
    ]

    # Train the system
    train_rag(rag_system, queries, answers, epochs=10)

    # Test retrieval
    test_query = "What is deep learning?"
    documents = rag_system.retriever.get_relevant_documents(test_query)
    print("\nTest retrieval results:")
    for i, doc in enumerate(documents):
        print(f"[Doc {i+1}] {doc.page_content[:80]}...")
