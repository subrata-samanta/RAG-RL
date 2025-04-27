import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from transformers import AutoTokenizer, AutoModel
from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# load_dotenv()

from google.colab import userdata
import os
os.environ["GROQ_API_KEY"] = userdata.get('groq_api_key')

class PolicyNetwork(nn.Module):
    """Custom policy network for document re-ranking"""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # Output score for each document
        )

    def forward(self, state):
        return self.net(state)

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)


class RLRAGSystem:
    def __init__(self, data_path):
        # Initialize RAG components
        self.load_documents(data_path)
        self.initialize_embeddings()
        self.initialize_retriever()
        self.initialize_policy()

        # Initialize LLM
        self.llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.reward_model = AutoModel.from_pretrained("bert-base-uncased")

    def load_documents(self, path):
        loader = TextLoader(path)
        documents = loader.load()
        splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        self.texts = splitter.split_documents(documents)

    def initialize_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = FAISS.from_documents(self.texts, self.embeddings)

    def initialize_retriever(self):
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

    def initialize_policy(self):
        state_dim = 384 * 2  # Query + document embedding
        self.policy = PolicyNetwork(state_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)

    def get_state(self, query, document):
        query_embed = self.embeddings.embed_query(query)
        doc_embed = self.embeddings.embed_query(document.page_content)
        return torch.FloatTensor(query_embed + doc_embed)

    def calculate_reward(self, query, documents, answer):
        # Semantic similarity reward
        with torch.no_grad():
            inputs = self.tokenizer([query], [answer], return_tensors='pt', padding=True, truncation=True)
            outputs = self.reward_model(**inputs)
            reward = torch.mean(outputs.last_hidden_state).item()

        # Diversity penalty
        doc_embeds = [self.embeddings.embed_query(d.page_content) for d in documents]

        # Check if doc_embeds has more than one element to avoid calculating correlation for a single document
        if len(doc_embeds) > 1:
            similarity_matrix = np.corrcoef(doc_embeds)
            diversity = 1 - np.mean(similarity_matrix[np.triu_indices(len(documents), 1)]) # Changed to get upper triangular elements excluding diagonal
        else:
            diversity = 0 # Set diversity to 0 if only one document is retrieved

        return reward * 0.8 + diversity * 0.2

    def train_step(self, query, true_answer):
        # Retrieve initial documents
        documents = self.retriever.get_relevant_documents(query)

        # Generate document scores
        states = [self.get_state(query, doc) for doc in documents]
        scores = [self.policy(state) for state in states]
        probs = torch.softmax(torch.stack(scores), dim=0)

        # Sample document order
        sorted_indices = torch.multinomial(probs, len(documents), replacement=False)
        ranked_docs = [documents[i] for i in sorted_indices]

        # Generate answer
        context = "\n".join([d.page_content for d in ranked_docs[:3]])
        messages = [
            SystemMessage(content=f"Answer based on:\n{context}"),
            HumanMessage(content=query)
        ]

        answer_message = self.llm(messages) # This line returns an AIMessage object
        answer = answer_message.content # Extract the content string from the AIMessage

        # Calculate reward
        reward = self.calculate_reward(query, ranked_docs, answer)

        # Policy gradient update
        loss = -torch.mean(torch.log(probs[sorted_indices])) * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), reward

# Training loop
def train_rag(rl_rag, queries, answers, epochs=50):
    for epoch in range(epochs):
        total_loss = 0
        total_reward = 0

        for query, answer in zip(queries, answers):
            loss, reward = rl_rag.train_step(query, answer)
            total_loss += loss
            total_reward += reward

        print(f"Epoch {epoch+1}: Loss: {total_loss/len(queries):.4f}, Reward: {total_reward/len(queries):.4f}")

with open('sample_data.txt', 'w') as f:
  f.write("Artificial Intelligence (AI) is the simulation of human intelligence in machines. AI systems are designed to perform tasks like visual perception, speech recognition, and decision-making. Modern AI techniques include machine learning, deep learning, and neural networks.\nMachine learning is a subset of AI that enables systems to learn from data without explicit programming. There are three main types: supervised learning (labeled data), unsupervised learning (unlabeled data), and reinforcement learning (reward-based learning).\nDeep learning uses artificial neural networks with multiple layers to model complex patterns. Common architectures include CNNs for image processing and RNNs for sequence data. Transformers have recently become popular for NLP tasks.\nNeural networks are computing systems inspired by biological neurons. They consist of interconnected nodes (neurons) organized in layers. Training involves forward propagation and backpropagation to adjust weights.\nReinforcement learning is a type of machine learning where agents learn by interacting with an environment. Key components include states, actions, rewards, and policies. Popular algorithms are Q-learning and policy gradient methods.\nNatural Language Processing (NLP) enables computers to understand human language. Techniques include tokenization, word embeddings (Word2Vec, GloVe), and transformer models (BERT, GPT).\nComputer Vision focuses on enabling machines to interpret visual data. Common tasks include image classification, object detection, and image segmentation. Popular frameworks are OpenCV and PyTorch Vision.\nThe Turing Test evaluates a machine's ability to exhibit intelligent behavior indistinguishable from humans. Alan Turing proposed this test in 1950 as a measure of true AI.\nEthics in AI involves addressing bias, privacy concerns, and transparency. Responsible AI development requires considering societal impacts and potential misuse of technology")

# !pip install faiss-cpu

# Usage example
if __name__ == "__main__":
    # Initialize system
    rag_system = RLRAGSystem("sample_data.txt")

    # Generate synthetic training data
     # Generate synthetic training data
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
        "Machine learning is a subset of AI that enables systems to learn from data without explicit programming, using methods like supervised, unsupervised, and reinforcement learning.",
        "Neural networks are computing systems inspired by biological neurons, consisting of interconnected nodes organized in layers that process information through forward propagation and backpropagation.",
        "Reinforcement learning involves agents learning through environment interaction using states, actions, and rewards, with algorithms like Q-learning and policy gradient methods.",
        "Main AI techniques include machine learning (supervised/unsupervised/reinforcement), deep learning (CNNs, RNNs, transformers), NLP methods, and computer vision algorithms.",
        "Deep learning architectures include CNNs for image processing, RNNs for sequence data, and transformer models which are particularly effective for NLP tasks.",
        "AI is the broader concept of machines performing intelligent tasks, while machine learning is a specific approach where systems learn patterns from data without explicit programming.",
        "Transformers in NLP process words in relation to all other words in a sentence using self-attention mechanisms, enabling better understanding of context compared to older RNN-based approaches.",
        "The Turing Test evaluates a machine's ability to exhibit behavior indistinguishable from humans, proposed by Alan Turing in 1950 as a measure of true artificial intelligence.",
        "Computer vision applications include image classification, object detection, facial recognition, medical image analysis, and autonomous vehicle navigation.",
        "AI ethics addresses critical concerns like algorithmic bias, data privacy, transparency in decision-making, and preventing misuse of autonomous systems."
    ]

    # Train the system
    train_rag(rag_system, queries, answers)

    # Test the optimized retriever
    test_query = "What is deep learning?"
    documents = rag_system.retriever.get_relevant_documents(test_query)
    states = [rag_system.get_state(test_query, doc) for doc in documents]
    scores = [rag_system.policy(state).item() for state in states]
    ranked_docs = sorted(zip(documents, scores), key=lambda x: -x[1])

    print("\nOptimized Retrieval Results:")
    for doc, score in ranked_docs:
        print(f"[Score: {score:.2f}] {doc.page_content[:80]}...")

