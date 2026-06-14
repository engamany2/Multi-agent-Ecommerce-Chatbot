# 🛒 AI-Powered E-Commerce Multi-Agent Assistant

An advanced, integrated E-Commerce Assistant powered by a Multi-Agent Architecture and Retrieval-Augmented Generation (RAG). The system dynamically routes user inquiries, manages conversational memory, and delivers highly personalized, accurate responses based on the store's data, user preferences, and context.

---

## 📌 Architecture Diagram

The diagram below illustrates the system architecture and data flow, from the initial user input to the final response rendered on the user interface:

![Architecture Diagram](image.jpeg)

---

## 🗺️ System Workflow & Components

The core of this project relies on an intelligent pipeline where specialized agents collaborate to handle user interactions seamlessly:

### 1. Router Agent (`router.py`)
* **Function:** Analyzes the incoming user message and past chat history to make an optimal **routing decision**, determining which specialized agent is best suited to handle the request.

### 2. Specialized Agents
Based on the Router's decision, the request is directed to one of three specialized paths:
* **Product Search Agent (`product_search.py`):**
  * **Mechanism:** Utilizes **RAG over the Product Catalog** (`data/`, `products_expanded.csv`, and `chroma_db`) to handle product names, specifications, filters, comparisons, and alternative recommendations.
* **Order Tracking Agent:**
  * **Mechanism:** Performs direct database queries using **SQL** to fetch instant status updates regarding shipments and orders.
* **FAQ Agent (`FAQ.py`):**
  * **Mechanism:** Employs **RAG (Embeddings & Semantic Search)** to search through store policy documents (`store_policy.txt`) to answer precise questions about returns, shipping, warranty, payment, or cancellations.

### 3. Memory Agent (`memory_agent.py`)
* **Function:** Extracts important interaction patterns and user preferences, dynamically updating the relational core memory (`memory_store.db`) to personalize future interactions and maintain continuous context.

### 4. Response Agent (`response.py`)
* **Function:** Merges and structures the data from the activated agents and updated memory to synthesize a natural, cohesive, and perfectly tailored final response for the user.

### 5. UI Layer (`app.py`)
* The final formatted response is delivered beautifully via an interactive web interface built entirely using **Streamlit**.

---

## 📂 Project Structure

```text
├── .venv/                  # Python Virtual Environment
├── chroma_db/              # Vector database storing catalog embeddings
├── data/                   # Raw data folder
├── .env                    # Environment variables (API keys, secrets)
├── app.py                  # Main Streamlit application interface
├── FAQ.py                  # FAQ Agent handling policy RAG
├── import_out_txt          # Utility text/logs output
├── memory_agent.py         # Agent managing conversational memory logs
├── memory_store.db         # SQLite database storing long-term user context
├── product_search.py       # Product Catalog Search Agent
├── products_expanded.csv   # E-commerce product catalog dataset
├── requirements.txt        # Project dependencies and libraries
├── response.py             # Response Agent synthesizing final output
├── router.py               # Router Agent handling dynamic query routing
└── store_policy.txt        # Store policies data for the FAQ Agent
