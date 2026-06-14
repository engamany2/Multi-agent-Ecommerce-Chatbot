import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Cached resources (initialized once) ---
_embedding_model = None
_vector_db = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    return _embedding_model

def _get_vector_db():
    global _vector_db
    if _vector_db is not None:
        return _vector_db
    
    PERSIST_DIR = "./chroma_db"
    embedding_model = _get_embedding_model()
    
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        _vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
    else:
        if not os.path.exists("store_policy.txt"):
            return None
            
        loader = TextLoader("store_policy.txt", encoding="utf-8")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        docs = text_splitter.split_documents(documents)

        _vector_db = Chroma.from_documents(
            docs,
            embedding_model,
            persist_directory=PERSIST_DIR
        )
    return _vector_db

def get_faq_answer(query: str) -> str:
    """
    دالة مطورة تسترجع الإجابات ذكياً:
    - تقرأ الـ Vector DB مباشرة لو مبنية (توفيراً للوقت).
    - تبنيها مرة واحدة فقط لو مش موجودة (تلقائياً بدلاً من 01_Data_Preparation).
    """
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    vector_db = _get_vector_db()
    if vector_db is None:
        return "Error: 'store_policy.txt' file not found in the path."

    # إعداد الـ Retriever بنفس الـ k=5
    retriever = vector_db.as_retriever(
        search_kwargs={"k": 5}
    )

    # تهيئة الـ LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=GROQ_API_KEY,
        temperature=0.3
    )

    # الـ Prompt والـ Chain
    prompt = ChatPromptTemplate.from_template("""
You are an FAQ assistant for 'High Tech' electronics store.
Answer ONLY using the provided context.
If the answer is not found in the context, say: "I couldn't find this information."
Reply in the SAME LANGUAGE as the question (Arabic -> Arabic, English -> English).

Context:
{context}

Question:
{input}

Answer:
""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # تنفيذ الاستدعاء وإعادة النتيجة الحقيقية
    try:
        response = retrieval_chain.invoke({"input": query})
        return response["answer"]
    except Exception as e:
        return f"Error inside FAQ Component: {e}"


# --- جزء الـ Testing التفاعلي المنفصل ---
if __name__ == "__main__":
    load_dotenv()
    print("\n👋 Hello! Ask me anything about our store policy (Interactive Test Mode)")
    print("(Type 'exit' to quit)\n")

    while True:
        question = input("Your question: ")
        
        if question.lower() == "exit":
            print("Goodbye!")
            break
            
        if question.strip() == "":
            print("Please write a question first! 😅\n")
            continue
            
        answer = get_faq_answer(question)
        print(f"\nAnswer:\n{answer}")
        print("\n" + "="*50 + "\n")