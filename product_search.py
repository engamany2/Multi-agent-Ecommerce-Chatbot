import os
import re
import warnings
import logging

warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.getLogger("transformers").setLevel(logging.ERROR)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class ProductSearch:

    def __init__(self, file_path="products_expanded.csv",
    groq_api_key=os.getenv("GROQ_API_KEY")):
        self.file_path = file_path
        self.groq_api_key = groq_api_key
        # موديل خفيف للترجمة فقط
        self.llm_light = ChatGroq(
            api_key=self.groq_api_key,
            model="llama-3.1-8b-instant",
            temperature=0
        )
        self.retriever = self._build_retriever()

    def _build_retriever(self):
        loader = CSVLoader(file_path=self.file_path, encoding='utf-16', csv_args={'delimiter': '\t'})
        documents = loader.load()
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        vector_db = FAISS.from_documents(documents, embeddings)
        return vector_db.as_retriever(search_kwargs={"k": 10})

    def _translate_query(self, query: str) -> str:
        """Translate Arabic queries to English for better FAISS retrieval."""
        has_arabic = bool(re.search(r'[\u0600-\u06FF]', query))
        if not has_arabic:
            return query
        
        translate_prompt = ChatPromptTemplate.from_template(
            "You are translating an Arabic e-commerce search query to English.\n"
            "Common Arabic electronics terms:\n"
            "- ساعات / ساعة = watches / smartwatch (NOT hours)\n"
            "- سماعات / سماعة = headphones / earbuds\n"
            "- تليفون / موبايل = phone / smartphone\n"
            "- شاشة = screen / monitor\n"
            "- لابتوب = laptop\n\n"
            "Output ONLY the English product search keywords, nothing else.\n\n"
            "Arabic query: {query}\n"
            "English keywords:"
        )
        try:
            chain = translate_prompt | self.llm_light | StrOutputParser()
            translated = chain.invoke({"query": query})
            return translated.strip()
        except Exception:
            # Fallback: basic keyword extraction
            return query

    def _extract_price_range(self, query: str):
        """Extract price range from query like '2000 :7000' or 'بين 2000 و 7000'."""
        import re
        # Match patterns like "2000:7000", "2000 - 7000", "بين 2000 و 7000", "من 2000 ل 7000"
        patterns = [
            r'(\d+)\s*[:：\-–]\s*(\d+)',  # 2000:7000 or 2000-7000
            r'بين\s*(\d+)\s*و\s*(\d+)',  # بين 2000 و 7000
            r'من\s*(\d+)\s*ل\s*(\d+)',   # من 2000 ل 7000
            r'(\d+)\s*to\s*(\d+)',        # 2000 to 7000
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                low, high = int(match.group(1)), int(match.group(2))
                return (min(low, high), max(low, high))
        return None

    def get_product(self, query: str) -> str:
        """Search for products and return RAW catalog data — NO LLM generation, NO hallucination."""
        import time
        for attempt in range(3):
            try:
                # 1. Translate Arabic to English for better retrieval
                search_query = self._translate_query(query)
                
                # 2. Retrieve more documents from FAISS for better filtering
                docs = self.retriever.invoke(search_query)
                
                if not docs:
                    return "⚠️ لا توجد منتجات مطابقة في الكتالوج."
                
                # 3. Price range filtering — if user specified a budget
                price_range = self._extract_price_range(query)
                if price_range:
                    min_price, max_price = price_range
                    filtered_docs = []
                    for doc in docs:
                        # Extract price from doc content
                        price_match = re.search(r'Price_EGP:\s*(\d+)', doc.page_content)
                        if price_match:
                            price = int(price_match.group(1))
                            if min_price <= price <= max_price:
                                filtered_docs.append(doc)
                    
                    if not filtered_docs:
                        return f"⚠️ NO PRODUCTS FOUND in the {min_price}-{max_price} EGP range. Our catalog does not have products at this price point for this category."
                    docs = filtered_docs
                
                # 4. Brand filtering — if user mentioned a specific brand
                brand_filter = self._detect_brand(query)
                if brand_filter:
                    filtered_docs = []
                    for doc in docs:
                        content_lower = doc.page_content.lower()
                        # Check both EN and AR product names for the brand
                        if brand_filter.lower() in content_lower:
                            filtered_docs.append(doc)
                    if filtered_docs:  # Only filter if we found matches
                        docs = filtered_docs
                
                # 5. Return RAW document data with STRICT warning
                formatted = []
                for doc in docs:
                    formatted.append(doc.page_content)
                
                header = ("=== CATALOG RESULTS ===\n"
                         "⚠️ STRICT: These are the ONLY products that exist. "
                         "DO NOT add, invent, or mention ANY product not listed below.\n\n")
                
                return header + "\n\n---\n\n".join(formatted)
                
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    if attempt < 2:
                        time.sleep(15)
                        continue
                    return "عذراً، النظام مشغول حالياً. جرب تاني بعد شوية."
                return f"Error: {e}"

    # Brand keyword mapping for filtering
    BRAND_MAP = {
        'ابل': 'Apple', 'أبل': 'Apple', 'آبل': 'Apple', 'apple': 'Apple', 'ايفون': 'Apple', 'آيفون': 'Apple', 'airpods': 'Apple',
        'سامسونج': 'Samsung', 'سامسونغ': 'Samsung', 'samsung': 'Samsung', 'جالاكسي': 'Samsung', 'galaxy': 'Samsung',
        'سوني': 'Sony', 'sony': 'Sony',
        'هواوي': 'Huawei', 'huawei': 'Huawei',
        'شاومي': 'Xiaomi', 'xiaomi': 'Xiaomi', 'ريدمي': 'Xiaomi',
        'ون بلس': 'OnePlus', 'oneplus': 'OnePlus',
        'أوبو': 'Oppo', 'oppo': 'Oppo',
        'jbl': 'JBL', 'جي بي ال': 'JBL',
        'أنكر': 'Anker', 'anker': 'Anker',
        'لوجيتك': 'Logitech', 'logitech': 'Logitech',
        'أمازفيت': 'Amazfit', 'amazfit': 'Amazfit',
        'جارمن': 'Garmin', 'garmin': 'Garmin',
        'فيتبيت': 'Fitbit', 'fitbit': 'Fitbit',
        'بيتس': 'Beats', 'beats': 'Beats',
    }

    def _detect_brand(self, query: str) -> str:
        """Detect a specific brand mentioned in the query."""
        query_lower = query.lower()
        for keyword, brand in self.BRAND_MAP.items():
            if keyword in query_lower:
                return brand
        return ""
