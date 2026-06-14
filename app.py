import streamlit as st
import os
import re
import json
from dotenv import load_dotenv
from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# استدعاء موديولات الفريق الحقيقية (الـ Core Modules)
from router import router_with_memory
from product_search import ProductSearch
from FAQ import get_faq_answer
from memory_agent import MemoryAgent  # استدعاء موديول الذاكرة

# --- 1. الإعدادات الأولية ---
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

# llm_fast = الموديل الكبير للردود الرئيسية (Writer فقط)
# llm_light = موديل صغير وسريع للمهام الخلفية (Critic, وغيرها) عشان نوفر الـ Rate Limit
llm_fast = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_light = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

@st.cache_resource
def init_engines():
    return ProductSearch()

product_search_engine = init_engines()

# --- 2. بناء الـ Graph وجلب البيانات الحقيقية ---
class AgentState(TypedDict):
    user_query: str
    language: str
    route_destination: str
    retrieved_info: str
    user_memories: str     # جلب الذاكرة التاريخية
    response_draft: str
    critic_score: int
    critic_feedback: str
    iterations: int

def router_node(state: AgentState):
    session_id = st.session_state.session_id
    config = {"configurable": {"session_id": session_id}}
    
    user_query = state["user_query"]
    
    # أ] توسيع الاستعلامات الغامضة فقط (مش كل الاستعلامات القصيرة)
    # "اشطا قولى" أو "عاوز اعرف اكتر" → غامضة، محتاجة سياق
    # "متاح التوصيل للمنوفيه" → واضحة، مش محتاجة سياق
    expanded_query = user_query
    recent_messages = st.session_state.get("messages", [])
    
    # كلمات بتدل إن الرسالة متابعة غامضة ومحتاجة سياق
    ambiguous_words = ["عليهم", "عليها", "عليه", "ده", "دي", "دول", "اكتر", "اكثر",
                       "قولى", "قولي", "كمان", "ايوا", "اشطا", "تمام", "طيب", "اه",
                       "more", "tell me", "yes", "ok", "sure", "them", "it", "this"]
    
    is_ambiguous = (len(user_query.strip()) < 25 and 
                    any(w in user_query.lower() for w in ambiguous_words) and
                    len(recent_messages) >= 2)
    
    if is_ambiguous:
        last_msgs = recent_messages[-2:]
        context_lines = []
        for msg in last_msgs:
            role = "User" if msg["role"] == "user" else "Agent"
            content = msg["content"][:150]
            context_lines.append(f"{role}: {content}")
        conversation_context = "\n".join(context_lines)
        expanded_query = f"Context: {conversation_context}\nCurrent question: {user_query}"
    
    # ب] تشغيل التوجيه — دايماً بالاستعلام الأصلي
    router_response = router_with_memory.invoke({"input": user_query}, config=config)
    destination = router_response.destination
    
    # ج] استدعاء الـ Memory Agent
    mem_agent = MemoryAgent(user_id=session_id)
    history_hits = mem_agent.retrieve(user_query, top_k=2)
    profile = mem_agent.get_user_profile()
    
    profile_facts = f"Budget: {profile.budget}, Preferred Brand: {profile.favorite_brand}, Preferred Category: {profile.preferred_category}"
    past_memories = "\n".join([f"- {h}" for h in history_hits])
    user_memories_context = f"User Historical Profile Facts:\n{profile_facts}\n\nPast Relevant Conversations Summaries:\n{past_memories}"
    mem_agent.close()
    
    context = ""
    # د] توجيه البيانات الحقيقية
    # Check for combined queries (product + FAQ in same message)
    faq_words = ['توصيل', 'شحن', 'ارجع', 'استرجاع', 'ارجاع', 'ضمان', 'تقسيط', 'اتصل', 'سياسة']
    product_words = ['سماعات', 'سماعة', 'تليفون', 'موبايل', 'ساعات', 'ساعة', 'ايفون', 'سامسونج', 'سعر']
    
    has_faq = any(w in user_query for w in faq_words)
    has_product = any(w in user_query for w in product_words)
    
    if has_faq and has_product:
        # Combined query — fetch from BOTH sources
        product_context = product_search_engine.get_product(expanded_query if is_ambiguous else user_query)
        faq_context = get_faq_answer(user_query)
        context = f"[PRODUCT DATA]\n{product_context}\n\n[STORE POLICY]\n{faq_context}"
        destination = "combined"
    elif destination == "product_search":
        context = product_search_engine.get_product(expanded_query if is_ambiguous else user_query)
    elif destination == "faq_agent":
        context = get_faq_answer(user_query)
    elif destination == "order_tracking":
        context = "نظام تتبع الشحنات قيد الربط. اطلب من العميل رقم الشحنة."
    elif destination == "chitchat":
        context = "رحب بالعميل بلطف بالعامية المصرية. لا تذكر أي منتجات. فقط قول أهلاً واسأله محتاج مساعدة في إيه."
    else:
        destination = "out_of_scope"
        context = "[OUT_OF_SCOPE] السؤال ده خارج نطاق متجر High Tech للإلكترونيات. رد بجملة واحدة فقط تعتذر فيها بلطف وتوجه العميل يسأل عن منتجات أو خدمات المتجر. ممنوع تكرار أي جملة."

    return {
        "route_destination": destination, 
        "retrieved_info": context, 
        "user_memories": user_memories_context
    }

def writer_node(state: AgentState):
    is_arabic = state.get("language") == "Arabic"
    
    if is_arabic:
        lang_instruction = """أنت لازم ترد بالعامية المصرية الطبيعية.
        أمثلة على الأسلوب المطلوب:
        - "أهلاً بيك! عندنا سماعات كتير حلوة، هقولك عليهم:"
        - "السماعة دي سعرها 4200 جنيه وفيها خاصية عزل الضوضاء."
        - "للأسف المنتج ده مش موجود عندنا دلوقتي، بس عندنا بدايل تانية ممتازة."
        - "أيوه بنوصل لكل المحافظات! القاهرة والجيزة خلال 24-48 ساعة."
        
        ممنوع تماماً:
        - الفصحى الرسمية (مثل: "أنا آسف، لم يتم العثور على", "سوف أتحقق")
        - خلط كلمات إنجليزية في الجمل العربية (ماعدا أسماء الماركات زي Samsung, Sony, Apple)
        - الـ Franco Arab (مثل: "3ayez", "sma3at")"""
    else:
        lang_instruction = "You must respond in clear, professional English."
    
    # إضافة سياق المحادثة الأخيرة للـ Writer عشان يفهم المتابعات
    recent_messages = st.session_state.get("messages", [])
    chat_context = ""
    if recent_messages:
        last_msgs = recent_messages[-4:]  # آخر 4 رسائل
        chat_lines = []
        for msg in last_msgs:
            role = "User" if msg["role"] == "user" else "Agent"
            chat_lines.append(f"{role}: {msg['content'][:200]}")
        chat_context = "\n".join(chat_lines)
    
    system_msg = f"""You are a friendly customer service agent for 'High Tech' electronics store.

{lang_instruction}

ABSOLUTE RULES — BREAKING ANY RULE IS A CRITICAL FAILURE:

1. PRODUCTS: Retrieved Info contains RAW CATALOG DATA with fields like Product_Name_AR, Price_EGP, Description_AR, Category, etc.
   - Extract product info ONLY from these catalog entries.
   - For each product that matches the user's question, format as: Product Name — Price جنيه — Description
   - Use Product_Name_AR for Arabic responses, Product_Name_EN for English responses.
   - Use the EXACT Price_EGP from the catalog. NEVER change or guess prices.
   - If the Category doesn't match what the user asked for (e.g., user asked for ساعات/watches but catalog shows headphone), SKIP that product.
   - NEVER invent products. If no matching products found in Retrieved Info, say the product is not available.

2. CHITCHAT/GREETING: If Retrieved Info says "رحب بالعميل" or similar:
   - ONLY greet the customer warmly and ask how you can help.
   - Do NOT mention ANY products or categories.
   - Example: "أهلاً بيك في High Tech! تقدر تسألني عن أي حاجة — منتجات، أسعار، توصيل، أو أي استفسار تاني."

3. FAQ/POLICY: If Retrieved Info contains policy information:
   - Answer directly from the policy data provided.
   - Do NOT make up ordering steps or processes not mentioned in the data.

4. FOLLOW-UP QUERIES: Look at Chat History for context when the user's message is short.

5. KEEP IT SHORT: 2-4 sentences for greetings, list format for products. Be direct.
6. NO EXCESSIVE APOLOGIES: Don't say "آسف" repeatedly.
7. OUT-OF-SCOPE: If Retrieved Info contains "[OUT_OF_SCOPE]", respond with EXACTLY ONE short sentence (max 2 sentences). Politely say this is outside the store's scope and ask the user to ask about electronics instead. Do NOT repeat any phrase. Do NOT add extra paragraphs or parenthetical notes. Just ONE clean response."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("user", "Chat History:\n{history}\n\nUser Query: {query}\n\nRetrieved Info: {info}\n\nUser Memories: {memories}\n\nCritic Feedback: {feedback}")
    ])
    chain = prompt | llm_fast
    # Retry on rate limit
    import time
    for attempt in range(3):
        try:
            response = chain.invoke({
                "history": chat_context,
                "query": state["user_query"],
                "info": state["retrieved_info"],
                "memories": state["user_memories"],
                "feedback": state.get("critic_feedback", "None")
            })
            return {"response_draft": response.content, "iterations": state.get("iterations", 0) + 1}
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                if attempt < 2:
                    time.sleep(15)
                    continue
                # Fallback to lighter model on last attempt
                chain_fallback = prompt | llm_light
                response = chain_fallback.invoke({
                    "history": chat_context,
                    "query": state["user_query"],
                    "info": state["retrieved_info"],
                    "memories": state["user_memories"],
                    "feedback": state.get("critic_feedback", "None")
                })
                return {"response_draft": response.content, "iterations": state.get("iterations", 0) + 1}
            raise e

def critic_node(state: AgentState):
    is_arabic = state.get("language") == "Arabic"
    
    if is_arabic:
        criteria = """Check these criteria:
        1. Is the response in natural Egyptian dialect (عامية مصرية)? If it uses formal Arabic like "لم يتم" or "سوف أتحقق", score LOW (1-4).
        2. Does it answer the user's question with real data? If it just apologizes without providing info, score LOW.
        3. Are products listed with names and prices? If products exist in context but aren't shown, score LOW.
        Respond with a NUMBER 1-10 ONLY."""
    else:
        criteria = "Check if the response is helpful, in proper English, and answers the question. Respond with a NUMBER 1-10 ONLY."
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", criteria),
        ("user", "User asked: {query}\n\nAgent responded: {draft}")
    ])
    chain = prompt | llm_light  # الكريتيك بيستخدم الموديل الخفيف عشان نوفر tokens
    try:
        result = chain.invoke({"query": state["user_query"], "draft": state["response_draft"]})
        score = int(''.join(filter(str.isdigit, result.content))[:2])
        if score > 10: score = int(str(score)[0])
    except:
        score = 8
    return {"critic_score": score, "critic_feedback": result.content}

def should_continue(state: AgentState):
    # تخطي الـ Critic loop للردود اللي مش محتاجة إعادة كتابة (ترحيب، خارج النطاق)
    route = state.get("route_destination", "")
    if route in ("chitchat", "out_of_scope"):
        return "end"
    if state["critic_score"] < 7 and state["iterations"] < 3:
        return "re_write"
    return "end"

# بناء الـ Graph
workflow = StateGraph(AgentState)
workflow.add_node("router_agent", router_node)
workflow.add_node("writer", writer_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("router_agent")
workflow.add_edge("router_agent", "writer")
workflow.add_edge("writer", "critic")
workflow.add_conditional_edges("critic", should_continue, {"re_write": "writer", "end": END})
app_graph = workflow.compile()

# --- 3. واجهة الـ Streamlit (UI) ---
st.set_page_config(page_title="High Tech AI Center", page_icon="💎", layout="wide")

st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;800&family=Outfit:wght@300;500;700&display=swap');
    
    * {
        font-family: 'Outfit', 'Cairo', sans-serif !important;
    }
    
    /* Main Background & Chat */
    .stApp {
        background-color: #0b0f19;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e2e8f0 !important;
        font-weight: 800 !important;
    }
    
    /* User Chat Bubble */
    [data-testid="chatAvatarIcon-user"] {
        background-color: #3b82f6 !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
        direction: rtl;
        text-align: right;
    }
    
    /* Assistant Chat Bubble */
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #10b981 !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: linear-gradient(135deg, #0f172a 0%, #020617 100%);
        border: 1px solid #1e293b;
        border-left: 4px solid #3b82f6;
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
        direction: rtl;
        text-align: right;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid #1e293b;
    }
    
    /* Chat Input */
    .stChatInputContainer {
        border-radius: 20px !important;
        border: 1px solid #3b82f6 !important;
        background-color: #0f172a !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #60a5fa;'>💎 High Tech - Premium Support</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.1rem;'>المساعد الذكي الفاخر لاختيار أفضل الأجهزة</p>", unsafe_allow_html=True)

# تثبيت سيسشن ثابت لكل مستخدم لاختبار حفظ الذاكرة واسترجاعها عبر الحوارات المختلفة
if "session_id" not in st.session_state:
    st.session_state.session_id = "customer_profile_session_1"

if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض المحادثة الجارية في الـ Sidebar لمراقبة الـ Profile الحقيقي المخزن
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=60)
    st.markdown("### 👤 User Memory Profile")
    st.caption("Powered by Gemini Long-Term Memory")
    st.divider()
    
    mem_agent_side = MemoryAgent(user_id=st.session_state.session_id)
    profile_side = mem_agent_side.get_user_profile()
    
    # Render Profile beautifully instead of raw JSON
    st.markdown("#### 🎯 Core Preferences")
    
    brand = profile_side.favorite_brand if profile_side.favorite_brand else "Not specified yet"
    budget = profile_side.budget if profile_side.budget else "Not specified yet"
    category = profile_side.preferred_category if profile_side.preferred_category else "Not specified yet"
    
    st.info(f"**Favorite Brand:** {brand}")
    st.success(f"**Budget:** {budget}")
    st.warning(f"**Preferred Category:** {category}")
    
    if profile_side.important_preferences:
        st.markdown("#### ⭐ Important Facts")
        for pref in profile_side.important_preferences:
            st.markdown(f"- {pref}")
            
    mem_agent_side.close()
    
    st.divider()
    st.markdown("*(The AI automatically updates this profile as you chat!)*")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("بماذا يمكنني مساعدتك اليوم في High Tech؟"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    is_arabic = bool(re.search(r'[\u0600-\u06FF]', prompt))
    user_lang = "Arabic" if is_arabic else "English"

    with st.chat_message("assistant"):
        with st.status("🧠 جاري تشغيل الـ Multi-Agents وتحليل الذاكرة الطويلة...", expanded=True) as status:
            
            inputs = {"user_query": prompt, "language": user_lang, "iterations": 0}
            final_response = ""
            
            for output in app_graph.stream(inputs):
                for key, value in output.items():
                    if "route_destination" in value:
                        st.write(f"🔀 **Router Decision:** Translated intent to `{value['route_destination'].upper()}`")
                    elif key == "writer":
                        st.write("⚙️ **Writer Agent** is personalizing response...")
                    elif key == "critic":
                        st.write(f"🔍 **Critic Agent** audited quality (Score: {value.get('critic_score', 'N/A')})")
                    
                    if "response_draft" in value:
                        final_response = value["response_draft"]
            
            # حفظ المحادثة في الذاكرة — نرسل رسالة المستخدم فقط + ملخص مختصر للرد
            st.write("📝 Updating user long-term memory...")
            mem_agent = MemoryAgent(user_id=st.session_state.session_id)
            # نبعت النص بشكل واضح عشان الـ EntityExtractor يفرق بين كلام اليوزر وكلام البوت
            current_turn_chat = f"User: {prompt}\nAgent: {final_response}"
            mem_agent.process_conversation(current_turn_chat)
            mem_agent.close()
            
            status.update(label="تمت المعالجة وتحديث الذاكرة الطويلة بنجاح!", state="complete", expanded=False)
            
        st.markdown(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        st.rerun() # لإعادة تحديث الـ Profile المعروض في الـ Sidebar فوراً