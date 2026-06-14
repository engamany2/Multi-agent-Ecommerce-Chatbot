import os
import re
from dotenv import load_dotenv
from typing import TypedDict, Literal

# 1. تحميل البيئة المحلية
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

# 2. استدعاء الموديولات الحقيقية الخاصة بالفريق بالكامل (Modular Code)
from router import router_with_memory
from product_search import ProductSearch
from FAQ import get_faq_answer
from memory_agent import MemoryAgent  # استدعاء عميل الذاكرة الذكي الجديد

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# --- 3. تعريف الـ AgentState المتكامل للمشروع مع الذاكرة ---
class AgentState(TypedDict):
    user_query: str
    language: str         
    route_destination: str 
    retrieved_info: str    
    user_memories: str     # الذاكرة المسترجعة من الـ Gemini MemoryAgent
    response_draft: str
    critic_score: int
    critic_feedback: str
    iterations: int

# تهيئة الموديل
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# استدعاء محركات البحث والذاكرة الحقيقية
product_search_engine = ProductSearch()

# --- 4. بناء الـ Nodes (العقد الحقيقية) ---

def router_node(state: AgentState):
    print(f"\n--- [NODE] ROUTER AGENT ---")
    session_id = "notebook_test_session"
    config = {"configurable": {"session_id": session_id}}
    
    # أ] استدعاء الـ Router
    router_response = router_with_memory.invoke({"input": state["user_query"]}, config=config)
    destination = router_response.destination
    print(f"Decision: Routed to -> {destination.upper()}")
    
    # ب] استدعاء الـ MemoryAgent لجلب معلومات العميل التاريخية (إن وجدت)
    memory_agent = MemoryAgent(user_id=session_id)
    history_hits = memory_agent.retrieve(state["user_query"], top_k=2)
    profile = memory_agent.get_user_profile()
    
    # تنسيق شكل الذاكرة المسترجعة
    profile_facts = f"Budget: {profile.budget}, Fav Brand: {profile.favorite_brand}"
    past_memories = "\n".join([f"- {h}" for h in history_hits])
    user_memories_context = f"Profile Facts: {profile_facts}\nPast Relevant Memories:\n{past_memories}"
    memory_agent.close()

    context = ""
    # ج] دمج خطوط جلب البيانات الحقيقية (Real Data Pipelines)
    if destination == "product_search":
        context = product_search_engine.get_product(state["user_query"])
    elif destination == "faq_agent":
        context = get_faq_answer(state["user_query"])
    elif destination == "order_tracking":
        context = "المعلومات للمراجع: نظام تتبع الشحنات قيد الربط، اطلب من العميل رقم الشحنة بلطف."
    elif destination == "chitchat":
        context = "المعلومات للمراجع: محادثة عامة أو ترحيب، رحب بالعميل بأسلوب High Tech."
    else:
        context = "المعلومات للمراجع: هذا الطلب خارج تخصص المتجر."

    return {
        "route_destination": destination, 
        "retrieved_info": context,
        "user_memories": user_memories_context
    }

def writer_node(state: AgentState):
    print(f"\n--- [NODE] WRITER AGENT (Iteration: {state.get('iterations', 0) + 1}) ---")
    target_lang = "Arabic (Egyptian dialect)" if state.get("language") == "Arabic" else "English"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a professional customer service agent for 'High Tech' electronics store.
        
        CRITICAL RULES — FOLLOW THEM EXACTLY:
        1. LANGUAGE: You MUST respond ONLY in {target_lang}. If Arabic: Use natural Egyptian Ammiya. NEVER mix English except brand names.
        2. SOURCE OF TRUTH: 'Retrieved Info' is YOUR ONLY SOURCE for product availability, names, prices. NEVER invent products or prices.
        3. PERSONALIZATION (Secondary): Use 'User Long-term Memories' ONLY to add a personal touch. NEVER let Memory override Retrieved Info.
        4. PRODUCT LISTING: Always list Product Name — Price EGP — Brief description.
        5. TONE: Friendly, professional, premium feel."""),
        ("user", "User Query: {query} \n\n Retrieved Info: {info} \n\n User Long-term Memories: {memories} \n\n Critic Feedback: {feedback}")
    ])
    chain = prompt | llm
    response = chain.invoke({
        "query": state["user_query"],
        "info": state["retrieved_info"],
        "memories": state["user_memories"],
        "feedback": state.get("critic_feedback", "لا يوجد")
    })
    return {"response_draft": response.content, "iterations": state.get("iterations", 0) + 1}

def critic_node(state: AgentState):
    print(f"\n--- [NODE] CRITIC AGENT ---")
    target_lang = "Arabic" if state.get("language") == "Arabic" else "English"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a strict quality controller. Give a score 1-10.
        Ensure language is STRICTLY {target_lang} and there is no hallucination or refusal. Respond with the NUMBER ONLY."""),
        ("user", f"User Query: {state['user_query']} \n\n Agent Response: {state['response_draft']}")
    ])
    chain = prompt | llm
    result = chain.invoke({})
    try:
        score = int(''.join(filter(str.isdigit, result.content)))
    except:
        score = 8 
        
    print(f"Quality Score given by Critic: {score}")
    return {"critic_score": score, "critic_feedback": result.content}

def should_continue(state: AgentState):
    if state["critic_score"] < 7 and state["iterations"] < 3:
        print("--- DECISION: RE-WRITE NEEDED ---")
        return "re_write"
    print("--- DECISION: FINISHED & APPROVED ---")
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

app = workflow.compile()
print("[SUCCESS] Full Modular Graph with Gemini Memory Architecture is ready!")