"""
Self-Improving Memory Agent
"""

import os
import json
import sqlite3
import math
import textwrap
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
load_dotenv()

import time
import google.generativeai as genai
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from google.api_core import exceptions
from langchain_groq import ChatGroq

def retry_on_quota(func):
    def wrapper(*args, **kwargs):
        retries = 3
        for i in range(retries):
            try:
                return func(*args, **kwargs)
            except exceptions.ResourceExhausted as e:
                if i < retries - 1:
                    print(f"\n[Quota Exceeded] Waiting 40 seconds before retrying API call... ({i+1}/{retries})")
                    time.sleep(40)
                else:
                    raise e
    return wrapper

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHAT_MODEL     = "gemini-2.0-flash"
EMBED_MODEL    = "models/gemini-embedding-2"
DB_PATH        = "memory_store.db"
TOP_K          = 5

genai.configure(api_key=GEMINI_API_KEY)


# -------------------------------------------------------------------------
# Pydantic Models (Validation Layer)
# -------------------------------------------------------------------------
class EntityProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    favorite_brand: str = ""
    budget: str = ""
    preferred_category: str = ""
    important_preferences: list[str] = Field(default_factory=list)
    other_facts: dict[str, str | int | float | bool | list[str]] = Field(default_factory=dict)


class MemoryNode(BaseModel):
    id: int | None = None
    user_id: str
    summary: str
    embedding: list[float]
    created_at: str


# -------------------------------------------------------------------------
# Database Layer (Storage Layer)
# -------------------------------------------------------------------------
class MemoryDB:
    def __init__(self, db_path: str = DB_PATH) -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT    NOT NULL,
                data        TEXT    NOT NULL,
                created_at  TEXT    NOT NULL,
                updated_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memories (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT    NOT NULL,
                summary     TEXT    NOT NULL,
                embedding   TEXT    NOT NULL,
                created_at  TEXT    NOT NULL
            );
        """)
        self.conn.commit()

    def upsert_entity(self, user_id: str, new_profile: EntityProfile) -> None:
        row = self.conn.execute(
            "SELECT id, data FROM entities WHERE user_id = ?", (user_id,)
        ).fetchone()

        now = datetime.utcnow().isoformat()
        if row:
            existing_data = json.loads(row[1])
            new_data = new_profile.model_dump()
            merged = _merge_profiles(existing_data, new_data)
            
            try:
                final_data = EntityProfile.model_validate(merged).model_dump()
            except ValidationError:
                final_data = merged

            self.conn.execute(
                "UPDATE entities SET data = ?, updated_at = ? WHERE id = ?",
                (json.dumps(final_data, ensure_ascii=False), now, row[0])
            )
        else:
            self.conn.execute(
                "INSERT INTO entities (user_id, data, created_at, updated_at) VALUES (?,?,?,?)",
                (user_id, new_profile.model_dump_json(), now, now)
            )
        self.conn.commit()

    def get_entity(self, user_id: str) -> EntityProfile:
        row = self.conn.execute(
            "SELECT data FROM entities WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row:
            try:
                return EntityProfile.model_validate_json(row[0])
            except ValidationError:
                pass
        return EntityProfile()

    def add_memory(self, user_id: str, summary: str, embedding: list[float]) -> int:
        now = datetime.utcnow().isoformat()
        cursor = self.conn.execute(
            "INSERT INTO memories (user_id, summary, embedding, created_at) VALUES (?,?,?,?)",
            (user_id, summary, json.dumps(embedding), now)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_all_memories(self, user_id: str) -> list[MemoryNode]:
        rows = self.conn.execute(
            "SELECT id, user_id, summary, embedding, created_at FROM memories WHERE user_id = ?",
            (user_id,)
        ).fetchall()
        return [
            MemoryNode(
                id=r[0],
                user_id=r[1],
                summary=r[2],
                embedding=json.loads(r[3]),
                created_at=r[4]
            ) for r in rows
        ]

    def close(self) -> None:
        self.conn.close()


def _merge_profiles(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for k, v in updates.items():
        if not v:
            continue
        if isinstance(v, list) and isinstance(result.get(k), list):
            result[k] = list(dict.fromkeys(result[k] + v))
        else:
            result[k] = v
    return result


# -------------------------------------------------------------------------
# LLM Layer
# -------------------------------------------------------------------------
class EntityExtractor:
    PROMPT = textwrap.dedent("""\
        Analyze the conversation below and extract ONLY the user's EXPLICITLY STATED preferences.

        CRITICAL RULES:
        - ONLY extract from lines starting with "User:" — COMPLETELY IGNORE lines starting with "Agent:".
        - Extract ONLY what the USER explicitly says about their preferences.
        - If the Agent recommends products, those are NOT the user's preferences.
        - Return ONLY clean JSON. No explanation, no markdown.

        FIELD DEFINITIONS:
        
        favorite_brand: The BRAND/MANUFACTURER the user is interested in.
          KNOWN BRANDS: Apple, Samsung, Sony, Huawei, Xiaomi, OnePlus, Oppo, JBL, Anker, Logitech, Amazfit, Garmin, Fitbit, Beats, Bose
          EXAMPLES:
          - User asks about "ابل واتش" → favorite_brand = "Apple" (NOT "ابل واتش")
          - User asks about "سامسونج S24" → favorite_brand = "Samsung"
          - User asks about "سماعات سوني" → favorite_brand = "Sony"
          - User asks about "ساعات" (no brand) → favorite_brand = ""

        preferred_category: The PRODUCT TYPE/CATEGORY the user wants.
          KNOWN CATEGORIES: تليفونات, سماعات, ساعات, لابتوب, اكسسوارات
          EXAMPLES:
          - User asks about "ابل واتش" → preferred_category = "ساعات" (NOT "ابل واتش")
          - User asks about "تليفونات سامسونج" → preferred_category = "تليفونات"
          - User asks about "سماعات" → preferred_category = "سماعات"

        budget: The user's stated price range (e.g., "2000-7000 EGP").
          - Only set if the user EXPLICITLY mentions a price/budget.
        
        important_preferences: List of specific features or requirements the user mentioned.

        Conversation:
        {conversation}

        Output Format:
        {{
          "favorite_brand": "",
          "budget": "",
          "preferred_category": "",
          "important_preferences": [],
          "other_facts": {{}}
        }}
    """).strip()

    # Keyword mappings for deterministic fallback
    BRAND_KEYWORDS = {
        'apple': 'Apple', 'ابل': 'Apple', 'أبل': 'Apple', 'آبل': 'Apple', 'ايفون': 'Apple', 'آيفون': 'Apple', 'airpods': 'Apple',
        'samsung': 'Samsung', 'سامسونج': 'Samsung', 'سامسونغ': 'Samsung', 'جالاكسي': 'Samsung', 'galaxy': 'Samsung',
        'sony': 'Sony', 'سوني': 'Sony',
        'huawei': 'Huawei', 'هواوي': 'Huawei',
        'xiaomi': 'Xiaomi', 'شاومي': 'Xiaomi', 'ريدمي': 'Xiaomi',
        'oneplus': 'OnePlus', 'ون بلس': 'OnePlus',
        'oppo': 'Oppo', 'أوبو': 'Oppo',
        'jbl': 'JBL', 'جي بي ال': 'JBL',
        'anker': 'Anker', 'أنكر': 'Anker',
        'logitech': 'Logitech', 'لوجيتك': 'Logitech',
        'amazfit': 'Amazfit', 'أمازفيت': 'Amazfit',
        'garmin': 'Garmin', 'جارمن': 'Garmin',
        'fitbit': 'Fitbit', 'فيتبيت': 'Fitbit',
        'beats': 'Beats', 'بيتس': 'Beats',
        'bose': 'Bose', 'بوز': 'Bose',
    }
    CATEGORY_KEYWORDS = {
        'ساعات': 'ساعات', 'ساعة': 'ساعات', 'واتش': 'ساعات', 'watch': 'ساعات',
        'سماعات': 'سماعات', 'سماعة': 'سماعات', 'headphone': 'سماعات', 'earbuds': 'سماعات',
        'تليفون': 'تليفونات', 'تليفونات': 'تليفونات', 'موبايل': 'تليفونات', 'phone': 'تليفونات',
        'لابتوب': 'لابتوب', 'laptop': 'لابتوب',
        'باور بانك': 'اكسسوارات', 'شاحن': 'اكسسوارات', 'ماوس': 'اكسسوارات', 'كيبورد': 'اكسسوارات',
    }

    def __init__(self) -> None:
        self.model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    def _keyword_extract(self, conversation: str) -> dict:
        """Deterministic keyword-based extraction as fallback for the LLM."""
        # Only look at User lines
        user_text = ""
        for line in conversation.split("\n"):
            if line.strip().startswith("User:"):
                user_text += " " + line.lower()
        
        brand = ""
        category = ""
        
        for keyword, brand_name in self.BRAND_KEYWORDS.items():
            if keyword in user_text:
                brand = brand_name
                break
        
        for keyword, cat_name in self.CATEGORY_KEYWORDS.items():
            if keyword in user_text:
                category = cat_name
                break
        
        return {"brand": brand, "category": category}

    @retry_on_quota
    def extract(self, conversation: str) -> EntityProfile | None:
        prompt = self.PROMPT.format(conversation=conversation)
        response = self.model.invoke(prompt)
        raw = response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        try:
            data = json.loads(raw)
            if not isinstance(data, dict) or not data:
                data = {}
            
            valid_keys = set(EntityProfile.model_fields.keys())
            filtered_data = {k: v for k, v in data.items() if k in valid_keys}
            
            # Keyword-based fallback: fill empty brand/category from conversation keywords
            kw_result = self._keyword_extract(conversation)
            if not filtered_data.get("favorite_brand") and kw_result["brand"]:
                filtered_data["favorite_brand"] = kw_result["brand"]
            if not filtered_data.get("preferred_category") and kw_result["category"]:
                filtered_data["preferred_category"] = kw_result["category"]
            
            return EntityProfile.model_validate(filtered_data)
        except (json.JSONDecodeError, ValidationError):
            # Even if LLM fails completely, try keyword extraction
            kw_result = self._keyword_extract(conversation)
            if kw_result["brand"] or kw_result["category"]:
                return EntityProfile(
                    favorite_brand=kw_result["brand"],
                    preferred_category=kw_result["category"]
                )
            return None


class ConversationSummarizer:
    PROMPT = textwrap.dedent("""
        Create a concise long-term memory summary from the conversation below.

        Rules:
        - Keep only important long-term information.
        - Remove temporary or one-time details.
        - Maximum 2 sentences.
        - Write in third-person about the user.

        Conversation:
        {conversation}

        Summary:
    """).strip()

    def __init__(self) -> None:
        self.model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    @retry_on_quota
    def summarize(self, conversation: str) -> str:
        prompt = self.PROMPT.format(conversation=conversation)
        return self.model.invoke(prompt).content.strip()


class MemoryUpdater:
    PROMPT = textwrap.dedent("""
        You are a memory manager. Merge the new information into the existing memory.

        Rules:
        - Keep all unique important facts.
        - If new info contradicts old info, prefer the newer one.
        - Remove duplicates.
        - Be concise — max 3 sentences total.
        - Output only the merged memory text, no explanation.

        Existing Memory:
        {existing_memory}

        New Information:
        {new_summary}

        Merged Memory:
    """).strip()

    def __init__(self) -> None:
        self.model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    @retry_on_quota
    def update(self, existing_memory: str, new_summary: str) -> str:
        if not existing_memory.strip():
            return new_summary

        prompt = self.PROMPT.format(existing_memory=existing_memory, new_summary=new_summary)
        return self.model.invoke(prompt).content.strip()


class RetrievalEngine:
    RERANK_PROMPT = textwrap.dedent("""
        Given the user query and the candidate memories below, return ONLY the
        memories that are actually relevant to the query.

        Output as a JSON list of the relevant memory texts.
        If none are relevant, return an empty list [].

        Query: {query}

        Candidate Memories:
        {candidates}

        Relevant Memories (JSON list):
    """).strip()

    def __init__(self) -> None:
        self.model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    @retry_on_quota
    def _embed(self, text: str) -> list[float]:
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=text,
            task_type="retrieval_query"
        )
        return result["embedding"]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x ** 2 for x in a))
        mag_b = math.sqrt(sum(x ** 2 for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    @retry_on_quota
    def retrieve(
        self,
        query: str,
        memories: list[MemoryNode],
        top_k: int = TOP_K,
        rerank: bool = True
    ) -> list[str]:
        if not memories:
            return []

        query_emb = self._embed(query)

        scored: list[tuple[float, str]] = []
        for mem in memories:
            score = self._cosine_similarity(query_emb, mem.embedding)
            scored.append((score, mem.summary))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_summaries = [s for _, s in scored[:top_k]]

        if not rerank:
            return top_summaries

        candidates_text = "\n".join(f"- {s}" for s in top_summaries)
        prompt = self.RERANK_PROMPT.format(query=query, candidates=candidates_text)
        response = self.model.invoke(prompt)
        raw = response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        try:
            filtered = json.loads(raw)
            return filtered if isinstance(filtered, list) else top_summaries
        except json.JSONDecodeError:
            return top_summaries


# -------------------------------------------------------------------------
# Agent Orchestrator
# -------------------------------------------------------------------------
class MemoryAgent:
    def __init__(self, user_id: str, db_path: str = DB_PATH) -> None:
        self.user_id = user_id
        
        # FIX: Update the db_path to point directly to memory_store.db or create it
        # Since memory_agent is a directory, DB_PATH should probably be in data or local.
        # But for now, we'll just use the default.
        self.db = MemoryDB(db_path)
        self.extractor = EntityExtractor()
        self.summarizer = ConversationSummarizer()
        self.updater = MemoryUpdater()
        self.retriever = RetrievalEngine()

    def process_conversation(self, conversation: str) -> dict[str, Any]:
        profile = self.extractor.extract(conversation)
        if profile:
            self.db.upsert_entity(self.user_id, profile)

        summary = self.summarizer.summarize(conversation)

        existing_memories = self.db.get_all_memories(self.user_id)
        if existing_memories:
            latest_summary = existing_memories[-1].summary
            merged_summary = self.updater.update(latest_summary, summary)
        else:
            merged_summary = summary

        @retry_on_quota
        def _get_embedding():
            return genai.embed_content(
                model=EMBED_MODEL,
                content=merged_summary,
                task_type="retrieval_document"
            )["embedding"]
            
        embedding = _get_embedding()

        self.db.add_memory(self.user_id, merged_summary, embedding)

        return {
            "entities": profile.model_dump() if profile else None,
            "summary": merged_summary,
            "status": "ok"
        }

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[str]:
        memories = self.db.get_all_memories(self.user_id)
        return self.retriever.retrieve(query, memories, top_k=top_k)

    def get_user_profile(self) -> EntityProfile:
        return self.db.get_entity(self.user_id)

    def close(self) -> None:
        self.db.close()
