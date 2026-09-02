"""LangGraph node implementations for chat workflow."""

from __future__ import annotations

import asyncio
import json
from langchain_core.messages.base import BaseMessage
from pathlib import Path
from typing import Any, Dict, List, Optional
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from loguru import logger
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from app.config import settings
from app.services.pinecone_service import (
    PineconeServiceError,
    RetrievedChunk,
    assemble_document_context,
    build_qna_context,
    build_summary_context,
    fetch_full_document_chunks,
    query_context,
)

def _preview(text: str, length: int = 120) -> str:
    text = (text or "").replace("\n", " ")
    return text if len(text) <= length else f"{text[:length]}…"


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _require_setting(value: Optional[str], name: str) -> str:
    if not value:
        raise ValueError(f"Missing required configuration: {name}")
    return value


def _load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


DECIDE_PROMPT = _load_prompt("decide_next_step.txt")
QNA_PROMPT = _load_prompt("qna.txt")
SUMMARY_PROMPT = _load_prompt("summarization.txt")


if settings.HUGGINGFACE_API_KEY and not settings.OPENAI_API_KEY:
    # Use Hugging Face.
    # NOTE: newer huggingface_hub versions reject max_retries on the async chat call,
    # so we keep the retry setting only for the OpenAI path and avoid passing it here.
    _hf_llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # Free model
        huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
        temperature=0.2,
        timeout=settings.LLM_TIMEOUT,
        task="text-generation",
    )
    _chat_llm = ChatHuggingFace(llm=_hf_llm)
elif settings.OPENAI_API_KEY:
    # Use OpenAI if available
    _chat_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=settings.OPENAI_API_KEY,
        timeout=settings.LLM_TIMEOUT,
        max_retries=settings.LLM_MAX_RETRIES,
    )
else:
    pipe = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_length=512,
        temperature=0.2,
    )
    _chat_llm = HuggingFacePipeline(pipeline=pipe)
    
_google_api_key = _require_setting(settings.GOOGLE_API_KEY, "GOOGLE_API_KEY")

_gemini_router_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-pro-preview",
    temperature=0.1,
    google_api_key=_google_api_key,
)

_gemini_summary_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-pro-preview",
    temperature=0.3,
    google_api_key=_google_api_key,
)


def _latest_human_text(messages: List[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _append_ai_message(messages: List[BaseMessage], content: str) -> List[BaseMessage]:
    updated = list(messages)
    updated.append(AIMessage(content=content))
    return updated


def _truncate(text: str, limit: int = 1500) -> str:
    text = text or ""
    return text if len(text) <= limit else f"{text[:limit]}…"


def _chunk_snapshot(chunks: List[Any], limit: int = 3) -> List[Dict[str, Any]]:
    snapshot: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks[:limit], start=1):
        if hasattr(chunk, "metadata"):
            metadata = getattr(chunk, "metadata") or {}
            text_value = getattr(chunk, "text", "")
            score_value = getattr(chunk, "score", 0.0)
        else:
            metadata = chunk.get("metadata") or {}
            text_value = chunk.get("text") or metadata.get("text") or ""
            score_value = chunk.get("score") or 0.0
        snapshot.append({
            "rank": idx,
            "score": round(score_value if isinstance(score_value, (int, float)) else 0.0, 4),
            "source_file": metadata.get("source_file"),
            "page_index": metadata.get("page_index"),
            "chunk_index": metadata.get("chunk_index"),
            "text_preview": _preview(text_value or metadata.get("text") or "", 200),
        })
    return snapshot


def _snapshot_json(snapshot: List[Dict[str, Any]]) -> str:
    if not snapshot:
        return "[]"
    return json.dumps(snapshot, ensure_ascii=False, indent=2)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _document_sort_key(chunk: RetrievedChunk) -> tuple[int, int]:
    metadata = chunk.metadata or {}
    return (
        _safe_int(metadata.get("page_index"), default=1_000_000),
        _safe_int(metadata.get("chunk_index"), default=1_000_000),
    )

async def decide_next_step(state):
    """Simple rule-based routing - no API needed"""
    message = state.get("message", "") or ""
    if not message:
        messages = state.get("messages", []) or []
        if messages:
            message = str(messages[-1].content) if hasattr(messages[-1], 'content') else str(messages[-1])
    
    message = message.lower()
    
    summary_keywords = ["summarize", "summary", "summarise", "sum up"]
    
    if any(word in message for word in summary_keywords):
        logger.info("Routing to summarization")
        return {"next": "summarize"}
    
    logger.info("Routing to QnA")
    return {"next": "qna"} 

async def question_answering_agent(state):
    """QnA agent using retrieved context"""
    
    # Get message from state
    message = state.get("message", "") or ""
    if not message:
        messages = state.get("messages", []) or []
        if messages:
            message = str(messages[-1].content) if hasattr(messages[-1], 'content') else str(messages[-1])
    
    logger.info(f"QnA agent received message: {message[:100]}")
    
    # Get retrieved chunks
    chunks = await query_context(
        query=message,  # Use extracted message
        dataset_name=state.get("dataset_name"),
        top_k=settings.RAG_TOP_K,
    )
    
    logger.info(f"Retrieved {len(chunks)} chunks")
    
    if not chunks:
        return {
            "answer": "I'm not sure based on the available information.",
            "context_chunks": 0,
            "citations": None,
        }
    
    # Build context
    context_text, citations = build_qna_context(chunks)
    
    # Return top chunk text as answer
    answer = chunks[0].text[:500]
    
    return {
        "answer": answer,
        "context_chunks": len(chunks),
        "citations": citations,
    }

async def summary_generation_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    messages: List[BaseMessage] = state.get("messages", []) or []
    dataset_name: Optional[str] = state.get("dataset_name")
    user_text = _latest_human_text(messages)

    logger.info(
        "Starting summarization agent",
        text_preview=_preview(user_text),
        dataset=dataset_name,
    )

    error_message: Optional[str] = None
    context_chunks: List[RetrievedChunk] = []

    truncation_applied = False
    context_text = ""
    context_chunk_count = 0

    try:
        if settings.SUMMARY_FULL_DOCUMENT_MODE:
            initial_chunks = await query_context(
                query=user_text,
                top_k=1,
                dataset_name=dataset_name,
            )
            initial_snapshot = _chunk_snapshot(initial_chunks, limit=1)
            logger.info(
                "Summarization top-1 retrieval ({} chunk) for dataset '{}':\n{}",
                len(initial_chunks),
                dataset_name or "default",
                _snapshot_json(initial_snapshot),
            )

            if initial_chunks:
                best_chunk = initial_chunks[0]
                metadata = best_chunk.metadata or {}
                document_id = metadata.get("document_id")
                source_file = metadata.get("source_file")
                logger.info(
                    "Summarization document target identified",
                    document_id=document_id,
                    source_file=source_file,
                    dataset=dataset_name,
                )

                full_chunks = await asyncio.to_thread(
                    fetch_full_document_chunks,
                    document_id=document_id,
                    source_file=source_file,
                    dataset_name=dataset_name,
                )

                if full_chunks:
                    context_chunks = sorted(full_chunks, key=_document_sort_key)
                    context_chunk_count = len(context_chunks)
                    context_text, truncation_applied = assemble_document_context(
                        context_chunks,
                        max_chars=settings.SUMMARY_MAX_CONTEXT_CHARS,
                    )
                    if truncation_applied:
                        logger.warning(
                            "Summarization context truncated",
                            document_id=document_id,
                            source_file=source_file,
                            max_chars=settings.SUMMARY_MAX_CONTEXT_CHARS,
                        )
                    logger.info(
                        "Summarization full document fetched",
                        document_id=document_id,
                        source_file=source_file,
                        total_chunks=context_chunk_count,
                    )
                    logger.info(
                        "Summarization context preview:\n{}",
                        _preview(context_text, 2000),
                    )
                else:
                    logger.warning(
                        "Full document retrieval returned no chunks",
                        document_id=document_id,
                        source_file=source_file,
                        dataset=dataset_name,
                    )

        if not context_text:
            if not settings.SUMMARY_FULL_DOCUMENT_MODE:
                context_chunks = await query_context(
                    query=user_text,
                    top_k=settings.RAG_TOP_K,
                    dataset_name=dataset_name,
                )
                snapshot = _chunk_snapshot(context_chunks)
                logger.info(
                    "Summarization context retrieved ({} chunks) for dataset '{}':\n{}",
                    len(context_chunks),
                    dataset_name or "default",
                    _snapshot_json(snapshot),
                )
                context_text = build_summary_context(context_chunks)
                context_chunk_count = len(context_chunks)
            else:
                context_text = "Context unavailable or insufficient."
                context_chunk_count = 0
    except PineconeServiceError as exc:
        error_message = str(exc)
        logger.error(f"Pinecone retrieval failed for summarization: {exc}")
        context_text = "Context unavailable or insufficient."
    except Exception as exc:  # pragma: no cover - safety net
        error_message = "Context retrieval failed. Proceeding without context."
        logger.error(f"Unexpected summarization retrieval error: {exc}")
        context_text = "Context unavailable or insufficient."

    if not context_text:
        context_text = "Context unavailable or insufficient."

    prompt = SUMMARY_PROMPT.format(text=user_text, context=context_text)
    logger.info(
        f"Summarization prompt constructed:\n{prompt}",
    )

    try:
        response = await _gemini_summary_llm.ainvoke([HumanMessage(content=prompt)])
        summary = (response.content or "").strip()
        if not summary:
            summary = "I was unable to create a summary with the available information."
        logger.info(
            "Summarization agent completed",
            summary_preview=_preview(summary),
            context_chunks=context_chunk_count,
        )
    except Exception as exc:
        logger.error(f"Summarization generation failed: {exc}")
        summary = "I'm sorry, I couldn't generate a summary right now."
        error_message = error_message or "Summary generation failed."

    updated_messages = _append_ai_message(messages, summary)

    result: Dict[str, Any] = {
        "summary": summary,
        "messages": updated_messages,
        "context_chunks": context_chunk_count,
    }
    if error_message:
        result["error"] = error_message
    return result


