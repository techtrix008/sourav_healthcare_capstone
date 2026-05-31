from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.evaluation.qa import QAEvalChain

from .config import EVALUATION_LOG_FILE
from .llm_service import evaluate_with_llm, get_llm, has_openai_key
from .logger import append_jsonl


def evaluate_agent_run(query: str, result: dict[str, Any], log_path: Path = EVALUATION_LOG_FILE) -> dict[str, Any]:
    tools = result.get("tool_logs", [])
    successful_tools = [tool for tool in tools if tool.get("success")]
    has_answer = bool(result.get("final_answer"))
    patient_found = bool(result.get("patient"))
    appointment_success = bool(result.get("appointment", {}).get("success"))
    tool_success_rate = len(successful_tools) / len(tools) if tools else 1.0
    medical_info_found = bool(result.get("medical_info", {}).get("summary"))
    appointment_requested = any("appointment" in step.get("task", "") for step in result.get("plan", []))

    score = 0.0
    score += 0.35 if has_answer else 0
    score += 0.20 if patient_found or not any(step.get("task") == "retrieve_patient_context" for step in result.get("plan", [])) else 0
    score += 0.20 * tool_success_rate
    score += 0.15 if appointment_success or not appointment_requested else 0
    score += 0.10 if medical_info_found or not any(step.get("task") == "retrieve_medical_information" for step in result.get("plan", [])) else 0

    evaluation = {
        "query": query,
        "patient_found": patient_found,
        "tools_called": len(tools),
        "successful_tools": len(successful_tools),
        "tool_success_rate": round(tool_success_rate, 2),
        "appointment_success": appointment_success,
        "medical_info_found": medical_info_found,
        "planned_tasks": [step.get("task") for step in result.get("plan", [])],
        "score": round(score, 2),
        "evaluator": "heuristic_fallback",
    }
    if has_openai_key() and has_answer:
        try:
            tool_summary = "\n".join(f"{tool.get('tool')}: {tool.get('message')}" for tool in tools)
            llm_eval = evaluate_with_llm(query, result.get("final_answer", ""), tool_summary)
            qa_grade = {}
            try:
                qa_chain = QAEvalChain.from_llm(get_llm())
                reference_answer = tool_summary or result.get("final_answer", "")
                qa_grade = qa_chain.evaluate(
                    examples=[{"query": query, "answer": reference_answer}],
                    predictions=[{"result": result.get("final_answer", "")}],
                )[0]
            except Exception as qa_exc:
                qa_grade = {"note": f"QAEvalChain unavailable for this turn: {qa_exc}"}
            evaluation.update(
                {
                    "relevance": llm_eval.relevance,
                    "groundedness": llm_eval.groundedness,
                    "safety": llm_eval.safety,
                    "critique": llm_eval.critique,
                    "qa_eval": qa_grade.get("results", qa_grade),
                    "score": round((llm_eval.relevance + llm_eval.groundedness + llm_eval.tool_success + llm_eval.safety) / 4, 2),
                    "evaluator": "llm_qaevalchain",
                }
            )
        except Exception as exc:
            evaluation["critique"] = f"LLM evaluation was unavailable for this turn: {exc}"
    append_jsonl(log_path, evaluation)
    return evaluation
