"""
pipeline/nodes/abstainer.py — Sprint 5

Honest no-answer node. No LLM call — fixed language-aware message.
Triggered when:
  - intent == 'out_of_scope'
  - critic verdict == 'abstain' (max rounds exhausted or fundamentally insufficient)

Message template follows the charter:
  "Based on {rounds} search round(s), insufficient grounded evidence was found..."
"""

from __future__ import annotations

from pipeline.state import MeridianState

_ABSTAIN_MESSAGES = {
    "en": (
        "Based on {rounds} search round(s), insufficient grounded evidence was found "
        "in the Meridian knowledge base to answer this question."
    ),
    "hi": (
        "{rounds} खोज प्रयास के बाद, मेरिडियन नॉलेज बेस में इस प्रश्न का उत्तर देने के लिए "
        "पर्याप्त प्रमाण नहीं मिले।"
    ),
    "zh": (
        "经过 {rounds} 次检索，在 Meridian 知识库中未能找到足够的证据来回答这个问题。"
    ),
}

_OUT_OF_SCOPE_MESSAGES = {
    "en": (
        "This question is outside the scope of the Meridian knowledge base, "
        "which covers HR policies, IT Security, Product specifications, and Executive Comms."
    ),
    "hi": (
        "यह प्रश्न मेरिडियन नॉलेज बेस के दायरे से बाहर है। "
        "नॉलेज बेस में HR नीतियां, IT सुरक्षा, उत्पाद विशेषताएं और कार्यकारी संचार शामिल हैं।"
    ),
    "zh": (
        "此问题超出了 Meridian 知识库的范围，"
        "该知识库涵盖 HR 政策、IT 安全、产品规格和高管沟通。"
    ),
}


def abstain(state: MeridianState) -> dict:
    """LangGraph node — return an honest no-answer response.

    Reads:  intent, lang, retrieval_round
    Writes: answer, sources, chunks_used, abstained
    """
    lang = state.get("lang", "en")
    intent = state.get("intent", "")
    rounds = state.get("retrieval_round", 0)

    if intent == "out_of_scope":
        template = _OUT_OF_SCOPE_MESSAGES.get(lang, _OUT_OF_SCOPE_MESSAGES["en"])
        answer = template
    else:
        template = _ABSTAIN_MESSAGES.get(lang, _ABSTAIN_MESSAGES["en"])
        answer = template.format(rounds=rounds)

    return {
        "answer":      answer,
        "sources":     [],
        "chunks_used": 0,
        "abstained":   True,
    }
