"""
User-prompt templates for every step of Graph-of-Agents.

The *system* prompt of each agent (its personality and rules) lives in its YAML file.
These templates build the *user* message for one step. They follow the prompts in
Appendix B of the Graph-of-Agents paper, adapted to Nash (Reformed theology + RAG context).

Keeping all prompts in one file means you can read the whole "conversation" of the
algorithm in one place, and the service files stay short.
"""


# Small models (8B) tend to put everything in "reasoning" and leave "answer" as one line,
# and to copy context labels such as "[G1]" into the reference lists. Other agents only
# read "answer" and the references, so these rules are added to every answering step.
ANSWER_RULES = (
    "Output rules:\n"
    "- 'reasoning': short working notes (at most 3 sentences).\n"
    "- 'answer': the COMPLETE answer for the reader, 2 to 4 paragraphs. Never leave it empty or write only 'yes'/'no'. "
    "You may cite context labels such as [V1] or [G2] inside the answer text.\n"
    "- 'scripture_references': real references with full book names, e.g. 'Romans 3:28', never labels like [G1]; "
    "give each one a short note on why it matters.\n"
    "- 'confessional_references' (if your schema has it): document WCF/WLC/WSC and a locator such as '11.1', with a short note."
)


# =========================================================
# STEP 1 — NODE SAMPLING (Meta-LLM chooses the agents)
# =========================================================
def node_sampling_prompt(query: str, k: int, agent_cards: str) -> str:
    return (
        f"Question:\n{query}\n\n"
        f"Agent cards:\n{agent_cards}\n\n"
        f"Select exactly {k} agents best suited to answer this question. "
        f"Use only the roles listed in the agent cards."
    )


# =========================================================
# STEP 2 — INITIAL RESPONSE (every selected agent answers alone)
# =========================================================
def initial_response_prompt(query: str, context: str) -> str:
    return (
        f"Context from Nash's knowledge base:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer the question from your specialist perspective, and say how sure you are in "
        "'confidence' (0.0 to 1.0).\n\n"
        f"{ANSWER_RULES}"
    )


# =========================================================
# STEP 3 — EDGE SAMPLING (each agent scores the other agents)
# =========================================================
def edge_scoring_prompt(query: str, labelled_responses: str, response_ids: list[str]) -> str:
    return (
        f"Question:\n{query}\n\n"
        f"Responses from other agents:\n{labelled_responses}\n\n"
        "Score each response from 0.0 to 1.0 on correctness, coherence and relevance to the question. "
        "Be critical: judge biblical faithfulness and confessional accuracy, not length or style. "
        f"Give exactly one score for each of these response ids: {', '.join(response_ids)}."
    )


# =========================================================
# STEP 4 — MESSAGE PASSING
# =========================================================
def source_to_target_prompt(query: str, context: str, own_response: str, source_responses: str) -> str:
    """A *target* agent reads the answers of its more relevant *source* agents."""
    return (
        f"Context from Nash's knowledge base:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        f"Your initial response:\n{own_response}\n\n"
        f"Responses from other agents, most relevant first:\n{source_responses}\n\n"
        "Integrate useful insights from these responses to improve your answer. "
        "Be critical — some information may be incorrect. Keep your own specialist perspective "
        "and give your refined answer in the same JSON format.\n\n"
        f"{ANSWER_RULES}"
    )


def target_to_source_prompt(query: str, context: str, own_response: str, target_responses: str) -> str:
    """A *source* agent reads how its *target* agents refined their answers after seeing its answer."""
    return (
        f"Context from Nash's knowledge base:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        f"Your current response:\n{own_response}\n\n"
        f"Other agents refined their answers after seeing yours, most relevant first:\n{target_responses}\n\n"
        "Use their improvements to finalize your response. "
        "Be critical — some information may be incorrect. Give your final answer in the same JSON format.\n\n"
        f"{ANSWER_RULES}"
    )


# =========================================================
# STEP 5 — GRAPH POOLING (mean pooling: Meta-LLM synthesises)
# =========================================================
def mean_pooling_prompt(query: str, context: str, weighted_responses: str) -> str:
    return (
        f"Context from Nash's knowledge base:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        f"Refined responses of the specialist agents, with their relevance scores:\n{weighted_responses}\n\n"
        "Synthesize these responses into one final answer. Give more weight to responses with higher "
        "relevance scores. Be critical — some information may be incorrect. List the genuine disagreements "
        "you could not resolve in 'unresolved_disagreements'.\n\n"
        f"{ANSWER_RULES}"
    )
