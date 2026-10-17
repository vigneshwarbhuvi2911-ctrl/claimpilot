"""The claims workflow as a LangGraph StateGraph."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agents.nodes import ClaimState, Deps, make_nodes, route_after_adjudication


def build_graph(deps: Deps):
    n = make_nodes(deps)
    g = StateGraph(ClaimState)
    for name, fn in n.items():
        g.add_node(name, fn)
    g.add_edge(START, "intake")
    g.add_edge("intake", "document_reader")
    g.add_edge("document_reader", "exclusion_screener")
    g.add_edge("exclusion_screener", "adjudicate")
    g.add_conditional_edges("adjudicate", route_after_adjudication,
                            {"request_documents": "request_documents", "recommendation_writer": "recommendation_writer"})
    g.add_edge("request_documents", "finalize")
    g.add_edge("recommendation_writer", "finalize")
    g.add_edge("finalize", END)
    return g.compile()
