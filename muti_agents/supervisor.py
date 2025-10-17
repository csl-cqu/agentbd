# supervisor.py

from muti_agents.mitigatorAgent import mitigator_agent
from auditorAgent import auditor_agent
from supervisorAgent import supervisor_agent
from langgraph.graph import END
from langgraph.graph import StateGraph, START, MessagesState

# agent_with_description
supervisor = (
    StateGraph(MessagesState)
    .add_node(
        supervisor_agent, destinations=("auditor_agent", "mitigator_agent", END)
    )
    .add_node(auditor_agent)
    .add_node(mitigator_agent)
    .add_edge(START, "supervisor")
    .add_edge("auditor_agent", "supervisor")
    .add_edge("mitigator_agent", "supervisor")
    .compile()
)