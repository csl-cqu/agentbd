# agents.py
from langgraph.prebuilt import create_react_agent
import os
from handoff import  create_task_description_handoff_tool
from config import OPENAI_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# handoff
assign_to_auditor_agent = create_task_description_handoff_tool(
    agent_name="auditor_agent",
    description="Assign task to a auditor agent.",
)

assign_to_mitigator_agent = create_task_description_handoff_tool(
    agent_name="mitigator_agent",
    description="Assign task to a mitigator agent.",
)

supervisor_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[
        assign_to_auditor_agent,
        assign_to_mitigator_agent,
    ],
    # prompt=(
    #     "You are a supervisor managing two agents:\n"
    #     "- a auditor agent. Assign detetect and validate-related tasks to this assistant\n"
    #     "- a mitigator agent. Assign mitigator-related tasks to this assistant\n"
    #     "- Assign work to one agent at a time, do not call agents in parallel.\n"
    #     "- If the Attack Type is not clean, you should ask the mitigator agent to mitigate the backdoor in the text.\n"
    #     "- You should ensure the text return to user is clean and without backdoor.\n"
    #     "- Do not do any work yourself.\n"
    #     "- Do not change the result name returned by the agent.\n"
    #     "- You should use the following format to respond the result: \n"
    #     "*** Response: \n"
    #     "### Mitigated Text: [text]\n"      
    # ),
    prompt = (
        "You are the Supervisor managing two specialized agents:\n\n"

        "AGENT ROLES:\n"
        "1. AUDITOR AGENT:\n"
        "- Responsible for:\n"
        "• Attack detection\n"
        "• Attack validation\n"
        "- Assign: All detection & validation tasks\n\n"

        "2. MITIGATOR AGENT:\n"
        "- Responsible for:\n"
        "• Backdoor removal\n"
        "• Text sanitization\n"
        "- Assign: All mitigation tasks\n\n"

        "OPERATION RULES:\n"
        "1. Sequential Processing:\n"
        "- Assign tasks to ONE agent at a time\n"
        "- NEVER call agents in parallel\n"
        "- Workflow: Auditor → Mitigator (if needed) → Final output\n\n"

        "2. Mitigation Trigger:\n"
        "- REQUIRED when:\n" 
        "• Attack Type ≠ 'clean'\n"
        "• Validation Result = 'FAIL'\n"
        "SKIP when:\n"
        "• Attack Type = 'clean'"
        "• Validation Result = 'PASS'\n\n"

        "3. Output Requirements:\n"
        "- FINAL output MUST be clean and backdoor-free\n"
        "- PRESERVE original result names from agents\n"
        "- DO NOT perform any direct text processing\n\n"

        "STRICT OUTPUT FORMAT:\n"
        "*** Response:\n"
        "### Mitigated Text: [final_clean_text]"
    ),
    name="supervisor",
)