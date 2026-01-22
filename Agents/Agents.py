import fastmcp as fm
from fastmcp import FastMCP
from pydantic_ai.toolsets.fastmcp import FastMCPToolset
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelMessage, ModelSettings, UsageLimitExceeded, UsageLimits
from pydantic_ai import ModelSettings
from pydantic_graph import BaseNode, Graph, GraphRunContext, End
from typing import List, Literal
import logfire
from utilities.Types import Settings, ChatState, Interrogator_Output, Prisoner_1_Output, Prisoner_2_Output
from utilities.Instructions import get_interrogator_instructions, get_Prisoner1_instructions, get_Prisoner2_instructions, get_summary_instructions
import asyncio

# =========================================
# Simple Agents
# =========================================


#####
#Very important to specify roleplaying in the instructions or else chatgpts legal guardrails wont allow it with some models.
####


toolset = FastMCPToolset("MCP_Server.py")

toolnames = list("nash_equilibria")


Prisoner_1 = Agent(model= 'openai:gpt-4o-mini',
                   output_type = str,
                   #update this with get instructions once in prod
                   #instructions = get_Prisoner1_instructions(toolnames), #task we need the agent to complete would go here
                   system_prompt = 'Limit your responses to 2-3 sentences', # context about the game and scenario here #'Role: You are a prisoner stuck inside of the prisoner dilemma scenario, where you and your friend, prisoner 2, have committed a crime...')
                   #deps_type = Settings,
                   name = "Prisoner 1",
                   toolsets = [toolset],
                   model_settings= ModelSettings(
                       max_tokens = 4000, #limiting the amount of tokens.
                       temperature= 1.0, #we want a very creative response for this research
                       timeout= 20 #20 seconds for timeout, does timeout here get superceeded by runtime ?
                   ),
                   retries = 1,
                   output_retries = 1,
                   instrument = True #set to true for logfire
)


Prisoner_2 = Agent(model= 'openai:gpt-4o-mini',
                   output_type = str,
                   #instructions = get_Prisoner2_instructions(toolnames), #task we need the agent to complete would go here
                   system_prompt = 'Limit your responses to 2-3 sentences', # context about the game and scenario here #'Role: You are a prisoner stuck inside of the prisoner dilemma scenario, where you and your friend, prisoner 2, have committed a crime...')
                   #deps_type = Settings,
                   name = "Prisoner 2",
                   toolsets = [toolset],
                   model_settings= ModelSettings(
                       max_tokens = 4000, #limiting the amount of tokens.
                       temperature= 1.0, #we want a very creative response for this research
                       timeout= 20 #20 seconds for timeout
                   ),
                   retries = 1,
                   output_retries = 1,
                   instrument = True #set to true for logfire
)

#---------
# Important note when setting the models
# In this case of the interrogator , when a model has to run tools.
'''
Tool usage requires two LLM calls:
The model outputs the tool call JSON
After your tool executes, the model produces the follow-up reasoning or next step
    - GPT-5 is larger → both of these steps take longer.
     If your graph calls a tool every turn, the slowdown compounds.
    - GPT-4.1 or GPT-4o-mini is usually 5–20× faster for tool JSON than GPT-5.

Because GPT-5 has stronger reasoning, it often:
    - decides it should use the tool
    - even when the answer doesn’t require it
    - OR it calls the same tool repeatedly
    - This matches exactly what you described earlier:
    - “my agent keeps calling the same tool”
    - Large models overfit to tool usage instructions.

    See more in the notes above.
'''
#---------




Interrogator_ = Agent(model= 'openai:gpt-4o-mini',
                   output_type = str,
                   #instructions = get_interrogator_instructions(), #task we need the agent to complete would go here
                   #system_prompt = 'You are helping me test research questions', # context about the game and scenario here #'Role: You are a prisoner stuck inside of the prisoner dilemma scenario, where you and your friend, prisoner 2, have committed a crime...')
                   #deps_type = Settings,
                   name = "Interrogator",
                   toolsets = [toolset],
                   model_settings= ModelSettings(
                       max_tokens = 5000, #limiting the amount of tokens.
                       temperature= 1.0, #we want a very creative response for this research
                       timeout= 20 #20 seconds for timeout
                   ),
                   retries = 1,
                   output_retries = 1,
                   instrument = True #set to true for logfire
)


Summarizer = Agent(model= 'openai:gpt-5-mini',
                   output_type = str,
                   instructions = get_summary_instructions(),
                   #system_prompt = 'You are helping me test research questions', # context about the game and scenario here #'Role: You are a prisoner stuck inside of the prisoner dilemma scenario, where you and your friend, prisoner 2, have committed a crime...')
                   #deps_type = Settings,
                   name = "Summarizer",
                   toolsets = [toolset],
                   model_settings= ModelSettings(
                       max_tokens = 4000, #limiting the amount of tokens.
                       temperature= 1.0, #we want a very creative response for this research
                       timeout= 60 #60 seconds for timeout
                   ),
                   retries = 1,
                   output_retries = 1,
                   instrument = True #set to true for logfire
)