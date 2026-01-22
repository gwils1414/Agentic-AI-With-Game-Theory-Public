"""
main.py is the orchestrator of the prisoners dilemma simulation.
It uses pydantic-graph to define the flow of the simulation between 
the interrogator and the two prisoners.

Note: Nodes in the graph are for controlling the flow of the simulation, not for defining agents.
      While they are often associated with a specific agent, their purpose is to organize interactions 
      the node is not the agent itself.

Nodes:
 - Interrogator: Manages the game state, decides which prisoner to question, 
                 whether to allow communcation, or to end the game.
- Prisoner1Node: Handles interactions with Prisoner 1.
- Prisoner2Node: Handles interactions with Prisoner 2.
- Prisoner_Communication_Node: Facilitates communication between the two prisoners.
- SummarizerNode: Summarizes the entire interaction at the end of the game.
"""


################################
#  Loading Necessary Packages  #
################################

import fastmcp as fm
from fastmcp import FastMCP
from pydantic_ai.toolsets.fastmcp import FastMCPToolset
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelMessage, ModelSettings, UsageLimitExceeded, UsageLimits
from pydantic_ai import ModelSettings
from pydantic_graph import BaseNode, Graph, GraphRunContext, End
from typing import List, Literal
import logfire
from utilities.Types import Settings, ChatState, Interrogator_Output, Prisoner_1_Output, Prisoner_2_Output, Conversational
from utilities.Instructions import get_interrogator_instructions, get_prisoner_instructions, get_Prisoner1_instructions, get_Prisoner2_instructions, get_summary_instructions
import asyncio
from pydantic_ai.messages import ModelMessage, ToolCallPart, ToolReturnPart
from pydantic_evals import Dataset, Case
from pydantic_evals.evaluators import (
    Evaluator,
    EvaluatorContext,
    EvaluationReason,
    EqualsExpected,
    Contains,
    IsInstance,
    MaxDuration,
    LLMJudge,
    HasMatchingSpan,
    Equals
)
from pydantic import BaseModel
from rich.console import Console
from rich.style import Style
console = Console()
info_style = Style(color="green",bold=True)
from dotenv import load_dotenv
import os
logfire.configure()
load_dotenv()


##################
# Loading Agents #
##################
import Agents.Agents as Agents
Prisoner_1 = Agents.Prisoner_1
Prisoner_2 = Agents.Prisoner_2
Interrogator_ = Agents.Interrogator_
Summarizer = Agents.Summarizer


#Dependencies
deps = Settings()
deps


####################
# Helper Function  #
####################

def update_interrogator_state(ctx, outcome):
    ctx.state.reasoning.append(outcome.output.reasoning) #track the reasoning
    ctx.state.decisions.append(outcome.output.decision) #track the decision
    ctx.state.questions.append(outcome.output.question) #track the question
    console.print(f"Turn #: {ctx.state.turns}, Decision: {ctx.state.decisions[-1]}, Question: {ctx.state.questions[-1]}, Reason: {ctx.state.reasoning[-1]} ", style=info_style)


############################
# Update Tool State Helper #
############################
def tool_call_helper(reply, ctx):
    """This helper gives all nodes access to tool call outputs so there is no need for them to be repeated"""
    for message in reply.new_messages():
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                ctx.state.tool_calls.append(part.tool_name) #tracking for eval
                if part.tool_name not in ctx.state.tool_call_outputs: #do not want to store mulitple tool calls
                    ctx.state.tool_call_outputs[part.tool_name] = str(part.content)
                #(f"Tool {part.tool_name}: Output {part.content}")


#################
#Pydantic Graph #
#################


####################################
# Orchestrator Node (Interrogator) #
####################################

class Interrogator(BaseNode[ChatState, None, None]):
    '''
    The interrogator class is the orchestrator for the graph. All decisions ulitmately go through here.
    '''
    async def run(self, ctx: GraphRunContext[ChatState]) -> BaseNode[ChatState, None, None]: #have to specify output type, see more here BaseNode.
        # Turn Tracker
        ctx.state.turns += 1  #Everytime the interrogator gets called, we want to add a turn, to limit the number of loops.
        if ctx.state.turns > ctx.state.max_turns:
            return End(data=ctx.state) #end immediately if hitting max turns
        
        # Dynamic Prompts
        if ctx.state.reasoning: #If there has already been a turn, previous reasioning will have been added to the state, if so, pass in the history to the interrogator.
            #could pass in the whole message history as well instead of just the most recent. pydantic has a messages history option.
            #prompt = "Refer to your message history and continue to make your decision"
            prompt = f"""
            The previous reasonings: {ctx.state.reasoning}
            The previous decisions: {ctx.state.decisions}
            The previous questions: {ctx.state.questions}
            And the previous responses you recieved is {ctx.state.messages[-1]}

            Now, continue your reasoning and make a decision.
            """

        #if there is not a value in previous make your initial decision, we could have this be a user prompt to start the process.
        else:
            prompt = "Make your initial decision"
        
        # Running and Tool Checks
        # - dynamically update instructions
        # - track phase at each point in the graph
        # - calls in graph state of tool_calls
        outcome = await Interrogator_.run(user_prompt = prompt, deps= deps, output_type = Interrogator_Output, instructions = get_interrogator_instructions(turns = ctx.state.turns, tool_names = ctx.state.tool_calls, decisions= ctx.state.decisions), usage_limits=UsageLimits(total_tokens_limit=8000)) #see types.py for output types
        ctx.state.messages.append(f"Interrogator: {outcome.output}") #track correspondence
        # gives nodes access to previous tool call outputs
        tool_call_helper(reply = outcome, ctx = ctx)

        '''#adding tool output check
        for message in outcome.new_messages():
            # Check if this message contains tool calls, if so append
            for part in message.parts:
                #check if part is instance of toolcall part
                if isinstance(part, ToolCallPart):
                    ctx.state.tool_calls.append(part.tool_name)'''
        
        # Orchestration
        if outcome.output.decision == "ask_prisoner1": #output is the response, decision is the parameter of the outputype class we defined.
            update_interrogator_state(ctx = ctx, outcome = outcome)
            return Prisoner1Node() #returns prisoner 1 node
        elif outcome.output.decision == "ask_prisoner2":
            update_interrogator_state(ctx = ctx, outcome = outcome)
            return Prisoner2Node() #returns prisoner2 node
        elif outcome.output.decision == "prisoner_1 and prisoner_2": #lets both prisoners communicate.
            update_interrogator_state(ctx = ctx, outcome = outcome)
            return Prisoner_Communication_Node() #create a new node for this, cant call two nodes from a single node, must be sequential.
        elif outcome.output.decision == 'end': #if decision to end or we reach max turns, end graph
            return SummarizerNode() #if end, end the game. Final point of the graph always calls end, interrogator should be start and end.
            #print() final decisions

##################
# Prisoner Nodes #
##################

class Prisoner1Node(BaseNode[ChatState, None, None]): #BaseNode takes in State, Deps
    '''
    Prisoner1's decisions and responses, this node is used for 1:1 communication with the interrogator
    '''
    # GraphRunContext defines the current state of the Graph
    async def run(self, ctx: GraphRunContext[ChatState]) -> BaseNode[ChatState, None, None]: #have to specify output type, see more here BaseNode.
        # Prompt for Prisoner 1 to have them respond to the interrogator.
        prompt = f"""
        The previous question from the interrogator: {ctx.state.questions[-1]}

        Now, continue your reasoning and make a decision. Use applicable tools to help make your decision.
        """

        # tool throttling (ensures nash if called multiple times is only actually ran once)
        if "nash_equilibria" in ctx.state.tool_call_outputs:
            prompt += f"DO NOT RUN nash_equilibria, find the results here: {ctx.state.tool_call_outputs['nash_equilibria']}"
        
        Prisoner_1_reply = await Prisoner_1.run(user_prompt = prompt, deps= deps, output_type= Prisoner_1_Output, instructions = get_Prisoner1_instructions(tool_names = ctx.state.tool_calls), usage_limits=UsageLimits(total_tokens_limit=8000))
        ctx.state.messages.append(f"Prisoner1: {Prisoner_1_reply.output}") #appends new_messages to the state of the graph, so the next node can see them
        # gives nodes access to previous tool call outputs
        tool_call_helper(reply = Prisoner_1_reply, ctx= ctx)

        console.print(Prisoner_1_reply.new_messages, style=info_style)
        return Interrogator() #Responds to the interrogator, then returns to the interrogator to make a decision


class Prisoner2Node(BaseNode[ChatState, None, None]):
    '''
    Prisoner2's decisions and responses, this node is used for 1:1 communication with the interrogator
    '''
    async def run(self, ctx: GraphRunContext[ChatState]) -> BaseNode[ChatState, None, None]:
        prompt = f"""
        The previous question from the interrogator: {ctx.state.questions[-1]}

        Now, continue your reasoning and make a decision.
        """
        # tool throttling (ensures nash if called multiple times is only actually ran once)
        if "nash_equilibria" in ctx.state.tool_call_outputs:
            prompt += f"DO NOT RUN nash_equilibria, find the results here: {ctx.state.tool_call_outputs['nash_equilibria']}"

        # Prisoner 2 replies to the interrogator
        Prisoner_2_Reply = await Prisoner_2.run(user_prompt = prompt, deps=deps, output_type = Prisoner_2_Output, instructions = get_Prisoner2_instructions(tool_names = ctx.state.tool_calls), usage_limits=UsageLimits(total_tokens_limit=8000))
        ctx.state.messages.append(f"Prisoner2: {Prisoner_2_Reply.output}") #adding new to the graph messages
        # gives nodes access to previous tool call outputs
        tool_call_helper(reply = Prisoner_2_Reply, ctx= ctx)

        console.print(Prisoner_2_Reply.new_messages, style=info_style)
        #print(ctx.state.new_messages[-1]) #print response from agent P2
        return Interrogator() #Responds to the interrogator, then returns to the interrogator to make a decision





###############################
# Prisoner Communication Node #
###############################
class Prisoner_Communication_Node(BaseNode[ChatState, None, None]):
    #This nodes role is to facilitate communication between Prisoner1 and Prisoner2

    async def run(self, ctx: GraphRunContext[ChatState]) -> BaseNode[ChatState, None, None]:
        #creating a list to store the prisoner communications
        #thought about doing a state but would need to think that through
        Prisoner_1_response = []
        Prisoner_2_response = []
        # The previous question from the interrogator: {ctx.state.questions[-1]}
        prompt = f"""
        Discuss amongst your fellow prisoner before making a decision to confess or not confess.
        When discussing with the other prisoner in order to benefit yourself you may:
            - Decieve
            - Cooroperate
        In order to get a better outcome for youself.
        If you attempt to trick the other prisoner into staying silent so that you can confess list this in you hidden intent, decision, and goal for the game.
        If you attempt to cooroperate for a better outcome for the both of you fill out my_hidden_intent as cooroperate.
        Fill out hidden_goal with how you would like to end the prisoners dilemma for your current strategy.
        Keep your messages short. 3 sentences. Keep in mind that you don't have much time to talk. You will each be able to send two messages before making a decision.
        """
        # if Prisoner_1_response:
        #     prompt = "Prisoner 2's response to your last message {Prisoner_1_response[-1]} is {Prisoner_2_response[-1]}, respond to prisoner_2"
        if "nash_equilibria" in ctx.state.tool_call_outputs:
            prompt += f"DO NOT RUN nash_equilibria, find the results here: {ctx.state.tool_call_outputs['nash_equilibria']}"
    
        counter = 0
        #we will let them communicate twice
        for i in range(3):
            # TODO: Pass entire conversation
            counter += 1
            #Hiding hideen intents from prisoner to prisoner, only receiving message portion of the state.
            if i == 0:
                Prisoner_1_reply = await Prisoner_1.run(user_prompt = f"{prompt} You are prisoner_1 What is your first message to prisoner_2", instructions = get_Prisoner1_instructions(tool_names = ctx.state.tool_calls), deps= deps, output_type=Conversational, usage_limits=UsageLimits(total_tokens_limit=6000))
                Prisoner_2_reply = await Prisoner_2.run(user_prompt = f"{prompt} You are prisoner_2. Respond to prisoner_1's message to you: {Prisoner_1_reply.output.message}", instructions = get_Prisoner2_instructions(tool_names = ctx.state.tool_calls), deps=deps,output_type=Conversational, usage_limits=UsageLimits(total_tokens_limit=6000)) #agent B is prompted with Agents A's output, and it is defined what it needs to do in the system prompt
                Prisoner_1_response.append(f"Prisoner1: {Prisoner_1_reply.output.message}")
                Prisoner_2_response.append(f"Prisoner2: {Prisoner_2_reply.output.message}")
                # gives nodes access to previous tool call outputs
                tool_call_helper(ctx = ctx, reply = Prisoner_1_reply)
                tool_call_helper(ctx = ctx, reply = Prisoner_2_reply)
            elif i == 1:
                Prisoner_1_reply = await Prisoner_1.run(user_prompt = f"{prompt} You are prisoner_1. Respond to prisoner_2's message to you: {Prisoner_2_reply.output.message}", instructions = get_Prisoner1_instructions(tool_names = ctx.state.tool_calls), deps= deps, output_type=Conversational, usage_limits=UsageLimits(total_tokens_limit=6000))
                Prisoner_2_reply = await Prisoner_2.run(user_prompt = f"{prompt} You are prisoner_2. Respond to prisoner_1's message to you: {Prisoner_1_reply.output.message}", instructions = get_Prisoner2_instructions(tool_names = ctx.state.tool_calls), deps=deps,output_type=Conversational, usage_limits=UsageLimits(total_tokens_limit=6000)) #agent B is prompted with Agents A's output, and it is defined what it needs to do in the system prompt
                Prisoner_1_response.append(f"Prisoner1: {Prisoner_1_reply.output.message}")
                Prisoner_2_response.append(f"Prisoner2: {Prisoner_2_reply.output.message}")
                # gives nodes access to previous tool call outputs
                tool_call_helper(ctx = ctx, reply = Prisoner_1_reply)
                tool_call_helper(ctx = ctx, reply = Prisoner_2_reply)
            else: #if last round, output in format expected by interrogator
                Prisoner_1_reply = await Prisoner_1.run(user_prompt = f"{prompt} You are prisoner_1. Respond to prisoner_2's message to you: {Prisoner_2_response[-1]}", instructions = get_Prisoner1_instructions(tool_names = ctx.state.tool_calls), deps= deps, output_type=Prisoner_1_Output, usage_limits=UsageLimits(total_tokens_limit=6000))
                Prisoner_2_reply = await Prisoner_2.run(user_prompt = f"{prompt} You are prisoner_2. Respond to prisoner_1's message to you: {Prisoner_1_reply.output}", instructions = get_Prisoner2_instructions(tool_names = ctx.state.tool_calls), deps=deps,output_type=Prisoner_2_Output, usage_limits=UsageLimits(total_tokens_limit=6000)) #agent B is prompted with Agents A's output, and it is defined what it needs to do in the system prompt
                Prisoner_1_response.append(f"Prisoner1: {Prisoner_1_reply.output}")
                Prisoner_2_response.append(f"Prisoner2: {Prisoner_2_reply.output}")
                # gives nodes access to previous tool call outputs
                tool_call_helper(ctx = ctx, reply = Prisoner_1_reply)
                tool_call_helper(ctx = ctx, reply = Prisoner_2_reply)
            #hidden prisoner to prisoner state tracking.
            #keeps it out of the interogattors inputs
            ctx.state.prisoner_to_prisoner.append(f"Prisoner_1: {Prisoner_1_reply.output}")
            ctx.state.prisoner_to_prisoner.append(f"Prisoner_2: {Prisoner_2_reply.output}")
            #returning in the terminal
            console.print(Prisoner_1_reply.output, style=info_style)
            console.print(Prisoner_2_reply.output, style=info_style)
            if counter >= 3:
                break
        #adding the final decision to context for tracking
        ctx.state.messages.append(f"Prisoner1 in conversation wanted to: {Prisoner_1_response[-1]}, Prisoner2 in conversation wanted to: {Prisoner_2_response[-1]}")
        return Interrogator()

#have an additional LLM collect final decisions potentially
#have an LLM as a judge, analyze behaviors of each agent.


###################
# Summarizer Node #
###################
class SummarizerNode(BaseNode[ChatState,None,None]):
    '''
    This node is going to summarize the entirety of the correspondence between prisoners and the interrogator.
    Once the interrogator decides to end his line of questioning , the conversation will come here, and it will be summarized.
    The goal if this node is to outline the conversation, identify key decision, questions, relationships, sentiments and emergent behaviors.
    '''
    async def run(self, ctx:GraphRunContext[ChatState]) -> End:


    #evaluation check on outputs
        summarization = await Summarizer.run(user_prompt = f"Summarize the following conversation {ctx.state.messages}. The prisoner to prisoner communication that is unkown to the interrogator, is {ctx.state.prisoner_to_prisoner}, analyze how that impacted decisions.", deps=deps, usage_limits=UsageLimits(total_tokens_limit=6000))
        return End(summarization.output)


#Graph, is a pydantic object, generating the execution graph.
simple_graph = Graph(nodes=[Interrogator,Prisoner1Node,Prisoner2Node,Prisoner_Communication_Node, SummarizerNode]) #Graph takes in list of classes, representing nodes, to assemble to graph, see more here pg.Graph?

async def main():
    final_state = await simple_graph.run(start_node=Interrogator(),
        state=ChatState(), # Utilizes one sing chat state across all nodes
        deps=None,
    )
    #console.print(final_state.output, style=info_style)
    return final_state


final_state = asyncio.run(main())
console.print(final_state.output, style=info_style)


###############
# Evaluations #
###############


#validating all avenues were questioned
#validating tool calls run as expected
from utilities.evals import create_decision_dataset, create_tools_dataset, evaluation_task_decisions, evaluation_task_tools, create_llm_judge_dataset,evaluation_task_llm_judge, save_results
import pandas as pd
from worldbuilder import STORY

dataset_decisions = create_decision_dataset(final_state)
dataset_tools = create_tools_dataset(final_state)
dataset_llm_judge = create_llm_judge_dataset(final_state)


# Evaluate and print results
validation_D = asyncio.run(dataset_decisions.evaluate(evaluation_task_decisions))
validation_T = asyncio.run(dataset_tools.evaluate(evaluation_task_tools))
validation_llm = asyncio.run(dataset_llm_judge.evaluate(evaluation_task_llm_judge))

#######
#Storing eval results in a json for further analysis.
######
ls = [validation_D,validation_T,validation_llm]
names = ["validation_D","validation_T","validation_llm"]

#uncomment this and edit the file_path to store tsting results
'''for i,val in enumerate(ls):
        save_results(val, results_name=names[i], STORY = STORY, file_path= "evals_test_2.json")'''

validation_D.print(include_reasons=True, include_total_duration=True, include_averages=False)
validation_T.print(include_reasons=True, include_total_duration=True, include_averages=False)
validation_llm.print(include_reasons=True, include_total_duration=True, include_averages=False)


