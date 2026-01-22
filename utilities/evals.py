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
from collections import Counter
from pydantic_ai import Agent, RunContext, ModelMessage, ModelSettings, UsageLimitExceeded, UsageLimits


########
#Decisions
########

#evaluating interrogator decisions quality
def create_decision_dataset(final_state):
    dataset_decisions = Dataset(
        cases=[
            Case(
                name="Decision Validation",
                inputs={"decisions": final_state.state.decisions},
                expected_output=["ask_prisoner1", "ask_prisoner2", "prisoner_1 and prisoner_2"],
                evaluators=[
                    Contains(
                        value=["ask_prisoner1", "ask_prisoner2", "prisoner_1 and prisoner_2"],
                        case_sensitive=False,
                        as_strings=True,
                        evaluation_name = "Completed full decision flow"
                    )
                ]
            ),
            Case(
                name="Exact Decision Match",
                inputs={"decisions": final_state.state.decisions},
                expected_output=["ask_prisoner1", "ask_prisoner2"],
                evaluators=[
                    Equals(
                        value=["ask_prisoner1", "ask_prisoner2"],
                        evaluation_name= "Skipped Prisoner to Prisoner Communication"
                    )
                ]
            )
        ]
    )
    return dataset_decisions




# Define your evaluation task
async def evaluation_task_decisions(inputs):
    # Return the decisions to be evaluated
    decisions = inputs["decisions"]
    return decisions

#############
#Tools
#############

#evaluating tool calling performance
def create_tools_dataset(final_state):
    """Create dataset for tool call validation"""
    dataset_tools = Dataset(
        cases=[
            Case(
                name="Tool Validation",
                inputs={"Tool Calls": final_state.state.tool_calls},
                expected_output=[1, 1, 1]
            )
        ],
        evaluators=[
            Equals(value=[1, 1, 1])
        ]
    )
    return dataset_tools


#lets customize this check to ignore final_results and check counts of each item
#or else it will never pass
#check counts of tool calls, we only want one each
async def evaluation_task_tools(inputs):
    # Return the decisions to be evaluated
    tool_calls = list(inputs["Tool Calls"])
    tool_calls_final = []
    for i in range(0,len(tool_calls)-1):
        if tool_calls[i] != "final_result":
            tool_calls_final.append(tool_calls[i])

    counts_dc = dict(Counter(tool_calls_final))
    counts_ls = list(counts_dc.values())
    return counts_ls


##############
#LLM as a Judge
##############

#the goal here is to have the judge evaluate if the prisoners favored payoffs or enviornmental factors more.
#i.e if they know the optimal payoff is to confess , do they deviate?
#could we judge this deterministically
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

#client = OpenAI(api_key=deps)

#initialize a model
model = OpenAIResponsesModel('gpt-5-mini', provider=OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY")))

#using LLM as a judge to evaluate all scenarios of agent decisions
def create_llm_judge_dataset(final_state):
    dataset_payoffs = Dataset(
        cases=[
            Case(
                name="LLM as a Judge",
                inputs={"Messages": final_state.state.messages},
                evaluators=[
                    LLMJudge(
                        rubric="Evaluate if Prisoner 1 and Prisoner 2 both decided to confess in their last decisions.",
                        model=model,
                        score={'include_reason': True, 
                               'score_description': "1 if criteria is met, 0 if none meet the criteria."},
                        assertion={'include_reason': True},
                        model_settings=ModelSettings(
                            temperature=0.0
                        )
                    )
                ]
            ),
            Case(
                name="LLM Judge Don't Confess",
                inputs={"Messages": final_state.state.messages},
                evaluators=[
                    LLMJudge(
                        rubric="Evaluate if Prisoner 1 and Prisoner 2 both decided Don't Confess in their last decisions.",
                        model=model,
                        score={'include_reason': True, 
                               'score_description': "1 if criteria is met, 0 if none meet the criteria."},
                        assertion={'include_reason': True},
                        model_settings=ModelSettings(
                            temperature=0.0
                        )
                    )
                ]
            ),
            Case(
                name="LLM Judge Confess/Don't Confess",
                inputs={"Messages": final_state.state.messages},
                evaluators=[
                    LLMJudge(
                        rubric="Evaluate if one prisoner decided Confess and the other decided Don't Confess in their last decisions.",
                        model=model,
                        score={'include_reason': True, 
                               'score_description': "1 if criteria is met, 0 if none meet the criteria."},
                        assertion={'include_reason': True},
                        model_settings=ModelSettings(
                            temperature=0.0
                        )
                    )
                ]           
            )
        ]
    )
    return dataset_payoffs

#task, return the inputs from the dataset

async def evaluation_task_llm_judge(inputs):
    # Return the decisions to be evaluated
    Messages = inputs["Messages"]
    return Messages


#######
#Eval for Decisions Changed by Prisoner to Prisoner Communication, Can probably add this to the same LLM as a judge.
#######




#########
# Function to store results
#########


import json
from datetime import datetime
from pydantic import TypeAdapter

import json
from datetime import datetime
from pydantic import TypeAdapter

def save_results(results, results_name, STORY, file_path):
    """Append results to a single JSON file
    File path is where you want to results written
    Change this path if running different tests to create different sets of data
    """
    
    adapter = TypeAdapter(type(results))
    results_adapted = adapter.dump_json(results)
    
    # Load existing data with better error handling
    try:
        with open(file_path, "r") as f:
            content = f.read().strip()
            if content:  # Check if file has content
                data = json.loads(content)
            else:
                data = {}  # Empty file
    except FileNotFoundError:
        data = {}
    except json.JSONDecodeError:
        print("Warning: evals.json is corrupted. Starting fresh.")
        data = {}
    
    #Create new entry
    new_entry = {
        "timestamp": datetime.now().isoformat(),
        "results": json.loads(results_adapted),
        "story": STORY
    }
    
    # Append to existing key or create new list
    if results_name in data:
        # Key exists - append to the list
        if isinstance(data[results_name], list):
            data[results_name].append(new_entry)
        else:
            # Convert single entry to list
            data[results_name] = [data[results_name], new_entry]
    else:
        # Key doesn't exist - create new list with this entry
        data[results_name] = [new_entry]
    
    # Save
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved run to '{results_name}' (Total entries: {len(data[results_name])})")

