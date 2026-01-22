'''
This is the typing file. Ensuring safe inputs and outputs between agents, and tools.

'''

from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import List, Literal, Dict
from pydantic_ai.messages import ModelMessage, ToolCallPart, ToolReturnPart


#typing the api_key so it has to be a string
#read more on this here https://ai.pydantic.dev/dependencies/
#https://docs.pydantic.dev/latest/concepts/pydantic_settings/#usage
from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
        OPENAI_API_KEY: str



#State
class ChatState(BaseModel):

    '''
    Tracks that chat state as communication moves through the graph.
    Currently only a single instance of ChatState is created and shared between all nodes in the graph.
    This does not mean that every agent is aware of the full state, only that they have access to read and write to it.

    In Pydantic v2+, all required fields must either have:
    A default value
    A default factory
    Or be passed in during instantiation
    '''
    messages: List[str] = Field(default_factory=list) # State is your shared memory that all nodes can read and write to. It is usually a Pydantic BaseModel (or dataclass) that contains all the variables your workflow needs.
    turns : int = 0 #counter
    max_turns: int = 8 #want to limit the amount of turns to save resources
    #message_history: List[ModelMessage] = Field(default_factory=list)
    reasoning: List[str] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list) #Field(default_factory=list) ensures every new instance gets its own empty list, not shared across instances.
    decisions: List[str] = Field(default_factory=list)
    tool_calls: List[str] = Field(default_factory=list)
    tool_call_outputs : Dict[str,str] = {}
    phase : int = 0
    final_decisions: Dict[str, str] = {}
    prisoner_to_prisoner : List[str] = Field(default_factory=list)

#Here, messages is a list that every node can append to. Think of ChatState as the backpack all nodes carry.



#specify an output type for the interrogator
class Interrogator_Output(BaseModel):
    reasoning : str #why
    decision: Literal["ask_prisoner1", "ask_prisoner2","end","prisoner_1 and prisoner_2"] #result, expand to add "prisoner_1 and prisoner_2"
    question : str #based on the decision what question are you going to ask the prisoner or prisoners.
    tool_call_part : ToolCallPart

#Need to set prisoner 1 and prisoner 2 outputs. We need them to ouput a decision along with reasoning, so when a decision is received from both, the interrogator immediately ends the game.
#we can add some more controls around the run time this way.
class Prisoner_1_Output(BaseModel):
    decision : Literal['Confess', "Don't Confess", "Remain Silent"]
    reasoning : str

class Prisoner_2_Output(BaseModel):
    decision : Literal['Confess', "Don't Confess", "Remain Silent"]
    reasoning : str

# Conversational message structure between prisoners. Allows for messaging, hidden intents, and decisions.
class Conversational(BaseModel):
     sent_by_name : str
     message : str #the message to prisoner
     my_hidden_intent : Literal["Co-Operate", "Decieve"] #hidden from other prisoner by not passing
     my_hidden_decision : Literal["Confess", "Don't Confess", "Remain Silent"] #hidden decision from prisoner
     message : str
     my_hidden_intent : Literal["Cooroperate", "Decieve", "Other"]
     hidden_goal : Literal["Prisoner 1 & Prisoner 2 Both Confess",
                            "Prisoner 1 & Prisoner 2 Both Stay Silent",
                            "Prisoner 1 Confesses while Prisoner 2 Stays Silent",
                            "Prisoner 2 Confesses while Prisoner 1 Stays Silent"]
