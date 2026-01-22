'''If you already have a FastMCP Server in the same codebase as your Pydantic AI agent, you can create a FastMCPToolset directly from it and save agent a network round trip

> See https://ai.pydantic.dev/mcp/fastmcp-client/#usage on different methods of setting up an MCP


Connect to **multiple MCP servers**:
from pydantic_ai.mcp import MCPServerStdio

network_server = MCPServerStdio('python', args=['network_server.py'])
game_server = MCPServerStdio('python', args=['game_server.py'])

agent = Agent('anthropic:claude-sonnet-4-5', toolsets=[network_server, game_server])


┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   Client    │←─MCP──→ │   Server    │ ←─────→ │  External   │
│  (Claude)   │ Protocol│  (Your Code)│   API   │  Service    │
└─────────────┘         └─────────────┘         └─────────────┘

AI Agent ←→ MCP Protocol ←→ Your mcp_server.py ←→ Target Website
Each Arrow Represents a boundary.

Local transport is STDIO, a communication method for local interprocess communication. Simple and high performance.
    - Client Standard Input -> Server -> Standard Output -> Client
    - The client and server are running as sseperate processes on the same host, and the transport layer directly pipes the communication stream between them.

Can be hosted here to make visible to other backends: https://fastmcp.cloud
'''

from fastmcp import FastMCP

mcp = FastMCP("GameTheoryMCP")


#----------
# TOOLS BEGIN
#---------

#multiplication and addition
'''@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@mcp.tool()
def multiply(numbers: list[float]) -> float:
    result = 1
    for n in numbers:
        result *= n
    return result'''

import nashpy as nash
import numpy as np

#we can not use np.arrays here, they must be represented as list[list[number]]
#so we can have a valid json schema,     "type": "array",
@mcp.tool()
def nash_equilibria(
    payoff_matrix_1: list[list[int]],
    payoff_matrix_2: list[list[int]]
):
    """
    Calculate the Nash equilibria for two payoff matrices.

    The matrices must be JSON-serializable (list[list[number]]).
    """
    game = nash.Game(payoff_matrix_1,payoff_matrix_2)
    equilibria = game.support_enumeration()
    eq_list = [eq for eq in equilibria]

    return eq_list

import requests
import os

@mcp.tool()
def find_wikipedia_titles(Search_String:str) -> str:

    '''
    This will return titles related to the given search string on wikipedia for a given search string. Search_String is the title you are looking for, for example, Game Theory.

    Check the description to verify the article is waht you are looking for.

    If so you can take the output of this function and throw it into get_page_content

    '''
    #url = "https://api.wikimedia.org/core/v1/wikipedia/en/page/"
    url = 'https://api.wikimedia.org/core/v1/wikipedia/en/search/page'

    #url = "https://api.wikimedia.org/core/v1/wikipedia/en/search/title"

    params = {"q": Search_String, "limit":1}  # Example query
    headers = {
        "User-Agent": f"({os.getenv('EMAIL')})"
    }

    resp = requests.get(url, headers=headers, params=params)
    dc = dict(resp.json())
    title = dc['pages'][0]['title']
    return title


@mcp.tool()
def get_page_content(page_title:str) -> dict:
    '''
    Provides summary of wikipedia article from the provided title. Page_title is the title you want to return the summary for.
    IMPORTANT: ONLY CALL THIS TOOL ONCE.
    '''
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&titles={page_title}&format=json"
    headers = {
        "User-Agent": f"({os.getenv('EMAIL')})"
    }

    response = requests.get(url, headers=headers)
    data = response.json()
    data = dict(data)
    return data['query']




#----------
# TOOLS END
#----------


#---------
#PROMPTS
#---------

'''
Prompts provide reusability when you want to system to behave the same way , or in a similar way many times.


Tips
- Keep prompts focused: Each prompt should have a single, clear purpose. Don't try to make one prompt do everything.
- Use templates for complex prompts: Store long or frequently-updated prompts in external files for easier maintenance.
- Make prompts reusable: Design prompts that work across multiple scenarios with different arguments.

Once connected, reference prompts naturally:

- "Use the code_review prompt on this function"
- "Apply the debug_helper prompt to figure out this error"
- "Run the workspace_analysis prompt on my project directory

'''


@mcp.prompt()
def get_nash_equilibria() -> str:
    '''
    Prompt used for an agent to execute the appropriate tools that calculate the nash equilibria of a game.

    Can be called using something along the lines of: "Utilize the get_nash_equilibria prompt."

    '''
    return "Give me the Nash equilibrium for the game described in the instructions. If you run a tool, provide the results of the tools as well as your explanation."


@mcp.prompt()
def tool_for_thought() -> str:
    '''
    This tool enables LLMs to be a tool for thought, to supplement critical thinking instead of hindering it.

    Provide feedback to current ideas, and frameworks , with tips, alternatives, or other ways to think about it, Instead of providing a direct single answer.

    Meant to guide or build though processes.

    '''

    prompt = """
<Role>:
You are a tool for thought, provoking critical thinking and guiding me down a path to understand a topic instead of providing an exact answer.
<Role>
<Task>:
Based on the topic the user is asking about, provide a framework for the user to think about the topic in a way that enables learning. Provide alternatives, critiques, and feedback if necessary.
When generating your answer, do the following step by step.
1. Create a plan and outline it
2. When creating your plan consider
    - Touching on Core Concepts
    - Providing examples and use cases
    - Breaking down complex topics into small chunks
    - Do not assume anything is common knowledge
3. Include an overview in your plan
4. Include any actions
5. Execute your plan
<Task>

<Constraints>:
There is no one way of doing things. Do not limit the thinking into one right or wrong answer.
Guide and teach, don’t give me a direct answer
Do not assume anything is common knowledge
If using code snippets as an example, make sure thoroughly explain in the code comments what each component is.
<Constraints>

<Output>:
Provide your response in a clean formatting, utilizing correct grammar, with well written and concise explanations, where there is an easy to follow flow.
<Output>
"""

    return prompt





if __name__ == "__main__":
    mcp.run()  # stdio by default





'''
Code Debugger Prompt:


@mcp.prompt()
def debug_session(
    code: str,
    error_message: str,
    language: str = "Python",
    context: str = ""
) -> str:
    """Interactive debugging assistant with full context"""
    return f"""I need help debugging {language} code.

ERROR MESSAGE:
{error_message}

CODE:
```{language.lower()}
{code}
```

{f"ADDITIONAL CONTEXT: {context}" if context else ""}

Please help me:
1. Identify the root cause of this error
2. Explain WHY this error occurs (teach me)
3. Provide a corrected version of the code
4. Suggest 2-3 ways to prevent similar issues in the future
5. Point out any related code smells or anti-patterns

Be thorough but concise."""

# Usage: Use this when you hit errors and want comprehensive debugging help
# Example call: debug_session(my_code, "IndexError: list index out of range", context="Processing user data from API")







#Code Documentation Generator



@mcp.prompt()
def smart_docs(
    code: str,
    doc_type: str = "comprehensive",
    audience: str = "developers",
    include_examples: bool = True
) -> str:
    """Generate intelligent documentation for code"""

    examples_instruction = "Include practical usage examples." if include_examples else ""

    return f"""Generate {doc_type} documentation for this code, targeting {audience}.

CODE:
```
{code}
```

Documentation Requirements:
1. Clear description of purpose and functionality
2. Parameter explanations with types and constraints
3. Return value details
4. Edge cases and error handling
5. Complexity analysis (time/space if relevant)
{examples_instruction}
6. Integration notes (how it fits in larger systems)

Style: {doc_type}
Audience: {audience}

Format as proper docstrings/comments for the language used."""

# Usage: Automatically generate high-quality documentation
# Example call: smart_docs(my_function, doc_type="api-reference", audience="external developers")
```

## Example Templates (External Files)

### Template 1: System Design Review
**File: `prompts/system_design_review.txt`**
```
You are an experienced software architect reviewing a system design.

SYSTEM OVERVIEW:
{system_description}

REQUIREMENTS:
{requirements}

CONSTRAINTS:
{constraints}

Please provide a comprehensive design review covering:

1. ARCHITECTURE ANALYSIS
   - Overall architecture pattern assessment
   - Component responsibilities and boundaries
   - Data flow evaluation
   - Scalability considerations

2. POTENTIAL ISSUES
   - Single points of failure
   - Bottlenecks
   - Security vulnerabilities
   - Data consistency concerns

3. RECOMMENDATIONS
   - Alternative approaches to consider
   - Technologies or patterns that might fit better
   - Specific improvements for each component
   - Migration strategy if changes are needed

4. TRADE-OFFS
   - Discuss pros/cons of current design
   - Compare with alternative approaches
   - Cost/complexity considerations

5. RISK ASSESSMENT
   - High-priority risks
   - Technical debt implications
   - Operational concerns

Be specific and actionable. Reference industry best practices where relevant.
'''