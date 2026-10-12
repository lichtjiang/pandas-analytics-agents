import os
import sys
from dataclasses import dataclass
from typing import Literal, Required, TypeAlias, TypedDict

import duckdb
import pandas as pd
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

#from visualization_specialist_langchain import visualization_graph
from pandas_analytics_agents.subgraph.langchain.visualization_specialist_langchain import (
    visualization_graph,
)
from pandas_analytics_agents.utils.dataset import load_covid_dataset
from pandas_analytics_agents.utils.duckdb import (
    connect_and_register_df,
    disconnect_and_unregister_df,
)

load_dotenv()

@dataclass
class AnalyticsContext:
    duckdb_conn: duckdb.DuckDBPyConnection
    covid_df: pd.DataFrame


# Tool "A", a specialized Python tool
@tool
def find_country_with_most_recovered(runtime: ToolRuntime) -> str:
    """
    Find the country that reported the largest number of recovered
    COVID-19 cases.
    
    Use this tool specifically for questions asking which country
    has the largest number of recovered cases.
    """
    covid_df = runtime.context.covid_df

    result = (
        covid_df
        .loc[covid_df["recovered"].idxmax()]
    )
    return (
        f"Country: {result['country']}\n"
        f"Recovered cases: {int(result['recovered'])}"
    )

# Tool "B", a specialized SQL tool that returns a table of results
@tool
def query_covid_table(sql: str, runtime: ToolRuntime) -> str:
    """
    Execute a read-only SQL query against the covid_19 table.
    This tool is intended for queries that return a table with
    multiple records.
    Available table:
    covid_19(
        country STRING,
        deaths BIGINT,
        recovered BIGINT
    )
    Only SELECT statements are allowed.
    """
    duckdb_conn = runtime.context.duckdb_conn
    sql_clean = sql.strip().lower()
    if not sql_clean.startswith("select"):
        return "ERROR: Only SELECT queries are allowed."
    try:
        result_df = duckdb_conn.execute(sql).df()
        if result_df.empty:
            return "The query returned no rows."
        # The DataFrame exists here as a real Pandas DataFrame.
        # Serialize it before returning it across the agent boundary.
        return result_df.to_json(
            orient="records",
            date_format="iso",
        )
    except Exception as e:
        return f"ERROR executing SQL: {e}"

# Tool "C", a specialized SQL tool that returns a single aggregate value
@tool
def aggregate_covid_data(sql: str, runtime: ToolRuntime) -> str:
    """
    Execute a read-only SQL aggregation against the covid_19 table.
    The SQL should return exactly one row and one value.
    Examples:
        SELECT MAX(deaths) FROM covid_19
        SELECT AVG(deaths) FROM covid_19
        SELECT SUM(recovered) FROM covid_19
    Only SELECT statements are allowed.
    """
    duckdb_conn = runtime.context.duckdb_conn

    sql_clean = sql.strip().lower()
    if not sql_clean.startswith("select"):
        return "ERROR: Only SELECT statements are allowed."
    try:
        result_df = duckdb_conn.execute(sql).df()
        if len(result_df) !=1 or len(result_df.columns) != 1:
            return (
                "ERROR: This tool expects SQL that returns "
                "exactly one row and one column."
            )
        value = result_df.iloc[0, 0]
        return str(value)
    except Exception as e:
        return f"ERROR executing aggregation SQL: {e}"

# Get the brain: LLM, in this case, a ChatOpenAI instance connected to a deployed model on Azure Foundry
MODEL_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME_FOUNDRYPOC", "gpt-5-nano")
OPENAI_ENDPOINT = os.environ.get("LANGCHAIN_AZURE_OPENAI_ENDPOINT_FOUNDRYPOC")
_AZURE_AI_SCOPE = "https://ai.azure.com/.default"


def build_foundry_chat_model() -> ChatOpenAI:
    """Builds a ChatOpenAI instance connected to a Foundry project."""
    credential = DefaultAzureCredential()

    # Get the base URL from the client and a token provider for authentication
    token_provider = get_bearer_token_provider(credential, _AZURE_AI_SCOPE)

    return ChatOpenAI(
        model=MODEL_DEPLOYMENT_NAME,
        base_url=OPENAI_ENDPOINT, #make sure there is no trailing "responses"
        # Two ways of obtaining api_key as follows
        api_key=token_provider, #preferred way for Foundy deployed model
        # api_key=API_KEY,
        # w/o the following argument, it defaults to chat completion api
        use_responses_api=True,
    )

# The brain, our LLM model
llm = build_foundry_chat_model()

# Three specialist agents
# The 4th specialist, code specialist, which is more complex, not created by calling create_agent,
# will be created later, after the code prompt is defined

# recovered_agent: answers questions about the country with the most recovered cases
recovered_agent = create_agent(
    model=llm,
    tools=[find_country_with_most_recovered],
    system_prompt="""
    You are the COVID recovered-cases specialist.
    Your job is to answer the user's question using the 
    find_country_with_most_recovered tool.
    Call the tool to obtain the factual answer.
    Do not calculate the answer yourself.
    After receiving the tool result, explain the answer
    concisely.
    """,
        name="recovered_specialist",
    )

# table_agent: answers questions that require returning multiple records from the covid_19 dataset
table_agent = create_agent(
    model=llm,
    tools=[query_covid_table],
    system_prompt="""
    You are the COVID table-analysis specialist.
    You answer questions that require returning multiple
    records from the covid_19 dataset.
    Generate an appropriate SELECT SQL query and call
    query_covid_table.
    Available table:
    covid_19(
        country STRING,
        deaths BIGINT,
        recovered BIGINT
    )
    Only generate SELECT statements.
    After receiving the table result, summarize it for 
    the user.
    """,
        name="table_specialist",
    )

# aggregate_agent: answers questions that ask for a single aggregate value
aggregate_agent = create_agent(
    model=llm,
    tools=[aggregate_covid_data],
    system_prompt="""
    You are the COVID aggregation specialist.
    You answer questions that require a single aggregate
     value or a single winning value.
    Generate a SELECT query that returns exactly one row 
    and exactly one column.
    Available table:
    covid_19(
        country STRING,
        deaths BIGINT,
        recovered BIGINT
    )
    Only generate SELECT statements.
    Use aggregate_covid_data to execute the query.
    After receiving the reslut, explain it to the user.
    """,
        name="aggregate_specialist",
    )

RouteLiteral: TypeAlias = Literal[  # noqa: UP040
    "aggregate",
    "code",
    "recovered",
    "table",
    "visualization",
]

# Graph state type definition
class WorkflowState(TypedDict, total=False):
    user_question: Required[str]
    route: Required[RouteLiteral]
    answer: str
    generated_code: str
    feedback: str
    attempt: int

    # Visualization
    visualization_sql: str
    visualization_spec: dict #VisualizationSpec
    visualization_path: str
    visualization_error: str

# Router agent
# Router agent's structured output type definition
class RouteDecision(BaseModel):
    route: RouteLiteral = Field(
        description="The specialist that should handle the question."
    )

# Here we wrap the LLM with structured output to ensure it returns a valid route decision
# For now, we use the same llm instance, but in a real application, youwe might want to 
# use a different model or configuration for the router.
router_model = llm.with_structured_output(RouteDecision)

ROUTER_PROMPT = """
You are the triage router for a COVID-19 data analysis system.

Choose exactly one specialist.

1. recovered
   Use when the user asks:
   "Which country reported the largest number of recovered cases?"

2. table
   Use when the user wants multiple records, rows, lists,
   filtering, sorting, grouping results, or a table.

3. aggregate
   Use when the user asks for a single aggregate/winner result,
   especially questions such as:
   "Which country has reported the largest number of deaths?"

4. code
   Use for all other questions.

5. visualization
   Use when the user asks to:
    - create a chart
    - plot data
    - visualize data
    - show a graph
    - show a trend visually
    - compare data visually

Important:
- Do not answer the user's question.
- Only classify the request.
- If the user explicitly asks for a visualization, choose visualization even if SQL analysis is also required internally.
- Return exactly one route.
"""


def router_node(state: WorkflowState):
    decision = router_model.invoke(
        [
            ("system", ROUTER_PROMPT),
            ("human", state["user_question"]),
        ]
    )
    return {
        "route": decision.route,
    }

# The 4th specialist, code specialist, is more complex and 
# requires a loop for code generation and revision based on user feedback. 
# The following prompt is used to guide the code specialist in generating Pandas code
#  to answer the user's question.
CODE_PROMPT = """
You are a Pandas data analysis coding specialist.

The available Pandas DataFrame is named `covid_df`.

Columns:

    country
    deaths
    recovered

Generate Python/Pandas code that answers the user's question.

Rules:
1. Only use the `covid_df` DataFrame
2. Do not import modules.
3. Do not read files.
4. Do not access the network.
5. Do not modify covid_df.
6. Return only executable Pandas code.
7. The code should print the final answer/result.

The user will execute the generated code and give you 
either the correct result or an error message.

If feedback is provided, revise the previous code.
"""

# We may need a more capable LLM for the code specialist, but for now, we will use the same llm instance.
code_specialist_model = llm

# Because code specialist is more complex, we directly define a custom node function for it, rather than using create_agent.
def code_specialist_node(state: WorkflowState):
    print("\ncode specialist received state:\n", state)
    attempt = state.get("attempt", 0)
    if attempt == 0 or "generated_code" not in state or (
        "generated_code" in state and state["generated_code"].strip() == "" 
    ) or (
        "feedback" in state and state["feedback"] == "No code is seen."
    ):
        prompt = (
            CODE_PROMPT
            + "\n\nUSER QUESTION:\n"
            + state["user_question"]
        )
        print("\nFirst attempt or non-first attempt but previous code lost\n")
    else:
        prompt = (
            CODE_PROMPT
            + "\n\nUSER QUESTION:\n"
            + state["user_question"]
            + "\n\nPREVIOUS CODE:\n"
            + state["generated_code"]
            + "\n\nUSER TEST RESULT / ERROR:\n"
            + state["feedback"]
            + "\n\nRevise the code."
        )
        print(f"\nnon-first attempt [{attempt}] with a prompt:\n", prompt)
    response = code_specialist_model.invoke(
        [
            ("system", CODE_PROMPT),
            ("human", prompt),
        ]
    )
    code = response.text
    print("\ncode generated by code specialist:\n", code)
    newState = {
        "generated_code": code,
        "attempt": attempt + 1,
    }
    print("\ncode specialist returning new state:\n", newState)
    return newState

def human_review_code_node(state: WorkflowState):
    if "generated_code" in state:
        code = state["generated_code"]
        code = code.strip()
    else:
        code = ""
    if code == "":
        return {
            "attempt": 0, #reset attempt, when a revision fails to generated new code ...
            "feedback": "No code is seen"
        }
    else:
        print("\ncode generated by code specialist:\n", code)
        print("\nAsk for user review")
        
    attempt = state.get("attempt", 1)
    feedback = interrupt(
        {
            "type": "pandas_code_review",
            "attempt": attempt,
            "question": state["user_question"],
            "code": code,
            "message": (
                "\n\nExecute this code against covid_df.\n"
                "If the result is correct, type PASS below.\n"
                "Otherwise, copy and past output here.\n"
            ),
        }
    )
    print("\nfeedback received from user:\n", feedback)
    if feedback == "PASS":
        next = "approved"
    else:
        next = feedback
    newState = {
        "feedback": next,
    }
    print("\nhuman reviewer returning new state:\n", newState)
    return newState


# After resuming from the interrupt, we need to decide whether to terminate or go around for the second attempt.
def code_review_loop_router(state: WorkflowState):
    if ("attempt" in state and state["attempt"] >= 3) or (
        "feedback" in state and state["feedback"] == "approved"
        ):
        print("\ncode specialist reached maximum attempts or code is approved by a human code reviewer, finishing.")
        return "finish"
    print("\nInform code specialist to revise the code.")
    return "revise"

# Three specialist nodes for the three specialist agents

def recovered_node(state: WorkflowState):
    result = recovered_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": state["user_question"],
                }
            ]
         }
    )
    return {
        "answer": result["messages"][-1].text
    }


def table_node(state: WorkflowState):
    result = table_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": state["user_question"],
                }
            ]
        }
    )
    return {
        "answer": result["messages"][-1].text
    }


def aggregate_node(state: WorkflowState):
    result = aggregate_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": state["user_question"],
                }
            ]
        }
    )
    return {
        "answer": result["messages"][-1].text #content
    }

# the 5th, visualization specialist node
def visualization_specialist_node(
    state: WorkflowState,
) -> dict:

    print(
        "\nEntering visualization specialist..."
    )

    # Get the parent's runtime config (contains parent's checkpointer)
    parent_config = get_config()

    # Create an isolated config for the subgraph
    #    - Remove checkpoint coordinates so the subgraph uses its OWN checkpointer
    #    - Keep thread_id
    subgraph_config = {
        "configurable": {
            "thread_id": parent_config["configurable"].get("thread_id", "default"),
            # NOTE: Do NOT include checkpoint_ns or checkpoint_id here
            # This forces the subgraph to use its own checkpointer instance
        }
    }
    result = visualization_graph.invoke(
        {
            "user_question":
                state["user_question"]
        },
        subgraph_config
    )

    print(
        "\nVisualization SQL:"
    )

    print(
        result["sql"]
    )


    print(
        "\nVisualization specification:"
    )

    print(
        result["visualization_spec"]
    )

    print(
        "\nVisualization generated:"
    )

    print(
        result["visualization_path"]
    )


    return {
        "visualization_sql":
            result["sql"],

        "visualization_spec":
            result["visualization_spec"],

        "visualization_path":
            result["visualization_path"],
    }



# Build the state graph

builder = StateGraph(
    WorkflowState,
    context_schema=AnalyticsContext
)

builder.add_node("router", router_node)

builder.add_node(
    "recovered_specialist",
    recovered_node,
)

builder.add_node(
    "table_specialist",
    table_node,
)

builder.add_node(
    "aggregate_specialist",
    aggregate_node,
)

builder.add_node(
    "code_specialist",
    code_specialist_node,
)

builder.add_node(
    "human_review_code",
    human_review_code_node,
)

builder.add_node(
    "visualization_specialist",
    visualization_specialist_node
)

# Connect the router to all five specialists, and connect the specialists to the end node.

builder.add_edge(START, "router")

# Conditional edges from the router to the specialists based on the route decision
def route_from_router(state: WorkflowState):
    print("\nRouter decision:", state["route"])
    return state["route"]

builder.add_conditional_edges(
    "router",
    route_from_router,
    {
        "recovered": "recovered_specialist",
        "table": "table_specialist",
        "aggregate": "aggregate_specialist",
        "code": "code_specialist",
        "visualization": "visualization_specialist"
    },
)
builder.add_edge(
    "code_specialist",
    "human_review_code"
)

# The code specialist has a loop for revision based on feedback.
builder.add_conditional_edges(
    "human_review_code",
    code_review_loop_router,
    {
        "revise": "code_specialist",
        "finish": END,
    },
)

builder.add_edge("recovered_specialist", END)
builder.add_edge("table_specialist", END)
builder.add_edge("aggregate_specialist", END)
builder.add_edge("visualization_specialist", END)

# Compile with a checkpointer
# This is essential for the HITL workflow, as it allows the graph to be paused and resumed based on a user feedback.

checkpointer = InMemorySaver()
graph = builder.compile(
    checkpointer=checkpointer
)

# Display ASCII art in console
print(graph.get_graph().draw_ascii())

def main():

    # Load the COVID-19 dataset and register it with DuckDB
    covid_df = load_covid_dataset()
    
    print("\nQuick verfication that covid_df is loaded\n")
    print(covid_df.info())
    print(covid_df.head())
    
    duckdb_conn = connect_and_register_df(covid_df)

    print("\nQuick verfication that the table is registered and accessible via SQL\n")
    print(
        duckdb_conn.execute(
            "SELECT * FROM covid_19 LIMIT 5"
        ).df()
    )

    context = AnalyticsContext(duckdb_conn=duckdb_conn, covid_df=covid_df)

    # Asking a question that should be routed to the recovered specialist
    prompt1 = "\nhit ENTER to proceed to test 'recovered' specialist or type 'quit' to exit:"
    user_input = input(prompt1)

    if user_input == "quit":
        print("\nExiting the test.")
        sys.exit()

    config1 = {
        "configurable": {
            "thread_id": "experiment-1"
        }
    }
    user_question = "Which country reported the largest number of recovered cases?"
    print(f"\nUser question: {user_question}\n")
    result = graph.invoke(
        {
            "user_question":
                user_question
        },
        config=config1,
        context=context
    )


    print(result["answer"])

    # Asking a question that should be routed to the aggregate specialist
    prompt2 = "\nhit any key to proceed to test 'aggregate' specialist"
    input(prompt2)


    config2 = {
            "configurable": {
                "thread_id": "experiment-2"
            }
    }
    user_question = "Which country has reported the largest number of deaths of COVID-19?"
    print(f"\nUser question: {user_question}\n")
    result = graph.invoke(
        {
            "user_question":
                user_question
        },
        config=config2,
        context=context
    )
    print(result["answer"])

    # Asking a question that should be routed to the table specialist
    prompt3 = "\nhit any key to proceed to test 'table' specialist"
    input(prompt3)

    config3 = {
        "configurable": {
            "thread_id": "experiment-3"
        }
    }

    user_question = "Which countries have reported more than 10000 death cases?"
    print(f"\nUser question: {user_question}\n")
    result = graph.invoke(
        {
            "user_question":
                user_question
        },
        config=config3,
        context=context
    )
    print(result["answer"])

    # Asking a question that should be routed to the code specialist
    prompt4 = "\nhit any key to proceed to test 'code' specialist"
    input(prompt4)

    config4 = {
        "configurable": {
            "thread_id": "experiment-4"
        }
    }
    user_question = "Please show me Pandas code to show the schema of the covid_19 dataset."
    print(f"\nUser question: {user_question}\n")
    result = graph.invoke(
        {
            "user_question":
                user_question           
        },
        config=config4,
        context=context
    )

    while "__interrupt__" in result:
        print("\nThe code specialist has requested user feedback.")
        request = result["__interrupt__"][0].value
        request_question = request["question"]
        request_code = request["code"]
        request_attempt = request["attempt"]
        request_message = request["message"]
        print("\n")
        print("=" * 60)
        print("HUMAN INTERVENTION REQUIRED")
        print("=" * 60)

        print("\nQuestion:")
        print(request_question)
        print("\nGenerated code:")
        print("-" * 60)
        print(request_code)
        print("-" * 60)  
        print(f"\nYou have {3-request_attempt} attempt(s) left")
        user_feedback = input(f"{request_message}\n\n")
        result = graph.invoke(
            Command(resume=user_feedback),
            config=config4,
            context=context
            )
        print(f"\nresult: {result}")

    if "answer" in result:
        print(result["answer"])


    # Asking a question that should be routed to the visualization specialist
    prompt5 = "\nhit any key to proceed to test 'visualization' specialist"
    input(prompt5)


    # Display ASCII art in the console
    print(visualization_graph.get_graph().draw_ascii())

    config5 = {
            "configurable": {
                "thread_id": "experiment-5"
            }
    }
    user_question = "Plot the top 10 countries by COVID-19 deaths."
    print(f"\nUser question: {user_question}\n")
    result = graph.invoke(
        {
            "user_question":
                user_question
        },
        config=config5,
        context=context
    )

    print(f"\nFinal result from visualization_graph.invoke: {result}")

    print("\nVisualization spec:")
    print(result["visualization_spec"])

    print("\nImage:")
    print(result["visualization_path"])

    print("Releasing duckdb connection and unregistering covid_19 table...")
    disconnect_and_unregister_df(duckdb_conn)

if __name__ == "__main__":
    main()