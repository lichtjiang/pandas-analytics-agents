"""COVID-19 workflow with a DevUI-visible router and specialist topology.

Unlike the first Agent Framework version in "main_workflow_maf_v1.py", 
each routing and specialist step is its own executor.  This makes 
the complete parent graph visible in DevUI.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from typing import Any, Literal

from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    handler,
    tool,
)
from agent_framework.foundry import FoundryChatClient
from agent_framework.openai import OpenAIChatClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from pandas_analytics_agents.subgraph.maf.visualization_specialist_maf import (
    workflow as visualization_workflow,
)
from pandas_analytics_agents.utils.dataset import load_covid_dataset
from pandas_analytics_agents.utils.duckdb import (
    connect_and_register_df,
    disconnect_and_unregister_df,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("covid-workflow-v2")


# Long-lived resources: encapsulate the DuckDB connection and DataFrame in a class together with the tool definitions, 
# so that they can be injected into the tools at runtime without being pickled and made available directly to the tools 
# via self.duckdb_conn and self.covid_df, without being passed as parameters to workflow.run() or agent.run(), which 
# would require pickling and fail because DuckDB connection cannot be pickled.
class AnalyticsTools:

    def __init__(self, duckdb_conn, covid_df):
        self.duckdb_conn = duckdb_conn
        self.covid_df = covid_df

    @tool
    def find_country_with_most_recovered(self) -> str:
        covid_df = self.covid_df
        row = covid_df.loc[covid_df["recovered"].idxmax()]
        return f"Country: {row['country']}\nRecovered cases: {int(row['recovered'])}"

    @tool
    def query_covid_table(self, sql: str) -> str:
        if not sql.strip().lower().startswith("select"):
            return "ERROR: Only SELECT queries are allowed."
        try:
            duckdb_conn = self.duckdb_conn
            result = duckdb_conn.execute(sql).df()
            return "The query returned no rows." if result.empty else result.to_json(orient="records")
        except Exception as exc:
            return f"ERROR executing SQL: {exc}"

    @tool
    def aggregate_covid_data(self, sql: str) -> str:
        if not sql.strip().lower().startswith("select"):
            return "ERROR: Only SELECT queries are allowed."
        try:
            duckdb_conn = self.duckdb_conn
            result = duckdb_conn.execute(sql).df()
            if len(result) != 1 or len(result.columns) != 1:
                return "ERROR: This tool expects exactly one row and one column."
            return str(result.iloc[0, 0])
        except Exception as exc:
            return f"ERROR executing aggregation SQL: {exc}"

# Load dataset, create DuckDB connection, and register the DataFrame
covid_df = load_covid_dataset()
duckdb_conn = connect_and_register_df(covid_df)
log.info("Data loaded: %d rows; DuckDB table covid_19 registered", len(covid_df))

# Create one long-lived instance for holding the DuckDB connection and DataFrame together with tool definitions
# This is necessary because the DuckDB connection cannot be pickled, and MAF context injection machinery will 
# try to pickle whatever in the context when passing them around the workflow and down to the agent.
analytics_tools = AnalyticsTools(duckdb_conn=duckdb_conn, covid_df=covid_df)


# Model layer
MODEL = os.environ.get("FOUNDRY_MODEL") or os.environ.get(
    "AZURE_OPENAI_DEPLOYMENT_NAME_FOUNDRYPOC", "gpt-5-nano"
)
PROJECT_ENDPOINT = os.environ.get("FOUNDRY_PROJECT_ENDPOINT") or os.environ.get("PROJECT_ENDPOINT_FOUNDRYPOC")
AZURE_ENDPOINT = os.environ.get("MAF_AZURE_OPENAI_ENDPOINT_FOUNDRYPOC")
API_KEY = os.environ.get("AZURE_OPENAI_API_KEY_FOUNDRYPOC")

if API_KEY and AZURE_ENDPOINT:
    client = OpenAIChatClient(model=MODEL, api_key=API_KEY, azure_endpoint=AZURE_ENDPOINT)
    log.info("Azure OpenAI client configured with API-key authentication for model %s", MODEL)
elif PROJECT_ENDPOINT:
    client = FoundryChatClient(project_endpoint=PROJECT_ENDPOINT, model=MODEL, credential=DefaultAzureCredential())
    log.info("Foundry client configured with Entra authentication for model %s", MODEL)
else:
    raise ValueError("Set Azure OpenAI API-key/endpoint or FOUNDRY_PROJECT_ENDPOINT.")

Route = Literal["recovered", "table", "aggregate", "code", "visualization"]


class RouteDecision(BaseModel):
    route: Route = Field(description="The one specialist that should handle the request.")


ROUTER_PROMPT = """You are a COVID-19 data-analysis triage router. Return exactly one route.
Use recovered only for the country with the largest recovered count; table for
multiple rows, filtering, sorting, or lists; aggregate for one aggregate/winner
value; code for Pandas code requests; visualization for chart, plot, graph,
trend, or visual comparison requests. Explicit visualization requests win."""


def make_agent(name: str, instructions: str, tools: list[Any] | None = None) -> Any:
    return client.as_agent(name=name, instructions=instructions, tools=tools or [])


router_agent = client.as_agent(name="Router", instructions=ROUTER_PROMPT)
recovered_agent = make_agent("RecoveredSpecialist", "Answer concisely using the recovered-country tool result.", [analytics_tools.find_country_with_most_recovered])
table_agent = make_agent("TableSpecialist", "Generate a read-only SELECT against covid_19, call query_covid_table, and summarize the rows.", [analytics_tools.query_covid_table])
aggregate_agent = make_agent("AggregateSpecialist", "Generate a one-row, one-column read-only SELECT, call aggregate_covid_data, and explain the result.", [analytics_tools.aggregate_covid_data])
code_agent = make_agent("CodeSpecialist", "Generate only executable Pandas code using covid_df. Do not import, read files, use the network, or mutate covid_df. Print the result.")


async def run_agent(agent: Any, question: str, context: str = "") -> str:
    log.info("Starting agent call: %s", getattr(agent, "name", type(agent).__name__))
    response = await agent.run((context + "\n\n" + question).strip())
    log.info("Agent call completed: %s", getattr(agent, "name", type(agent).__name__))
    return response.text or str(response.value)


class RouterExecutor(Executor):
    @handler
    async def process(self, question: str, ctx: WorkflowContext[dict[str, Any]]) -> None:
        log.info("Router executor received question: %s", question)
        decision = await router_agent.run(question, options={"response_format": RouteDecision})
        route = decision.value.route
        log.info("Router selected route=%s", route)
        await ctx.send_message({"user_question": question, "route": route})


class RecoveredSpecialistExecutor(Executor):
    @handler
    async def process(self, state: dict[str, Any], ctx: WorkflowContext[Any]) -> None:
        answer = await run_agent(recovered_agent, state["user_question"]) #, find_country_with_most_recovered())
        await ctx.yield_output({**state, "answer": answer})


class TableSpecialistExecutor(Executor):
    @handler
    async def process(self, state: dict[str, Any], ctx: WorkflowContext[Any]) -> None:
        answer = await run_agent(table_agent, state["user_question"])
        await ctx.yield_output({**state, "answer": answer})


class AggregateSpecialistExecutor(Executor):
    @handler
    async def process(self, state: dict[str, Any], ctx: WorkflowContext[Any]) -> None:
        answer = await run_agent(aggregate_agent, state["user_question"])
        await ctx.yield_output({**state, "answer": answer})


class CodeSpecialistExecutor(Executor):
    @handler
    async def process(self, state: dict[str, Any], ctx: WorkflowContext[Any]) -> None:
        question = state["user_question"]
        code = await run_agent(code_agent, question)
        for attempt in range(1, 4):
            print(f"\nGenerated Pandas code (attempt {attempt}/3):\n{code}")
            feedback = input("Execute it against covid_df, then enter PASS or the error/result: ").strip()
            if feedback.upper() == "PASS":
                break
            code = await run_agent(code_agent, question, f"Previous code:\n{code}\nUser test result / error:\n{feedback}\nRevise the code.")
        await ctx.yield_output({**state, "generated_code": code, "answer": code})


class VisualizationSpecialistExecutor(Executor):
    @handler
    async def process(self, state: dict[str, Any], ctx: WorkflowContext[Any]) -> None:
        log.info("Starting visualization sub-workflow")
        result = await visualization_workflow.run({"user_question": state["user_question"]})
        value = getattr(result, "value", result)
        output = next((getattr(event, "data", None) for event in reversed(value) if getattr(event, "type", None) == "output" and isinstance(getattr(event, "data", None), dict)), None) if not isinstance(value, dict) else value
        if not isinstance(output, dict):
            raise TypeError("Visualization sub-workflow returned no output dictionary.")
        log.info("Visualization sub-workflow completed")
        await ctx.yield_output({**state, **output})


router = RouterExecutor(id="router")
recovered = RecoveredSpecialistExecutor(id="recovered_specialist")
table = TableSpecialistExecutor(id="table_specialist")
aggregate = AggregateSpecialistExecutor(id="aggregate_specialist")
code = CodeSpecialistExecutor(id="code_specialist")
visualization = VisualizationSpecialistExecutor(id="visualization_specialist")

workflow = (
    WorkflowBuilder(start_executor=router, name="COVID Triage Router Workflow")
    .add_edge(router, recovered, lambda state: state["route"] == "recovered")
    .add_edge(router, table, lambda state: state["route"] == "table")
    .add_edge(router, aggregate, lambda state: state["route"] == "aggregate")
    .add_edge(router, code, lambda state: state["route"] == "code")
    .add_edge(router, visualization, lambda state: state["route"] == "visualization")
    .build()
)


def run_devui() -> None:
    from agent_framework.devui import serve

    log.info("Starting MAF DevUI for the complete COVID-19 workflow topology")
    devui_token = os.environ.get("MAF_DEVUI_TOKEN", "hellow_token_321")
    serve(entities=[workflow], auth_token=devui_token, auto_open=True, mode="developer")


async def run_cli() -> None:

    while True:
        question = input("Ask a COVID-19 data question (hit ENTER to quit): ").strip()
        if not question:
            return
        result = await workflow.run(question)
        value = getattr(result, "value", result)
        if isinstance(value, dict):
            print(value)
        else:
            output = next((getattr(event, "data", None) for event in reversed(value) if isinstance(getattr(event, "data", None), dict)), None)
            print(output)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", help="Launch the workflow through MAF DevUI.")
    args = parser.parse_args()
    if args.dev:
        run_devui()
    else:
        asyncio.run(run_cli())

    # Clean up: unregister the DataFrame and close the DuckDB connection
    disconnect_and_unregister_df(duckdb_conn)
