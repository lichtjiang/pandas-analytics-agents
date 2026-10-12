"""COVID-19 triage/router workflow implemented with Microsoft Agent Framework.

The visualization workflow is kept as a separate workflow and is invoked by
the visualization specialist.  The other specialists are Agent Framework
agents using the same data and tools as the LangGraph implementation.

Note though, we only have one executor, which is the router. 
The other specialists are agents that are invoked.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from typing import Any, Literal

from agent_framework import (
    Executor,
    FunctionInvocationContext,
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

# Two reasons to have a global variable to hold runtime-dependencies for tool calling
# 1) We need a way to pass the function_invocation_kwargs to the agents for DevUI. 
#   Currently the DevUI doesn't support passing runtime dependencies like duckdb connection into a workflow when running it,
#   i.e. workflow.run({"user_question": question}, function_invocation_kwargs={"duckdb_conn": duckdb_conn}) doesn't work because DevUI doesn't support
#   that yet.  So we use a global variable to inject function_invocation_kwargs into agent.run() at runtime.
# 2) DuckDB connection and covid_df are runtime dependencies that are created at runtime. 
#   Preferrably, we pass both in a dict to workflow.run() like this
#   workflow.run({"user_question": question}, function_invocation_kwargs={"duckdb_conn": duckdb_conn, "covid_df": covid_df})
#   But DuckDB connection cannot be pickled, so we cannot pass it as a parameter to workflow.run() 
#   because MAF context injection machinery will try to pickle whatever in the context when passing them around the workflow and down to the agent.
#   So, we skip passing the duckdb connection and covid_df as parameters to workflow.run()
#   and instead use a global variable to hold them and directlly inject them into agent.run() at runtime, 
#   so that they will be injected into a tool calling and available via FunctionInvocationContext.kwargs.
function_invocation_kwargs = {}
# Note: in V2, we will refactor the code to avoid using a global variable and instead use a class to hold the runtime dependencies and tool definitions together,
#   and then inject that class instance into the agents at runtime, so that the tools can access the runtime dependencies via the class instance without being pickled.

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("covid-workflow")

@tool
def find_country_with_most_recovered(ctx: FunctionInvocationContext) -> str:
    covid_df = ctx.kwargs["covid_df"]
    row = covid_df.loc[covid_df["recovered"].idxmax()]
    return f"Country: {row['country']}\nRecovered cases: {int(row['recovered'])}"

@tool
def query_covid_table(sql: str, ctx: FunctionInvocationContext) -> str:
    if not sql.strip().lower().startswith("select"):
        return "ERROR: Only SELECT queries are allowed."
    try:
        duckdb_conn = ctx.kwargs["duckdb_conn"]
        result = duckdb_conn.execute(sql).df()
        return "The query returned no rows." if result.empty else result.to_json(orient="records")
    except Exception as exc:
        return f"ERROR executing SQL: {exc}"

@tool
def aggregate_covid_data(sql: str, ctx: FunctionInvocationContext) -> str:
    if not sql.strip().lower().startswith("select"):
        return "ERROR: Only SELECT queries are allowed."
    try:
        duckdb_conn = ctx.kwargs["duckdb_conn"]
        result = duckdb_conn.execute(sql).df()
        if len(result) != 1 or len(result.columns) != 1:
            return "ERROR: This tool expects exactly one row and one column."
        return str(result.iloc[0, 0])
    except Exception as exc:
        return f"ERROR executing aggregation SQL: {exc}"


MODEL = os.environ.get("FOUNDRY_MODEL") or os.environ.get(
    "AZURE_OPENAI_DEPLOYMENT_NAME_FOUNDRYPOC", "gpt-5-nano"
)
ENDPOINT = os.environ.get("FOUNDRY_PROJECT_ENDPOINT") or os.environ.get(
    "AZURE_OPENAI_ENDPOINT_FOUNDRYPOC"
)
API_KEY = os.environ.get("AZURE_OPENAI_API_KEY_FOUNDRYPOC")
AZURE_OPENAI_ENDPOINT = os.environ.get("MAF_AZURE_OPENAI_ENDPOINT_FOUNDRYPOC")
if not ENDPOINT and not AZURE_OPENAI_ENDPOINT:
    raise ValueError("Set FOUNDRY_PROJECT_ENDPOINT or AZURE_OPENAI_ENDPOINT_FOUNDRYPOC.")
if API_KEY and AZURE_OPENAI_ENDPOINT:
    client = OpenAIChatClient(
        model=MODEL,
        api_key=API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    log.info("Azure OpenAI client configured with API-key authentication for model %s", MODEL)
else:
    client = FoundryChatClient(
        project_endpoint=ENDPOINT,
        model=MODEL,
        credential=DefaultAzureCredential(),
    )
    log.info("Foundry client configured with Entra authentication for model %s", MODEL)


Route = Literal["recovered", "table", "aggregate", "code", "visualization"]


class RouteDecision(BaseModel):
    route: Route = Field(description="The one specialist that should handle the request.")


ROUTER_PROMPT = """You are a COVID-19 data-analysis triage router. Return exactly one route.
Use recovered only for the country with the largest recovered count; table for
multiple rows, filtering, sorting, or lists; aggregate for one aggregate/winner
value; code for Pandas code requests; visualization for chart, plot, graph,
trend, or visual comparison requests. Explicit visualization requests win."""


def _agent(name: str, instructions: str, tools: list[Any] | None = None):
    return client.as_agent(name=name, instructions=instructions, tools=tools or [])


recovered_agent = _agent("RecoveredSpecialist", """Answer using the recovered-country result supplied by the tool context. Be concise.""", [find_country_with_most_recovered])
table_agent = _agent("TableSpecialist", """Answer table questions. Generate a read-only SELECT against covid_19 with columns country, deaths, recovered, call query_covid_table, and summarize the returned rows.""", [query_covid_table])
aggregate_agent = _agent("AggregateSpecialist", """Answer single-value questions. Generate a read-only SELECT aggregate against covid_19, call aggregate_covid_data, and explain the result.""", [aggregate_covid_data])
code_agent = _agent("CodeSpecialist", """Generate only executable Pandas code using the covid_df DataFrame. Do not import, read files, use the network, or mutate covid_df. Print the result.""")

async def _run_agent(agent: Any, question: str, extra: str = "", *, function_invocation_kwargs: dict[str, Any] | None = None) -> str:
    log.info("Starting agent call: %s", getattr(agent, "name", type(agent).__name__))
    response = await agent.run(
                                (extra + "\n\n" + question).strip(),
                                function_invocation_kwargs=function_invocation_kwargs,
                            )
    log.info("Agent call completed: %s", getattr(agent, "name", type(agent).__name__))
    return response.text or str(response.value)


def _workflow_output(result: Any) -> dict[str, Any]:
    """Extract the last non-empty output event from a workflow result.

    Agent Framework may append a completion event with ``data=None`` after the
    actual output event, so the last event is not necessarily the result.
    """
    value = getattr(result, "value", result)
    if isinstance(value, dict):
        return value
    if not value:
        raise RuntimeError("Workflow completed without producing an output event.")
    for event in reversed(value):
        data = getattr(event, "data", None)
        if getattr(event, "type", None) == "output" and isinstance(data, dict):
            return data
    for event in reversed(value):
        data = getattr(event, "data", None)
        if isinstance(data, dict):
            return data
    raise RuntimeError("Workflow completed, but no dictionary output was found.")


class TriageExecutor(Executor):
    """Routes one request and returns the specialist's final state."""

    @handler
    async def process(self, state: str | dict, ctx: WorkflowContext[Any]) -> None:
        print(f"type of state: {type(state)}")
        print("Processing state:", state)
        # DevUI uses the handler's primary input type to render its input
        # control and normally sends a string.  The CLI path still sends the
        # original state dictionary, so support both invocation styles.
        if isinstance(state, str):
            question = state
            initial_state: dict[str, Any] = {"user_question": question}
        elif isinstance(state, dict):
            question = state.get("user_question") or state.get("message") or state.get("input")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    "Expected a question string, or an object containing "
                    "'user_question' or 'message' or 'input'."
                )
            initial_state = {**state, "user_question": question}
        else:
            raise TypeError(f"Unsupported workflow input type: {type(state).__name__}")
        log.info("Workflow started; routing question: %s", question)
        log.info("Router model call started")
        decision = await client.as_agent(
            name="Router", instructions=ROUTER_PROMPT
        ).run(question, options={"response_format": RouteDecision})
        route = decision.value.route
        log.info("Router model call completed; selected route=%s", route)
        result: dict[str, Any] = {**initial_state, "route": route}

        if route == "recovered":
            result["answer"] = await _run_agent(
                recovered_agent, 
                question,
                function_invocation_kwargs=function_invocation_kwargs
            )
        elif route == "table":
            result["answer"] = await _run_agent(
                table_agent, 
                question, 
                function_invocation_kwargs=function_invocation_kwargs
            )
        elif route == "aggregate":
            result["answer"] = await _run_agent(
                aggregate_agent, 
                question, 
                function_invocation_kwargs=function_invocation_kwargs
            )
        elif route == "code":
            log.info("Entering code specialist")
            code = await _run_agent(code_agent, question)
            for attempt in range(1, 4):
                print(f"\nGenerated Pandas code (attempt {attempt}/3):\n{code}")
                feedback = input("Execute it against covid_df, then enter PASS or the error/result: ").strip()
                if feedback.upper() == "PASS":
                    break
                code = await _run_agent(
                    code_agent,
                    question,
                    f"Previous code:\n{code}\n\nUser test result / error:\n{feedback}\nRevise the code.",
                )
            result["generated_code"] = code
            result["answer"] = result["generated_code"]
        else:
            log.info("Entering visualization specialist; starting sub-workflow")
            subresult = await visualization_workflow.run({"user_question": question})
            log.info("Visualization sub-workflow completed; extracting final output")
            final = _workflow_output(subresult)
            result.update({k: final[k] for k in ("sql", "visualization_spec", "visualization_path") if k in final})
        log.info("Workflow completed successfully; emitting output")
        await ctx.yield_output(result)


triage_executor = TriageExecutor(id="triage_router")
workflow = WorkflowBuilder(start_executor=triage_executor).build()


async def main() -> None:
    while True:
        question = input("Ask a COVID-19 data question (hit Enter to exit): ").strip()
        if not question:
            return
        log.info("Submitting question to Agent Framework workflow")
        # Note: we don't pass function_invocation_kwargs here because duckdb_conn cannot be pickled ...
        result = await workflow.run({"user_question": question})
        final = _workflow_output(result)
        log.info("Received final workflow result")
        print(final)


def run_devui() -> None:
    """Serve this workflow through Agent Framework DevUI and open a browser."""
    from agent_framework.devui import serve

    log.info("Starting MAF DevUI for the COVID-19 workflow")
    # Note: there is no way, as I know, to pass function_invocation_kwargs to the workflow when running it through DevUI,
    #   so we rely on the global variable instead.
    serve(
        entities=[workflow],
        auto_open=True,
        mode="developer",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the COVID-19 agent workflow.")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Launch the workflow through Microsoft Agent Framework DevUI.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Load dataset, create DuckDB connection, and register the DataFrame
    covid_df = load_covid_dataset()
    duckdb_conn = connect_and_register_df(covid_df)
    function_invocation_kwargs = {
        "duckdb_conn": duckdb_conn, 
        "covid_df": covid_df
    }
    # Then run the main workflow or DevUI based on command-line arguments
    args = parse_args()
    if args.dev:
        run_devui()
    else:
        asyncio.run(main())
    # Clean up: unregister the DataFrame and close the DuckDB connection
    disconnect_and_unregister_df(duckdb_conn)
