from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler
from agent_framework.foundry import FoundryChatClient
from agent_framework.observability import configure_otel_providers
from agent_framework.openai import OpenAIChatClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from pandas_analytics_agents.utils.duckdb import get_df_from_sql_clean

# Confiture OpenTelemetry based on environment variables
configure_otel_providers(enable_sensitive_data=True)

load_dotenv()
outputs = os.environ.get("OUTPUT_DIR", "./outputs")

logger = logging.getLogger(__name__)


def execute_covid_sql(sql: str) -> pd.DataFrame:
    sql_clean = sql.strip()
    if not sql_clean.lower().startswith("select"):
        raise ValueError("Only SELECT statements are allowed.")
    #return duckdb_conn.execute(sql_clean).df()
    #df = duckdb_conn.execute(sql_clean).df()
    df = get_df_from_sql_clean(sql_clean, None)
    return df


# ============================================================
#  Visualization specification
# ============================================================

class VisualizationSpec(BaseModel):
    chart_type: Literal["bar", "line", "scatter", "pie", "histogram", "box"] = Field(
        description="Type of chart to generate."
    )
    x_column: str | None = Field(default=None, description="Column used for the x-axis or category.")
    y_column: str | None = Field(default=None, description="Column used for the y-axis or numeric value.")
    color_column: str | None = Field(default=None, description="Optional column used to group/color the data.")
    title: str = Field(description="Human-readable chart title.")


# This module uses postponed annotations.  Explicitly resolve them before the
# model is passed to the OpenAI structured-output schema builder.
VisualizationSpec.model_rebuild()


# ============================================================
#  Workflow state
# ============================================================

VisualizationState = dict[str, Any]


# ============================================================
#  Model
# ============================================================

MODEL_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME_FOUNDRYPOC", "gpt-5-nano")
FOUNDRY_PROJECT_ENDPOINT = os.environ.get("FOUNDRY_PROJECT_ENDPOINT") or os.environ.get(
    "AZURE_OPENAI_ENDPOINT_FOUNDRYPOC"
)
FOUNDRY_MODEL = os.environ.get("FOUNDRY_MODEL") or MODEL_DEPLOYMENT_NAME


def build_foundry_chat_client() -> FoundryChatClient:
    api_key = os.environ.get("AZURE_OPENAI_API_KEY_FOUNDRYPOC")
    azure_endpoint = os.environ.get("MAF_AZURE_OPENAI_ENDPOINT_FOUNDRYPOC")
    if api_key and azure_endpoint:
        return OpenAIChatClient(
            model=FOUNDRY_MODEL,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
        )
    if not FOUNDRY_PROJECT_ENDPOINT:
        raise ValueError(
            "Set FOUNDRY_PROJECT_ENDPOINT (or AZURE_OPENAI_ENDPOINT_FOUNDRYPOC) for Foundry inference."
        )

    return FoundryChatClient(
        project_endpoint=FOUNDRY_PROJECT_ENDPOINT,
        model=FOUNDRY_MODEL,
        credential=DefaultAzureCredential(),
    )


client = build_foundry_chat_client()


# ============================================================
#  Prompts
# ============================================================

SQL_SYSTEM_PROMPT = """
You are the SQL generation component of a data visualization system.

Your job is to translate the user's visualization request into ONE read-only DuckDB SQL SELECT statement against the table:

    covid_19

The table contains the COVID-19 dataset, which has the following columns:

Data columns (total 15 columns):
 #   Column                  Non-Null Count  Dtype
---  ------                  --------------  -----
 0   country                 187 non-null    str
 1   Confirmed               187 non-null    int64
 2   deaths                  187 non-null    int64
 3   recovered               187 non-null    int64
 4   Active                  187 non-null    int64
 5   New cases               187 non-null    int64
 6   New deaths              187 non-null    int64
 7   New recovered           187 non-null    int64
 8   Deaths / 100 Cases      187 non-null    float64
 9   Recovered / 100 Cases   187 non-null    float64
 10  Deaths / 100 Recovered  187 non-null    float64
 11  Confirmed last week     187 non-null    int64
 12  1 week change           187 non-null    int64
 13  1 week % increase       187 non-null    float64
 14  WHO Region              187 non-null    str

Generate SQL only.

Rules:
1. Only generate SELECT statements. Do NOT generate any comments or explanations.
2. Do not modify data.
3. Do not use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, COPY, ATTACH, or other side-effecting statements.
4. Select only the columns necessary for the requested visualization.
5. If the user asks for "top N", use ORDER BY and LIMIT.
6. Make the result reasonably sized for visualization.
7. Do not generate matplotlib, seaborn, or Python code.

The SQL should produce exactly the dataset needed by the visualization.
"""


VISUALIZATION_SYSTEM_PROMPT = """
You are the visualization-design component of a data analysis system.

Given:
1. The user's original request.
2. The actual DataFrame produced by the SQL query.

Generate a VisualizationSpec describing how the DataFrame should be plotted.

You MUST choose columns that actually exist in the DataFrame.

Available chart types:
- bar
- line
- scatter
- pie
- histogram
- box

Guidelines:
bar:
    categorical x + numeric y
line:
    ordered/sequential x + numeric y
scatter:
    numeric x + numeric y
pie:
    categorical x + numeric y
histogram:
    numeric x
box:
    categorical x + numeric y

Prefer bar charts for ranking/comparison questions.
Prefer line charts for trends over an ordered dimension.
Prefer scatter plots for relationships between two numerical variables.

Do not generate Python, matplotlib, or seaborn code.
Generate only the structured VisualizationSpec.
"""


# ============================================================
#  Helpers
# ============================================================

def clean_sql_text(sql: str) -> str:
    sql = sql.strip()
    if sql.startswith("```"):
        lines = sql.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        sql = "\n".join(lines).strip()
    return sql


def prepare_dataframe_context(df: pd.DataFrame) -> str:
    dtypes = "\n".join(f"- {column}: {dtype}" for column, dtype in df.dtypes.items())
    sample = df.head(20).to_string(index=False)
    return f"""
Columns:

{dtypes}

Number of rows: {len(df)}

Sample rows:

{sample}
""".strip()


def render_visualization(df: pd.DataFrame, spec: VisualizationSpec) -> str:
    plt.figure(figsize=(12, 7))

    if spec.chart_type == "bar":
        sns.barplot(data=df, x=spec.x_column, y=spec.y_column, hue=spec.color_column, legend=(spec.color_column is not None))
    elif spec.chart_type == "line":
        sns.lineplot(data=df, x=spec.x_column, y=spec.y_column, hue=spec.color_column)
    elif spec.chart_type == "scatter":
        sns.scatterplot(data=df, x=spec.x_column, y=spec.y_column, hue=spec.color_column)
    elif spec.chart_type == "pie":
        plt.pie(df[spec.y_column], labels=df[spec.x_column], autopct="%1.1f%%")
    elif spec.chart_type == "histogram":
        sns.histplot(data=df, x=spec.x_column)
    elif spec.chart_type == "box":
        sns.boxplot(data=df, x=spec.x_column, y=spec.y_column, hue=spec.color_column)
    else:
        raise ValueError(f"Unsupported chart type: {spec.chart_type}")

    plt.title(spec.title)
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_dir = Path(outputs)
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(output_path)


# ============================================================
#  Executors
# ============================================================

class GenerateSQLExecutor(Executor):
    @handler
    async def process(self, state: VisualizationState, ctx: WorkflowContext[VisualizationState]) -> None:
        try:
            print("\nStarting to generate sql ...\n")
            response = await client.as_agent(
                name="GenerateSQL",
                instructions=SQL_SYSTEM_PROMPT,
            ).run(state["user_question"])

            sql = clean_sql_text((response.text or "").strip())
            print(f"Generated sql:\n{sql}\n\n")
            await ctx.send_message({**state, "sql": sql})
        except Exception as exc:
            logger.exception("SQL generation failed")
            await ctx.yield_output({**state, "status": "error", "error": f"SQL generation failed: {exc}"})


class ExecuteSQLExecutor(Executor):
    @handler
    async def process(self, state: VisualizationState, ctx: WorkflowContext[VisualizationState]) -> None:
        print("\nExecuting sql ...\n")
        try:
            # This is a synchronous function, so we run it in a separate thread to avoid blocking the event loop.
            #df = execute_covid_sql(state["sql"])
            df = await asyncio.to_thread(
                execute_covid_sql,
                state["sql"],
            )
            if df.empty:
                raise ValueError("The SQL query returned no rows.")
            print(f"Returned DataFrame shape: {df.shape}\n\n")
            await ctx.send_message({**state, "dataframe": df})
        except Exception as exc:
            logger.exception("SQL execution failed")
            await ctx.yield_output({**state, "status": "error", "error": f"SQL execution failed: {exc}"})


class GenerateVisualizationSpecExecutor(Executor):
    @handler
    async def process(self, state: VisualizationState, ctx: WorkflowContext[VisualizationState]) -> None:
        try:
            print("\nGenerating visualization spec ...\n")
            df = state["dataframe"]
            context = prepare_dataframe_context(df)
            response = await client.as_agent(
                name="GenerateVisualizationSpec",
                instructions=VISUALIZATION_SYSTEM_PROMPT,
            ).run(
                f"""
User request:

{state["user_question"]}

DataFrame produced by SQL:

{context}
""".strip(),
                options={"response_format": VisualizationSpec},
            )

            spec = response.value
            if spec is None:
                raise ValueError("Failed to generate a structured visualization specification.")
            print(f"Visualization spec generated:\n{spec}\n\n")
            await ctx.send_message({**state, "visualization_spec": spec})
        except Exception as exc:
            logger.exception("Visualization specification generation failed")
            await ctx.yield_output({**state, "status": "error", "error": f"Visualization specification generation failed: {exc}"})


class ValidateVisualizationSpecExecutor(Executor):
    @handler
    async def process(self, state: VisualizationState, ctx: WorkflowContext[VisualizationState]) -> None:
        try:
            print("\nValidating visualization spec ...\n")
            spec = state["visualization_spec"]
            df = state["dataframe"]
            columns = set(df.columns)

            if spec.x_column is not None and spec.x_column not in columns:
                raise ValueError(f"x_column '{spec.x_column}' does not exist in DataFrame.")
            if spec.y_column is not None and spec.y_column not in columns:
                raise ValueError(f"y_column '{spec.y_column}' does not exist in DataFrame.")
            if spec.color_column is not None and spec.color_column not in columns:
                raise ValueError(f"color_column '{spec.color_column}' does not exist in DataFrame.")

            if spec.chart_type in {"bar", "line", "scatter", "box"}:
                if not spec.x_column:
                    raise ValueError(f"{spec.chart_type} chart requires x_column.")
                if not spec.y_column:
                    raise ValueError(f"{spec.chart_type} chart requires y_column.")

            if spec.chart_type == "histogram" and not spec.x_column:
                raise ValueError("Histogram requires x_column.")

            if spec.chart_type == "pie":
                if not spec.x_column:
                    raise ValueError("Pie chart requires x_column.")
                if not spec.y_column:
                    raise ValueError("Pie chart requires y_column.")

            print("Visualization spec validation completed!\n\n")
            await ctx.send_message(state)
        except Exception as exc:
            logger.exception("Visualization specification validation failed")
            await ctx.yield_output({**state, "status": "error", "error": f"Visualization specification validation failed: {exc}"})


class RenderVisualizationExecutor(Executor):
    @handler
    async def process(self, state: VisualizationState, ctx: WorkflowContext[VisualizationState]) -> None:
        try:
            print("\nStarting to plot ...\n")
            path = render_visualization(state["dataframe"], state["visualization_spec"])
            print("Plotting done.\n\n")
            await ctx.yield_output({**state, "status": "success", "visualization_path": path})
        except Exception as exc:
            logger.exception("Visualization rendering failed")
            await ctx.yield_output({**state, "status": "error", "error": f"Visualization rendering failed: {exc}"})


# ============================================================
#  Build workflow
# ============================================================

generate_sql_executor = GenerateSQLExecutor(id="generate_sql")
execute_sql_executor = ExecuteSQLExecutor(id="execute_sql")
generate_visualization_spec_executor = GenerateVisualizationSpecExecutor(id="generate_visualization_spec")
validate_visualization_spec_executor = ValidateVisualizationSpecExecutor(id="validate_visualization_spec")
render_visualization_executor = RenderVisualizationExecutor(id="render_visualization")

workflow = (
    WorkflowBuilder(start_executor=generate_sql_executor)
    .add_chain(
        [
            generate_sql_executor,
            execute_sql_executor,
            generate_visualization_spec_executor,
            validate_visualization_spec_executor,
            render_visualization_executor,
        ]
    )
    .build()
)


# ============================================================
#  Local run
# ============================================================

async def main() -> None:
    result = await workflow.run(
        {
            "user_question": "Show me the top 10 countries by COVID-19 deaths."
        }
    )

    events = result.value if hasattr(result, "value") else result
    if not events:
        raise ValueError("Workflow returned no events.")

    final_event = next(
        (event for event in reversed(events) if getattr(event, "type", None) == "output"),
        events[-1],
    )

    final_state = getattr(final_event, "data", final_event)
    if final_state.get("error"):
        print(f"Error: {final_state['error']}")
    print("SQL:")
    print(final_state["sql"])
    print("\nVisualization spec:")
    print(final_state["visualization_spec"])
    print("\nImage:")
    print(final_state["visualization_path"])


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
