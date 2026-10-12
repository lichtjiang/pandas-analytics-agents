"""COVID-19 workflow with DevUI-compatible human review for generated code.

This version keeps the v2 parent topology but replaces the code specialist's
blocking ``input()`` loop with Agent Framework ``request_info`` and a
``response_handler``.  For fun, instead of imported as a module, the v2 file 
is imported only for its data, agents, and non-code executor implementations;
 it is not modified.
"""
import argparse
import asyncio
import importlib.util
import logging
import os
from pathlib import Path
from typing import Any

from agent_framework import (
    Executor,
    InMemoryCheckpointStorage,
    WorkflowBuilder,
    WorkflowContext,
    handler,
    response_handler,
)
from agent_framework.observability import configure_otel_providers
from pydantic import BaseModel, Field

# Confiture OpenTelemetry based on environment variables
configure_otel_providers(enable_sensitive_data=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("covid-workflow-v3")


def load_v2() -> Any:
    path = Path(__file__).with_name("main_workflow_maf_v2.py")
    spec = importlib.util.spec_from_file_location("main_workflow_maf_v2", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v2 = load_v2()
# v2 uses postponed annotations and is loaded dynamically, so Pydantic needs
# the dynamically loaded Route alias supplied explicitly for structured output.
v2.RouteDecision.model_rebuild(_types_namespace={"Route": v2.Route})


class CodeReviewRequest(BaseModel):
    question: str = Field(description="The original user question.")
    code: str = Field(description="Pandas code to execute against covid_df.")
    attempt: int = Field(description="Current review attempt, from 1 through 3.")
    instructions: str = Field(description="Instructions for reviewing the generated code.")


class CodeSpecialistHITL(Executor):
    """Generate code, pause in DevUI, and revise it from reviewer feedback."""

    @handler
    async def process(self, state: dict[str, Any], ctx: WorkflowContext[Any]) -> None:
        question = state["user_question"]
        code = await v2.run_agent(v2.code_agent, question)
        await self._request_review(state, code, 1, ctx)

    async def _request_review(
        self,
        state: dict[str, Any],
        code: str,
        attempt: int,
        ctx: WorkflowContext[Any],
    ) -> None:
        log.info("Requesting human code review, attempt %d/3", attempt)
        await ctx.request_info(
            CodeReviewRequest(
                question=state["user_question"],
                code=code,
                attempt=attempt,
                instructions=(
                    "Execute this code against covid_df. Reply PASS if correct; "
                    "otherwise provide the execution result or error so the code "
                    "specialist can revise it.\n\n"
                ),
            ),
            response_type=str,
            # Each review attempt must have its own request ID. Reusing the
            # executor ID causes a revised request to collide with the
            # already-resolved request when the workflow is resumed.
            request_id=f"code-review-{id(self)}-{attempt}",
        )

    @response_handler
    async def handle_review(
        self,
        request: CodeReviewRequest,
        feedback: str,
        ctx: WorkflowContext[dict[str, Any]],
    ) -> None:
        feedback = feedback.strip()
        log.info("Received human code review for attempt %d", request.attempt)
        if feedback.upper() == "PASS" or request.attempt >= 3:
            await ctx.yield_output(
                {
                    "user_question": request.question,
                    "route": "code",
                    "generated_code": request.code,
                    "answer": request.code,
                    "review": feedback,
                }
            )
            return

        revised_code = await v2.run_agent(
            v2.code_agent,
            request.question,
            f"Previous code:\n{request.code}\n\nUser test result / error:\n{feedback}\nRevise the code.",
        )
        await self._request_review(
            {"user_question": request.question},
            revised_code,
            request.attempt + 1,
            ctx,
        )


# Reuse v2's visible router and the four non-code specialist executors.  The
# visualization executor still invokes the existing visualization sub-workflow.
router = v2.RouterExecutor(id="router")
recovered = v2.RecoveredSpecialistExecutor(id="recovered_specialist")
table = v2.TableSpecialistExecutor(id="table_specialist")
aggregate = v2.AggregateSpecialistExecutor(id="aggregate_specialist")
code = CodeSpecialistHITL(id="code_specialist")
visualization = v2.VisualizationSpecialistExecutor(id="visualization_specialist")

workflow = (
    WorkflowBuilder(
        start_executor=router,
        name="COVID Triage Router Workflow with HITL",
        checkpoint_storage=InMemoryCheckpointStorage(),
    )
    .add_edge(router, recovered, lambda state: state["route"] == "recovered")
    .add_edge(router, table, lambda state: state["route"] == "table")
    .add_edge(router, aggregate, lambda state: state["route"] == "aggregate")
    .add_edge(router, code, lambda state: state["route"] == "code")
    .add_edge(router, visualization, lambda state: state["route"] == "visualization")
    .build()
)


def run_devui() -> None:
    from agent_framework.devui import serve

    log.info("Starting MAF DevUI with code-specialist HITL enabled")
    devui_token = os.environ.get("MAF_DEVUI_TOKEN", "hellow_token_321")
    # Given that we have called configure_otel_providers() to configure exporters/instrumentation at application level,
    # calling multiple instrumentation/configuration mechanisms can be undesirable if they register providers more than once.
    # Agent Framework’s observability documentation recommends configuring providers once during application startup.
    # So, we disable instrumentation in DevUI to avoid potential conflicts or duplicate registrations by setting instrumentation_enabled=False.
    serve(entities=[workflow], instrumentation_enabled=False, auth_token=devui_token, auto_open=True, mode="developer")


async def run_cli() -> None:
    while True:
        question = input("Ask a COVID-19 data question (hit ENTER to exit): ").strip()
        if not question:
            break

        # A request_info event ends the current run in an idle state.  Resume the
        # same workflow by passing the reviewer response keyed by request ID.
        result = await workflow.run(question)
        while True:
            events = getattr(result, "value", result)
            pending = next(
                (event for event in events if getattr(event, "type", None) == "request_info"),
                None,
            )
            if pending is None:
                break

            request = pending.data
            print("\nGenerated Pandas code:\n")
            print(request.code)
            #feedback = input("Enter PASS, or the execution result/error for revision: ").strip()
            feedback = input(request.instructions).strip()
            result = await workflow.run(responses={pending.request_id: feedback})

        log.info("Workflow result: %s", getattr(result, "value", result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", help="Launch the workflow through MAF DevUI.")
    args = parser.parse_args()
    if args.dev:
        run_devui()
    else:
        asyncio.run(run_cli())
