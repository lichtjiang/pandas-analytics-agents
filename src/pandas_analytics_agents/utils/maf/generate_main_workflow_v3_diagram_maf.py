from agent_framework import WorkflowViz

from pandas_analytics_agents.main_workflow.maf.main_workflow_maf_v3 import workflow

viz = WorkflowViz(workflow)
print(viz.to_mermaid())
