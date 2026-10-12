from agent_framework import WorkflowViz

from pandas_analytics_agents.subgraph.maf.visualization_specialist_maf import workflow

viz = WorkflowViz(workflow)
print(viz.to_mermaid())
