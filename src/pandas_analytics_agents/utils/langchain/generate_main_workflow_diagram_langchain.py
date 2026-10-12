import os
from pathlib import Path

from pandas_analytics_agents.main_workflow.langchain.main_workflow_langchain import (
    graph,
)


def main():
    outputs = os.environ.get("OUTPUT_DIR", "./outputs")
    IMG_FILE_NAME = "main_workflow_diagram_langchain.png"

    OUTPUT_DIR = Path(outputs)
    OUTPUT_DIR.mkdir(
        exist_ok=True
    )
    
    OUTPUT_IMG_FILE = OUTPUT_DIR / IMG_FILE_NAME

    print(f"OUTPUT_IMG_FILE: {os.path.abspath(OUTPUT_IMG_FILE)}")

    print(graph.get_graph().draw_ascii())
    print(graph.get_graph().draw_mermaid())


    png_bytes = graph.get_graph().draw_mermaid_png()

    with open(OUTPUT_IMG_FILE, "wb") as f:
        f.write(png_bytes)


if __name__ == "__main__":
    main()