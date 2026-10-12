import os
from pathlib import Path

from pandas_analytics_agents.subgraph.langchain.visualization_specialist_langchain import (
    visualization_graph,
)


def main():
    outputs = os.environ.get("OUTPUT_DIR", "./outpouts")
    IMG_FILE_NAME = "visualization_subgraph_diagram_langchain.png"

    OUTPUT_DIR = Path(outputs)
    OUTPUT_DIR.mkdir(
        exist_ok=True
    )
    
    OUTPUT_IMG_FILE = OUTPUT_DIR / IMG_FILE_NAME

    print(f"OUTPUT_IMG_FILE: {os.path.abspath(OUTPUT_IMG_FILE)}")

    print(visualization_graph.get_graph().draw_ascii())
    print(visualization_graph.get_graph().draw_mermaid())


    png_bytes = visualization_graph.get_graph().draw_mermaid_png()

    with open(OUTPUT_IMG_FILE, "wb") as f:
        f.write(png_bytes)


if __name__ == "__main__":
    main()