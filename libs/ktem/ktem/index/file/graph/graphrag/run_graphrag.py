from graphrag.cli.main import app

# from graphrag.cli.index import index_cli
from .graph_extractor import GraphExtractor
from .index_builder import build_index
import graphrag.index.graph.extractors as extractors
import graphrag.api as api
import pandas as pd
import graphrag.index.flows.extract_graph as extract_graph


# overwrite the oob graph extractor to support meta data, datatime only
extractors.GraphExtractor = GraphExtractor
api.build_index = build_index


def _merge_relationships(relationship_dfs) -> pd.DataFrame:
    all_relationships = pd.concat(relationship_dfs, ignore_index=False)
    return (
        all_relationships.groupby(["source", "target"], sort=False)
        .agg(
            {
                "description": list,  # Aggregate "description" as a list
                "source_id": list,  # Aggregate "source_id" as a list
                "weight": "sum",  # Sum the "weight" column
                "edge_name_timestamp": lambda x: ", ".join(
                    x.astype(str)
                ),  # Concatenate "x" with a comma
            }
        )
        .reset_index()
    )


extract_graph._merge_relationships = _merge_relationships


# def mian_index(root):
#     index_cli(
#         root_dir=root,
#         verbose=False,
#         resume=resume,
#         memprofile=memprofile,
#         cache=cache,
#         logger=LoggerType(logger),
#         config_filepath=config,
#         dry_run=dry_run,
#         skip_validation=skip_validation,
#         output_dir=output,
#     )


if __name__ == "__main__":
    app(prog_name="graphrag")
