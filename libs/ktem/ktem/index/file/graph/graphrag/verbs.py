"""All the steps to transform final relationships."""

from typing import cast

import pandas as pd
from datashaper import (
    Table,
    verb,
)
from datashaper.table_store.types import VerbResult, create_verb_result

from graphrag.storage.pipeline_storage import PipelineStorage
from graphrag.index.operations.compute_edge_combined_degree import (
    compute_edge_combined_degree,
)


# @verb(
#     name="create_final_relationships_multi_version",
#     treats_input_tables_as_immutable=True,
# )
# async def create_final_relationships_multi_version(
#     input: VerbInput,
#     callbacks: VerbCallbacks,
#     runtime_storage: PipelineStorage,
#     **_kwargs: dict,
# ) -> VerbResult:
#     """All the steps to transform final super relationships."""
#     entity_graph = await runtime_storage.get("base_entity_graph")
#     nodes = cast(pd.DataFrame, get_required_input_table(input, "nodes").table)

#     output = create_final_relationships_flow(
#         entity_graph,
#         nodes,
#         callbacks,
#     )

#     return create_verb_result(cast(Table, output))


@verb(
    name="create_final_relationships_multi_version",
    treats_input_tables_as_immutable=True,
)
async def create_final_relationships_multi_version(
    runtime_storage: PipelineStorage,
    **_kwargs: dict,
) -> VerbResult:
    """All the steps to transform final relationships."""
    base_relationship_edges = await runtime_storage.get("base_relationship_edges")
    base_entity_nodes = await runtime_storage.get("base_entity_nodes")

    output = create_final_relationships_flow(base_relationship_edges, base_entity_nodes)

    return create_verb_result(cast("Table", output))


# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""All the steps to transform final relationships."""


def create_final_relationships_flow(
    base_relationship_edges: pd.DataFrame, base_entity_nodes: pd.DataFrame
) -> pd.DataFrame:
    """All the steps to transform final relationships."""
    relationships = base_relationship_edges
    relationships["combined_degree"] = compute_edge_combined_degree(
        relationships,
        base_entity_nodes,
        node_name_column="title",
        node_degree_column="degree",
        edge_source_column="source",
        edge_target_column="target",
    )

    # allow duplicate source + target, with different relationship name
    # name and date meta data are extracted from field edge_name_timestamp in format "name1@date,name2@date"
    # Add new columns for name and date
    relationships["name"] = ""
    relationships["date"] = ""

    # Extract name and date from edge_name_timestamp
    def extract_name_date(row):
        if pd.isna(row["edge_name_timestamp"]):
            return "", ""
        name_date_pairs = row["edge_name_timestamp"].split(",")
        names = []
        dates = []
        for pair in name_date_pairs:
            name, date = pair.split("@")
            names.append(name)
            dates.append(date)
        return names, dates

    relationships[["name", "date"]] = relationships.apply(
        extract_name_date, axis=1, result_type="expand"
    )

    # Explode the DataFrame to add new rows for each name and date
    relationships = relationships.explode(["name", "date"])

    # Reset index after exploding
    relationships = relationships.reset_index(drop=True)

    # Ensure all columns are kept the same
    final_columns = [
        "id",
        "human_readable_id",
        "source",
        "target",
        "description",
        "weight",
        "combined_degree",
        "text_unit_ids",
        "name",
        "date",
    ]

    return relationships.loc[:, final_columns]
