import pandas as pd
from typing import cast
import kuzu
import json
import uuid
import numpy as np
from kotaemon.base import Document

# # import everything into graphdb
#
# ## node types
#
# - `__Entity__`
# - `__Document__`
# - `__Chunk__`
# - `__Community__`
# - `__Covariate__`
#


def read_indexer_entities(final_nodes: pd.DataFrame, final_entities: pd.DataFrame):
    """Read in the Entities from the raw indexing outputs."""
    nodes_df = final_nodes
    entities_df = final_entities

    nodes_df = cast(pd.DataFrame, nodes_df[["id", "degree", "community"]])

    nodes_df["community"] = nodes_df["community"].fillna(-1)
    nodes_df["community"] = nodes_df["community"].astype(int)
    nodes_df["degree"] = nodes_df["degree"].astype(int)

    # group entities by id and degree and remove duplicated community IDs
    nodes_df = nodes_df.groupby(["id", "degree"]).agg({"community": set}).reset_index()
    nodes_df["community"] = nodes_df["community"].apply(lambda x: [str(i) for i in x])
    final_df = nodes_df.merge(entities_df, on="id", how="inner").drop_duplicates(
        subset=["id"]
    )

    # todo: pre 1.0 back-compat where title was name
    if "title" not in final_df.columns:
        final_df["title"] = final_df["name"]

    # read entity dataframe to knowledge model objects
    return final_df


def add_source_file_to_db(conn: kuzu.Connection, documents: list[Document]):
    conn.execute(
        "CREATE NODE TABLE FileChunk(id STRING, page_label STRING, text STRING, PRIMARY KEY (id))"
    )
    conn.execute(
        "CREATE NODE TABLE File(id STRING, file_name STRING, file_path STRING, file_type STRING, file_size INT64, creation_date STRING, last_modified_date STRING, collection_name STRING, thumbnail_doc_id STRING, PRIMARY KEY (id))"
    )
    conn.execute("CREATE REL TABLE HAS_FILE_CHUNK(FROM File TO FileChunk)")

    # Initialize lists to store data for each DataFrame
    file_chunk_data = []
    file_data = []
    file_file_chunk_rel_data = []

    # Use a set to keep track of unique file_ids to avoid duplicates in file_df
    unique_file_ids = set()

    # Iterate through the list of Document objects
    for doc in documents:
        # Extract data for file_chunk_df
        file_chunk_data.append(
            {
                "id": doc.doc_id,
                "page_label": doc.metadata["page_label"],
                "text": doc.text,
            }
        )

        # Extract data for file_df (only if file_id is not already processed)
        file_id = doc.metadata["file_id"]
        if file_id not in unique_file_ids:
            unique_file_ids.add(file_id)
            file_data.append(
                {
                    "id": file_id,
                    "file_name": doc.metadata["file_name"],
                    "file_path": doc.metadata["file_path"],
                    "file_type": doc.metadata["file_type"],
                    "file_size": doc.metadata["file_size"],
                    "creation_date": doc.metadata["creation_date"],
                    "last_modified_date": doc.metadata["last_modified_date"],
                    "collection_name": doc.metadata["collection_name"],
                    "thumbnail_doc_id": doc.metadata["thumbnail_doc_id"],
                }
            )

        # Extract data for file_file_chunk_rel_df
        file_file_chunk_rel_data.append({"FROM": file_id, "TO": doc.doc_id})

    # Create DataFrames
    file_chunk_df = pd.DataFrame(file_chunk_data)
    file_df = pd.DataFrame(file_data)
    file_file_chunk_rel_df = pd.DataFrame(file_file_chunk_rel_data)

    print(f"file chunks count: {len(file_chunk_df)}")
    print(f"file count: {len(file_df)}")
    print(f"file file chunk relation count: {len(file_file_chunk_rel_df)}")

    # add data to graph database
    print("COPY FileChunk from file_chunk_df")
    conn.execute("COPY FileChunk from file_chunk_df")
    print("COPY File from file_df")
    conn.execute("COPY File from file_df")
    print("COPY HAS_FILE_CHUNK from file_file_chunk_rel_df")
    conn.execute("COPY HAS_FILE_CHUNK from file_file_chunk_rel_df")

    return file_chunk_df


def create_graph_db(path: str, docs: list[Document]):
    GRAPHRAG_FOLDER = f"{path}/output"

    final_entity_file = f"{GRAPHRAG_FOLDER}/create_final_entities.parquet"
    final_relationships_file = (
        f"{GRAPHRAG_FOLDER}/create_final_relationships_multi_version.parquet"
    )
    final_chunk_file = f"{GRAPHRAG_FOLDER}/create_final_text_units.parquet"
    final_documents_file = f"{GRAPHRAG_FOLDER}/create_final_documents.parquet"
    final_community_file = f"{GRAPHRAG_FOLDER}/create_final_communities.parquet"
    final_community_report_file = (
        f"{GRAPHRAG_FOLDER}/create_final_community_reports.parquet"
    )
    final_nodes_file = f"{GRAPHRAG_FOLDER}/create_final_nodes.parquet"

    db = kuzu.Database(f"{path}/graph_db")
    conn = kuzu.Connection(db)

    ## nodes
    conn.execute(
        "CREATE NODE TABLE Entity(id STRING, degree INT64, community STRING[], human_readable_id INT64, title STRING, type STRING, description STRING, text_unit_ids STRING[], PRIMARY KEY (id))"
    )
    conn.execute(
        "CREATE NODE TABLE Document(id STRING, human_readable_id INT64, title STRING, text STRING, text_unit_ids STRING[], PRIMARY KEY (id))"
    )
    conn.execute(
        "CREATE NODE TABLE Chunk(id STRING, human_readable_id INT64, text STRING, n_tokens INT64, document_ids STRING[], entity_ids STRING[], relationship_ids STRING[], PRIMARY KEY (id))"
    )
    conn.execute(
        "CREATE NODE TABLE Community(id STRING, human_readable_id INT64, community INT64, parent INT64, level INT64, title STRING, entity_ids STRING[], relationship_ids STRING[], text_unit_ids STRING[], period DATE, size INT64, PRIMARY KEY (id))"
    )
    conn.execute(
        "CREATE NODE TABLE CommunityReport(id STRING, human_readable_id INT64, community INT64, parent INT64, level INT64, title STRING, summary STRING, full_content STRING, rank FLOAT, rank_explanation STRING, findings STRING, full_content_json STRING, period DATE, size INT64, PRIMARY KEY (id))"
    )
    conn.execute(
        "CREATE NODE TABLE Finding(id STRING, community STRING, explanation STRING, summary STRING, PRIMARY KEY (id))"
    )

    ## edges
    ### entity edges
    conn.execute(
        "CREATE REL TABLE RELATED(FROM Entity TO Entity, id STRING, human_readable_id INT64, source STRING, target STRING, description STRING, weight FLOAT, combined_degree INT64, text_unit_ids STRING[], name STRING, data DATE)"
    )
    ### chunk -PART_OF-> document
    conn.execute("CREATE REL TABLE PART_OF(FROM Chunk TO Document)")
    ### chunk -HAS_ENTITY-> entity
    conn.execute("CREATE REL TABLE HAS_ENTITY(FROM Chunk TO Entity)")
    ###  community -HAS_CHUNK-> Chunk
    conn.execute("CREATE REL TABLE HAS_CHUNK(FROM Community TO Chunk)")
    ###  entity -IN_COMMUNITY-> Community
    conn.execute("CREATE REL TABLE IN_COMMUNITY(FROM Entity TO Community)")
    ### community -HAS_REPORT-> CommunityReport
    conn.execute("CREATE REL TABLE HAS_REPORT(FROM Community TO CommunityReport)")
    ### community -HAS_FINDING-> Finding
    conn.execute("CREATE REL TABLE HAS_FINDING(FROM Community TO Finding)")

    _final_entity_df = pd.read_parquet(final_entity_file)
    final_relationships_df = pd.read_parquet(final_relationships_file)
    final_chunk_df = pd.read_parquet(final_chunk_file)
    final_documents_df = pd.read_parquet(final_documents_file)
    final_community_df = pd.read_parquet(final_community_file)
    final_community_report_df = pd.read_parquet(final_community_report_file)
    final_nodes_df = pd.read_parquet(final_nodes_file)

    ## post read process in order to support kuzu data type
    final_entity_df = read_indexer_entities(final_nodes_df, _final_entity_df)
    final_entity_df_2 = final_entity_df.copy()
    final_entity_df_2["text_unit_ids"] = final_entity_df_2["text_unit_ids"].apply(
        lambda x: x.tolist() if x is not None else []
    )

    final_relationships_df_2 = final_relationships_df.copy()
    final_relationships_df_2["text_unit_ids"] = final_relationships_df_2[
        "text_unit_ids"
    ].apply(lambda x: x.tolist() if x is not None else [])
    # Convert 'date' column to datetime, treating empty strings as NaT
    final_relationships_df_2["date"] = pd.to_datetime(
        final_relationships_df_2["date"], errors="coerce"
    )
    # Replace NaT (resulting from empty strings) with None
    final_relationships_df_2["date"] = final_relationships_df_2["date"].apply(
        lambda x: None if pd.isna(x) else x
    )
    # Now, the 'date' column is either a valid date or None

    final_chunk_df_2 = final_chunk_df.copy()
    final_chunk_df_2["document_ids"] = final_chunk_df_2["document_ids"].apply(
        lambda x: x.tolist() if x is not None else []
    )
    final_chunk_df_2["entity_ids"] = final_chunk_df_2["entity_ids"].apply(
        lambda x: x.tolist() if x is not None else []
    )
    final_chunk_df_2["relationship_ids"] = final_chunk_df_2["relationship_ids"].apply(
        lambda x: x.tolist() if x is not None else []
    )

    final_documents_df_2 = final_documents_df.copy()
    final_documents_df_2["text_unit_ids"] = final_documents_df_2["text_unit_ids"].apply(
        lambda x: x.tolist() if x is not None else []
    )
    final_documents_df_2 = final_documents_df_2.drop_duplicates(subset=["id"])

    final_community_df_2 = final_community_df.copy()
    final_community_df_2["entity_ids"] = final_community_df_2["entity_ids"].apply(
        lambda x: x.tolist() if x is not None else []
    )
    final_community_df_2["relationship_ids"] = final_community_df_2[
        "relationship_ids"
    ].apply(lambda x: x.tolist() if x is not None else [])
    final_community_df_2["text_unit_ids"] = final_community_df_2["text_unit_ids"].apply(
        lambda x: x.tolist() if x is not None else []
    )

    final_community_report_df_2 = final_community_report_df.copy()
    final_community_report_df_2["findings"] = final_community_report_df_2[
        "findings"
    ].apply(lambda x: json.dumps(x.tolist() if x is not None else []))

    ## import nodes data directly fromn df
    print("COPY Entity FROM final_entity_df_2")
    conn.execute("COPY Entity FROM final_entity_df_2")
    print("COPY Document FROM final_documents_df_2")
    conn.execute("COPY Document FROM final_documents_df_2")
    print("COPY Chunk FROM final_chunk_df_2")
    conn.execute("COPY Chunk FROM final_chunk_df_2")
    print("COPY Community FROM final_community_df_2")
    conn.execute("COPY Community FROM final_community_df_2")
    print("COPY CommunityReport FROM final_community_report_df_2")
    conn.execute("COPY CommunityReport FROM final_community_report_df_2")

    # Create empty findings dataframe with desired schema
    community_id_map = dict(
        zip(final_community_df["community"], final_community_df["id"])
    )
    findings_df = pd.DataFrame(columns=["id", "community", "explanation", "summary"])

    # Iterate through each row in source dataframe
    for _, row in final_community_report_df.iterrows():
        community = community_id_map.get(row["community"])
        findings = row["findings"]

        # Create new rows for each finding
        new_rows = []
        for finding in findings:
            new_row = {
                "id": str(uuid.uuid4()),
                "community": community,
                "explanation": finding["explanation"],
                "summary": finding["summary"],
            }
            new_rows.append(new_row)

        # Append new rows to findings_df
        findings_df = pd.concat(
            [findings_df, pd.DataFrame(new_rows)], ignore_index=True
        )

    # Set column dtypes
    findings_df = findings_df.astype(
        {
            "id": "string",
            "community": "string",
            "explanation": "string",
            "summary": "string",
        }
    )

    ## import nodes of finding
    conn.execute("COPY Finding from findings_df")

    ## import edges from relation df

    # Create a title to id mapping from entity dataframe
    entity_id_map = dict(zip(final_entity_df["title"], final_entity_df["id"]))

    # Create FROM and TO columns by mapping titles to ids
    final_relationships_df_2.insert(
        0, "FROM", final_relationships_df_2["source"].map(entity_id_map)
    )
    final_relationships_df_2.insert(
        1, "TO", final_relationships_df_2["target"].map(entity_id_map)
    )

    # Import the relationships
    print("COPY RELATED from final_relationships_df_2")
    conn.execute("COPY RELATED from final_relationships_df_2")

    ## import chunk -> document
    # Create chunk_doc_rel_df by exploding document_ids array into separate rows

    chunk_doc_rel_df = pd.DataFrame(
        {
            "FROM": np.repeat(
                final_chunk_df["id"].values, final_chunk_df["document_ids"].apply(len)
            ),
            "TO": np.concatenate(final_chunk_df["document_ids"].values),
        }
    )

    # Ensure column types
    chunk_doc_rel_df = chunk_doc_rel_df.astype({"FROM": "string", "TO": "string"})

    print("COPY PART_OF from chunk_doc_rel_df")
    conn.execute("COPY PART_OF from chunk_doc_rel_df")

    ## import chunk -HAS_ENTITY-> entity
    # Create chunk_entity_rel_df by exploding entity_ids array into separate rows
    chunk_entity_rel_df = pd.DataFrame(
        {
            "FROM": np.repeat(
                final_chunk_df["id"].values, final_chunk_df["entity_ids"].apply(len)
            ),
            "TO": np.concatenate(final_chunk_df["entity_ids"].values),
        }
    )

    # Ensure column types
    chunk_entity_rel_df = chunk_entity_rel_df.astype({"FROM": "string", "TO": "string"})

    print("COPY HAS_ENTITY from chunk_entity_rel_df")
    conn.execute("COPY HAS_ENTITY from chunk_entity_rel_df")

    # Import community -HAS_CHUNK-> Chunk
    # Create empty dataframe to store chunk IDs for each community
    community_chunk_rel_df = pd.DataFrame(
        {
            "FROM": np.repeat(
                final_community_df["id"].values,
                final_community_df["text_unit_ids"].apply(len),
            ),
            "TO": np.concatenate(final_community_df["text_unit_ids"].values),
        }
    )

    # Ensure column types
    community_chunk_rel_df = community_chunk_rel_df.astype(
        {"FROM": "string", "TO": "string"}
    )

    print("COPY HAS_CHUNK from community_chunk_rel_df")
    conn.execute("COPY HAS_CHUNK from community_chunk_rel_df")

    # Import entity -IN_COMMUNITY-> Community
    # Create entity-community relationship dataframe
    entity_community_rel_df = pd.DataFrame(
        {
            "FROM": np.concatenate(final_community_df["entity_ids"].values),
            "TO": np.repeat(
                final_community_df["id"].values,
                final_community_df["entity_ids"].apply(len),
            ),
        }
    )

    # Ensure column types
    entity_community_rel_df = entity_community_rel_df.astype(
        {"FROM": "string", "TO": "string"}
    )

    print("COPY IN_COMMUNITY from entity_community_rel_df")
    conn.execute("COPY IN_COMMUNITY from entity_community_rel_df")

    # Import community -HAS_REPORT-> CommunityReport
    # Merge community and report dataframes on community column
    community_report_rel_df = pd.merge(
        final_community_df[["id", "community"]],
        final_community_report_df[["id", "community"]],
        on="community",
        suffixes=("_community", "_report"),
    )

    # Rename columns to match required format
    community_report_rel_df = community_report_rel_df.rename(
        columns={"id_community": "FROM", "id_report": "TO"}
    )

    # Select only the FROM/TO columns
    community_report_rel_df = community_report_rel_df[["FROM", "TO"]]

    community_report_rel_df = community_report_rel_df.astype(
        {"FROM": "string", "TO": "string"}
    )

    print("COPY HAS_REPORT from community_report_rel_df")
    conn.execute("COPY HAS_REPORT from community_report_rel_df")

    # Import community -HAS_FINDING-> Finding
    community_finding_rel_df = pd.DataFrame(
        {"FROM": findings_df["community"], "TO": findings_df["id"]}
    )

    community_finding_rel_df = community_finding_rel_df.astype(
        {"FROM": "string", "TO": "string"}
    )

    print("COPY HAS_FINDING from community_finding_rel_df")
    conn.execute("COPY HAS_FINDING from community_finding_rel_df")

    # connect graphrag schema with the kotaemone data schema
    file_chunk_df = add_source_file_to_db(conn, docs)
    file_chunk_df["title"] = file_chunk_df["id"] + ".txt"
    document_file_chunk_rel_df = pd.merge(
        final_documents_df_2,
        file_chunk_df,
        on="title",
        how="inner",
        suffixes=("_from", "_to"),
    )
    document_file_chunk_rel_df["FROM"] = document_file_chunk_rel_df["id_from"]
    document_file_chunk_rel_df["TO"] = document_file_chunk_rel_df["id_to"]
    document_file_chunk_rel_df2 = document_file_chunk_rel_df[["FROM", "TO"]]
    print(f"merged document count: {len(document_file_chunk_rel_df2)}")

    conn.execute("CREATE REL TABLE FROM_FILE_CHUNK(FROM Document TO FileChunk)")

    print("COPY FROM_FILE_CHUNK from document_file_chunk_rel_df2")
    conn.execute("COPY FROM_FILE_CHUNK from document_file_chunk_rel_df2")

    return conn
