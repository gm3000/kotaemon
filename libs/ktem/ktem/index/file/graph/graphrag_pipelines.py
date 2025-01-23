import os
import shutil
import subprocess
from pathlib import Path
from typing import Generator
from uuid import uuid4

import ast
import tiktoken
from decouple import config
from ktem.db.models import engine
from sqlalchemy.orm import Session
from theflow.settings import settings

from kotaemon.base import Document, Param, RetrievedDocument

from ..pipelines import BaseFileIndexRetriever, IndexDocumentPipeline, IndexPipeline
from .visualize import create_knowledge_graph, visualize_graph
from .graphrag.graph_db import create_graph_db
from .graphrag.prompts import entityextraction
from .graphrag.context_gdb_builder import LocalGraphDBSearchMixedContext

try:
    import tiktoken
    from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
    from graphrag.query.llm.oai.embedding import OpenAIEmbedding
    from graphrag.query.llm.oai.typing import OpenaiApiType
    from graphrag.vector_stores.lancedb import LanceDBVectorStore
    import kuzu
except ImportError:
    print(
        (
            "GraphRAG with graph db dependencies not installed. "
            "Try `pip install graphrag kuzu future` to install. "
            "GraphRAG retriever pipeline will not work properly."
        )
    )


filestorage_path = Path(settings.KH_FILESTORAGE_PATH) / "graphrag"
filestorage_path.mkdir(parents=True, exist_ok=True)

GRAPHRAG_KEY_MISSING_MESSAGE = (
    "GRAPHRAG_API_KEY is not set. Please set it to use the GraphRAG retriever pipeline."
)


def check_graphrag_api_key():
    return len(os.getenv("GRAPHRAG_API_KEY", "")) > 0


def prepare_graph_index_path(graph_id: str):
    root_path = Path(filestorage_path) / graph_id
    input_path = root_path / "input"

    return root_path, input_path


class GraphRAGIndexingPipeline(IndexDocumentPipeline):
    """GraphRAG specific indexing pipeline"""

    def route(self, file_path: str | Path) -> IndexPipeline:
        """Simply disable the splitter (chunking) for this pipeline"""
        pipeline = super().route(file_path)
        pipeline.splitter = None

        return pipeline

    def store_file_id_with_graph_id(self, file_ids: list[str | None]):
        # create new graph_id and assign them to doc_id in self.Index
        # record in the index
        graph_id = str(uuid4())
        with Session(engine) as session:
            nodes = []
            for file_id in file_ids:
                if not file_id:
                    continue
                nodes.append(
                    self.Index(
                        source_id=file_id,
                        target_id=graph_id,
                        relation_type="graph",
                    )
                )

            session.add_all(nodes)
            session.commit()

        return graph_id

    def write_docs_to_files(self, graph_id: str, docs: list[Document]):
        root_path, input_path = prepare_graph_index_path(graph_id)
        input_path.mkdir(parents=True, exist_ok=True)

        for doc in docs:
            if doc.metadata.get("type", "text") == "text":
                with open(input_path / f"{doc.doc_id}.txt", "w") as f:
                    f.write(doc.text)

        return root_path

    def call_graphrag_index(self, graph_id: str, all_docs: list[Document]):
        if not check_graphrag_api_key():
            raise ValueError(GRAPHRAG_KEY_MISSING_MESSAGE)

        # call GraphRAG index with docs and graph_id
        input_path = self.write_docs_to_files(graph_id, all_docs)
        input_path = str(input_path.absolute())

        # Construct the command
        command_init = [
            "graphrag",
            "init",
            "--root",
            input_path,
        ]

        # Run the command_init
        yield Document(
            channel="debug",
            text="[GraphRAG] Creating index... This can take a long time.",
        )
        result = subprocess.run(command_init, capture_output=True, text=True)
        print(result.stdout)
        # TODO: tobe removed, use customized entity extraction prompt
        with open(f"{input_path}/prompts/entity_extraction.txt", "w") as file:
            file.write(entityextraction)

        # run graphrag index as sub moodule because the hacks that I made
        command = [
            "python",
            "-m",
            "ktem.index.file.graph.graphrag.run_graphrag",
            "index",
            "--root",
            input_path,
        ]

        # copy customized GraphRAG config file if it exists
        if config("USE_CUSTOMIZED_GRAPHRAG_SETTING", default="value").lower() == "true":
            setting_file_path = os.path.join(os.getcwd(), "settings.yaml.example")
            destination_file_path = os.path.join(input_path, "settings.yaml")
            try:
                shutil.copy(setting_file_path, destination_file_path)
            except shutil.Error:
                # Handle the error if the file copy fails
                print("failed to copy customized GraphRAG config file. ")

        # Run the command and stream stdout
        with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as process:
            if process.stdout:
                for line in process.stdout:
                    yield Document(channel="debug", text=line)

        # import graphrag artifact into graphdb
        create_graph_db(input_path, all_docs)

    def stream(
        self, file_paths: str | Path | list[str | Path], reindex: bool = False, **kwargs
    ) -> Generator[
        Document, None, tuple[list[str | None], list[str | None], list[Document]]
    ]:
        file_ids, errors, all_docs = yield from super().stream(
            file_paths, reindex=reindex, **kwargs
        )

        # assign graph_id to file_ids
        graph_id = self.store_file_id_with_graph_id(file_ids)
        # call GraphRAG index with docs and graph_id
        yield from self.call_graphrag_index(graph_id, all_docs)

        return file_ids, errors, all_docs


class GraphRAGRetrieverPipeline(BaseFileIndexRetriever):
    """GraphRAG specific retriever pipeline"""

    Index = Param(help="The SQLAlchemy Index table")
    file_ids: list[str] = []

    @classmethod
    def get_user_settings(cls) -> dict:
        return {
            "search_type": {
                "name": "Search type",
                "value": "local",
                "choices": ["local"],
                "component": "dropdown",
                "info": "Whether to use local or global search in the graph.",
            }
        }

    def _build_graph_search(self):
        assert (
            len(self.file_ids) <= 1
        ), "GraphRAG retriever only supports one file_id at a time"

        file_id = self.file_ids[0]
        # retrieve the graph_id from the index
        with Session(engine) as session:
            graph_id = (
                session.query(self.Index.target_id)
                .filter(self.Index.source_id == file_id)
                .filter(self.Index.relation_type == "graph")
                .first()
            )
            graph_id = graph_id[0] if graph_id else None
            assert graph_id, f"GraphRAG index not found for file_id: {file_id}"

        root_path, _ = prepare_graph_index_path(graph_id)
        LANCEDB_URI = f"{root_path}/output/lancedb"
        graph_db_uri = root_path / "graph_db"

        # load graph database
        db = kuzu.Database(graph_db_uri, read_only=True)
        conn = kuzu.Connection(db)

        # load description embeddings to an in-memory lancedb vectorstore
        # to connect to a remote db, specify url and port values.
        description_embedding_store = LanceDBVectorStore(
            collection_name="default-entity-description",
        )
        description_embedding_store.connect(db_uri=LANCEDB_URI)

        # initialize default settings
        embedding_model = os.getenv(
            "GRAPHRAG_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        embedding_api_key = os.getenv("GRAPHRAG_API_KEY")
        embedding_api_base = os.getenv("GRAPHRAG_API_BASE")

        # # use customized GraphRAG settings if the flag is set
        # if config("USE_CUSTOMIZED_GRAPHRAG_SETTING", default="value").lower() == "true":
        #     settings_yaml_path = Path(root_path) / "settings.yaml"
        #     with open(settings_yaml_path, "r") as f:
        #         settings = yaml.safe_load(f)
        #     if settings["embeddings"]["llm"]["model"]:
        #         embedding_model = settings["embeddings"]["llm"]["model"]
        #     if settings["embeddings"]["llm"]["api_key"]:
        #         embedding_api_key = settings["embeddings"]["llm"]["api_key"]
        #     if settings["embeddings"]["llm"]["api_base"]:
        #         embedding_api_base = settings["llm"]["api_base"]

        text_embedder = OpenAIEmbedding(
            api_key=embedding_api_key,
            api_base=embedding_api_base,
            api_type=OpenaiApiType.OpenAI,
            model=embedding_model,
            deployment_name=embedding_model,
            max_retries=20,
        )
        token_encoder = tiktoken.get_encoding("cl100k_base")

        graph_context_builder = LocalGraphDBSearchMixedContext(
            conn=conn,
            entity_text_embeddings=description_embedding_store,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,  # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
            text_embedder=text_embedder,
            token_encoder=token_encoder,
        )
        return graph_context_builder, conn

    def _to_document(self, header: str, context_text: str) -> RetrievedDocument:
        return RetrievedDocument(
            text=context_text,
            metadata={
                "file_name": header,
                "type": "table",
                "llm_trulens_score": 1.0,
            },
            score=1.0,
        )

    def format_context_records(
        self, context_records, conn: kuzu.Connection
    ) -> list[RetrievedDocument]:
        entities = context_records.get("entities", [])
        relationships = context_records.get("relationships", [])
        reports = context_records.get("reports", [])
        sources = context_records.get("sources", [])

        docs = []

        context: str = ""

        header = "<b>Entities</b>\n"
        context = entities[["entity", "description"]].to_markdown(index=False)
        docs.append(self._to_document(header, context))

        header = "\n<b>Relationships</b>\n"
        context = relationships[["source", "target", "description"]].to_markdown(
            index=False
        )
        docs.append(self._to_document(header, context))

        header = "\n<b>Reports</b>\n"
        context = ""
        for idx, row in reports.iterrows():
            title, content = row["title"], row["content"]
            context += f"\n\n<h5>Report <b>{title}</b></h5>\n"
            context += content
        docs.append(self._to_document(header, context))

        # we will trace back to original files
        for idx, row in sources.iterrows():
            title, content, document_id = (
                row["id"],
                row["text"],
                ast.literal_eval(row["document_ids"]),
            )
            context = f"\n\n<h5>Source <b>#{title}</b></h5>\n"
            context += content
            # search document - file chunk data from conn
            cypher = """
                MATCH (d:Document)-[:FROM_FILE_CHUNK]->(fc:FileChunk)<-[:HAS_FILE_CHUNK]-(f:File)
                WHERE d.id IN $doc_ids
                RETURN fc.*, f.*
            """
            db_query_result = conn.execute(cypher, {"doc_ids": document_id[:1]})
            result_df = db_query_result.get_as_df()
            # only get first
            first_row = result_df.iloc[0]
            docs.append(
                RetrievedDocument(
                    text=context,
                    metadata={
                        "file_name": first_row["f.file_name"],
                        "type": first_row["f.file_type"],
                        "file_type": first_row["f.file_type"],
                        "file_path": first_row["f.file_path"],
                        "page_label": first_row["fc.page_label"],
                        "llm_trulens_score": 1.0,
                    },
                    score=1.0,
                )
            )
        # docs.append(self._to_document(header, context))

        return docs

    def plot_graph(self, context_records):
        relationships = context_records.get("relationships", [])
        G = create_knowledge_graph(relationships)
        plot = visualize_graph(G)
        return plot

    def generate_relevant_scores(self, text, documents: list[RetrievedDocument]):
        return documents

    def run(
        self,
        text: str,
    ) -> list[RetrievedDocument]:
        if not self.file_ids:
            return []

        if not check_graphrag_api_key():
            raise ValueError(GRAPHRAG_KEY_MISSING_MESSAGE)

        context_builder, conn = self._build_graph_search()

        local_context_params = {
            "text_unit_prop": 0.5,
            "community_prop": 0.1,
            "conversation_history_max_turns": 5,
            "conversation_history_user_turns_only": True,
            "top_k_mapped_entities": 10,
            "top_k_relationships": 10,
            "include_entity_rank": False,
            "include_relationship_weight": False,
            "include_community_rank": False,
            "return_candidate_context": False,
            "embedding_vectorstore_key": EntityVectorStoreKey.ID,
            # set this to EntityVectorStoreKey.TITLE i
            # f the vectorstore uses entity title as ids
            "max_tokens": 12_000,
            # change this based on the token limit you have on your model
            # (if you are using a model with 8k limit, a good setting could be 5000)
        }

        result = context_builder.build_context(
            query=text,
            conversation_history=None,
            **local_context_params,
        )
        context_records = result.context_records
        documents = self.format_context_records(context_records, conn)
        plot = self.plot_graph(context_records)

        return documents + [
            RetrievedDocument(
                text="",
                metadata={
                    "file_name": "GraphRAG",
                    "type": "plot",
                    "data": plot,
                },
            ),
        ]
