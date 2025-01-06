import logging
from copy import deepcopy
from typing import Any, List

import pandas as pd
import tiktoken

from graphrag.model.entity import Entity
from graphrag.query.context_builder.builders import ContextBuilderResult
from graphrag.query.context_builder.community_context import (
    build_community_context,
)
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)
from graphrag.query.context_builder.entity_extraction import (
    EntityVectorStoreKey,
)
from graphrag.query.context_builder.local_context import (
    build_covariates_context,
    build_entity_context,
    build_relationship_context,
    get_candidate_context,
)
from graphrag.query.context_builder.source_context import (
    build_text_unit_context,
    count_relationships,
)
from graphrag.query.input.retrieval.community_reports import (
    get_candidate_communities,
)
from graphrag.query.input.retrieval.text_units import get_candidate_text_units
from graphrag.query.llm.base import BaseTextEmbedding
from graphrag.query.llm.text_utils import num_tokens
from graphrag.query.structured_search.base import LocalContextBuilder
from graphrag.vector_stores.base import BaseVectorStore

from graphrag.query.input.loaders.dfs import (
    read_community_reports,
    read_relationships,
    read_entities,
    read_text_units,
)

from kuzu import Connection

log = logging.getLogger(__name__)


class LocalGraphDBSearchMixedContext(LocalContextBuilder):
    def __init__(
        self,
        conn: Connection,
        entity_text_embeddings: BaseVectorStore,
        text_embedder: BaseTextEmbedding,
        token_encoder: tiktoken.Encoding | None = None,
        embedding_vectorstore_key: str = EntityVectorStoreKey.ID,
    ):
        self.conn = conn
        self.entity_text_embeddings = entity_text_embeddings
        self.text_embedder = text_embedder
        self.token_encoder = token_encoder
        self.embedding_vectorstore_key = embedding_vectorstore_key
        self.covariates = []

    def build_context(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        include_entity_names: list[str] | None = None,
        exclude_entity_names: list[str] | None = None,
        conversation_history_max_turns: int | None = 5,
        conversation_history_user_turns_only: bool = True,
        max_tokens: int = 8000,
        text_unit_prop: float = 0.5,
        community_prop: float = 0.25,
        top_k_mapped_entities: int = 10,
        top_k_relationships: int = 10,
        include_community_rank: bool = False,
        include_entity_rank: bool = False,
        rank_description: str = "number of relationships",
        include_relationship_weight: bool = False,
        relationship_ranking_attribute: str = "rank",
        return_candidate_context: bool = False,
        use_community_summary: bool = False,
        min_community_rank: int = 0,
        community_context_name: str = "Reports",
        column_delimiter: str = "|",
        **kwargs: dict[str, Any],
    ) -> ContextBuilderResult:
        """
        Build data context for local search prompt.

        Build a context by combining community reports and entity/relationship/covariate tables, and text units using a predefined ratio set by summary_prop.
        """
        if include_entity_names is None:
            include_entity_names = []
        if exclude_entity_names is None:
            exclude_entity_names = []
        if community_prop + text_unit_prop > 1:
            value_error = (
                "The sum of community_prop and text_unit_prop should not exceed 1."
            )
            raise ValueError(value_error)

        # map user query to entities
        # if there is conversation history, attached the previous user questions to the current query
        if conversation_history:
            pre_user_questions = "\n".join(
                conversation_history.get_user_turns(conversation_history_max_turns)
            )
            query = f"{query}\n{pre_user_questions}"

        # TODO: all_entities_dict are all entities under community level
        selected_entities = self._map_query_to_entities(
            query=query,
            text_embedding_vectorstore=self.entity_text_embeddings,
            text_embedder=self.text_embedder,
            embedding_vectorstore_key=self.embedding_vectorstore_key,
            include_entity_names=include_entity_names,
            exclude_entity_names=exclude_entity_names,
            k=top_k_mapped_entities,
            oversample_scaler=2,
        )

        # build context
        final_context = list[str]()
        final_context_data = dict[str, pd.DataFrame]()

        if conversation_history:
            # build conversation history context
            (
                conversation_history_context,
                conversation_history_context_data,
            ) = conversation_history.build_context(
                include_user_turns_only=conversation_history_user_turns_only,
                max_qa_turns=conversation_history_max_turns,
                column_delimiter=column_delimiter,
                max_tokens=max_tokens,
                recency_bias=False,
            )
            if conversation_history_context.strip() != "":
                final_context.append(conversation_history_context)
                final_context_data = conversation_history_context_data
                max_tokens = max_tokens - num_tokens(
                    conversation_history_context, self.token_encoder
                )

        # build community context
        community_tokens = max(int(max_tokens * community_prop), 0)
        community_context, community_context_data = self._build_community_context(
            selected_entities=selected_entities,
            max_tokens=community_tokens,
            use_community_summary=use_community_summary,
            column_delimiter=column_delimiter,
            include_community_rank=include_community_rank,
            min_community_rank=min_community_rank,
            return_candidate_context=return_candidate_context,
            context_name=community_context_name,
        )
        if community_context.strip() != "":
            final_context.append(community_context)
            final_context_data = {**final_context_data, **community_context_data}

        # build local (i.e. entity-relationship-covariate) context
        local_prop = 1 - community_prop - text_unit_prop
        local_tokens = max(int(max_tokens * local_prop), 0)
        local_context, local_context_data = self._build_local_context(
            selected_entities=selected_entities,
            max_tokens=local_tokens,
            include_entity_rank=include_entity_rank,
            rank_description=rank_description,
            include_relationship_weight=include_relationship_weight,
            top_k_relationships=top_k_relationships,
            relationship_ranking_attribute=relationship_ranking_attribute,
            return_candidate_context=return_candidate_context,
            column_delimiter=column_delimiter,
        )
        if local_context.strip() != "":
            final_context.append(str(local_context))
            final_context_data = {**final_context_data, **local_context_data}

        text_unit_tokens = max(int(max_tokens * text_unit_prop), 0)
        text_unit_context, text_unit_context_data = self._build_text_unit_context(
            selected_entities=selected_entities,
            max_tokens=text_unit_tokens,
            return_candidate_context=return_candidate_context,
        )

        if text_unit_context.strip() != "":
            final_context.append(text_unit_context)
            final_context_data = {**final_context_data, **text_unit_context_data}

        return ContextBuilderResult(
            context_chunks="\n\n".join(final_context),
            context_records=final_context_data,
        )

    def _map_query_to_entities(
        self,
        query: str,
        text_embedding_vectorstore: BaseVectorStore,
        text_embedder: BaseTextEmbedding,
        include_entity_names: list[str] | None = None,
        exclude_entity_names: list[str] | None = None,
        k: int = 10,
        oversample_scaler: int = 2,
        embedding_vectorstore_key: str = EntityVectorStoreKey.ID,
    ) -> list[Entity]:
        """Extract entities that match a given query using semantic similarity of text embeddings of query and entity descriptions."""
        if include_entity_names is None:
            include_entity_names = []
        if exclude_entity_names is None:
            exclude_entity_names = []

        def query_entity(cypher, params):
            db_query_result = self.conn.execute(cypher, params)
            entity_df = db_query_result.get_as_df()
            # Create a dictionary for renaming r.* columns
            rename_dict = {
                col: col.replace("e.", "")
                for col in entity_df.columns
                if col.startswith("e.")
            }
            # Apply the renaming
            entity_df = entity_df.rename(columns=rename_dict)
            return read_entities(entity_df)

        matched_entities = []
        if query != "":
            # get entities with highest semantic similarity to query
            # oversample to account for excluded entities
            search_results = text_embedding_vectorstore.similarity_search_by_text(
                text=query,
                text_embedder=lambda t: text_embedder.embed(t),
                k=k * oversample_scaler,
            )
            entity_ids = [result.document.id for result in search_results]

            cypher_q = """
                MATCH (e:Entity)
                WHERE e.id IN $entity_ids
                RETURN e.*
            """
            matched_entities = query_entity(cypher_q, {"entity_ids": entity_ids})
        else:
            matched_entities = query_entity(
                """
                MATCH (e:Entity)
                RETURN e.*
                ORDER BY e.degree DESC
                LIMIT $k
                """,
                {"k": k},
            )

        # filter out excluded entities
        if exclude_entity_names:
            matched_entities = [
                entity
                for entity in matched_entities
                if entity.title not in exclude_entity_names
            ]

        # add entities in the include_entity list
        included_entities = []
        if len(include_entity_names) > 0:
            included_entities = query_entity(
                """
                    MATCH (e:Entity)
                    WHERE e.title IN $names
                    RETURN e.*
                """,
                {"names": include_entity_names},
            )
        return included_entities + matched_entities

    def _build_community_context(
        self,
        selected_entities: list[Entity],
        max_tokens: int = 4000,
        use_community_summary: bool = False,
        column_delimiter: str = "|",
        include_community_rank: bool = False,
        min_community_rank: int = 0,
        return_candidate_context: bool = False,
        context_name: str = "Reports",
    ) -> tuple[str, dict[str, pd.DataFrame]]:
        """Add community data to the context window until it hits the max_tokens limit."""
        if len(selected_entities) == 0:
            return ("", {context_name.lower(): pd.DataFrame()})

        # community_matches = {}
        # for entity in selected_entities:
        #     # increase count of the community that this entity belongs to
        #     if entity.community_ids:
        #         for community_id in entity.community_ids:
        #             community_matches[community_id] = (
        #                 community_matches.get(community_id, 0) + 1
        #             )

        # # sort communities by number of matched entities and rank
        # selected_communities = [
        #     self.community_reports[community_id]
        #     for community_id in community_matches
        #     if community_id in self.community_reports
        # ]
        # for community in selected_communities:
        #     if community.attributes is None:
        #         community.attributes = {}
        #     community.attributes["matches"] = community_matches[community.id]

        # selected_communities.sort(
        #     key=lambda x: (x.attributes["matches"], x.rank),  # type: ignore
        #     reverse=True,  # type: ignore
        # )
        # for community in selected_communities:
        #     del community.attributes["matches"]  # type: ignore

        cummunity_report_cypher = """
        MATCH (e:Entity)-[:IN_COMMUNITY]->(c:Community)-[:HAS_REPORT]->(r:CommunityReport)
        WHERE e.id IN $entity_ids
        RETURN e.*, c.*, r.*
        """
        db_query_result = self.conn.execute(
            cummunity_report_cypher, {"entity_ids": [e.id for e in selected_entities]}
        )
        cummunity_report_df = db_query_result.get_as_df()
        # sort by entity count and community rank, reverse
        # Get all columns that start with 'r.' from community report
        r_columns = [col for col in cummunity_report_df.columns if col.startswith("r.")]

        # Create aggregation dictionary
        agg_dict = {"e.id": "count"}
        # Add all r.* columns to take their first value
        for col in r_columns:
            agg_dict[col] = "first"

        result = (
            cummunity_report_df.groupby("c.community")
            .agg(agg_dict)
            .reset_index()
            .sort_values(by=["e.id", "r.rank"], ascending=[False, False])
        )

        # Optionally rename the count column
        result = result.rename(columns={"e.id": "entity_count"})
        # Create a dictionary for renaming r.* columns
        rename_dict = {
            col: col.replace("r.", "") for col in result.columns if col.startswith("r.")
        }
        # Apply the renaming
        result = result.rename(columns=rename_dict)
        # create list of CommunityReport
        selected_communities = read_community_reports(result)

        context_text, context_data = build_community_context(
            community_reports=selected_communities,
            token_encoder=self.token_encoder,
            use_community_summary=use_community_summary,
            column_delimiter=column_delimiter,
            shuffle_data=False,
            include_community_rank=include_community_rank,
            min_community_rank=min_community_rank,
            max_tokens=max_tokens,
            single_batch=True,
            context_name=context_name,
        )
        if isinstance(context_text, list) and len(context_text) > 0:
            context_text = "\n\n".join(context_text)

        if return_candidate_context:
            candidate_context_data = get_candidate_communities(
                selected_entities=selected_entities,
                community_reports=selected_communities,
                use_community_summary=use_community_summary,
                include_community_rank=include_community_rank,
            )
            context_key = context_name.lower()
            if context_key not in context_data:
                context_data[context_key] = candidate_context_data
                context_data[context_key]["in_context"] = False
            else:
                if (
                    "id" in candidate_context_data.columns
                    and "id" in context_data[context_key].columns
                ):
                    candidate_context_data["in_context"] = candidate_context_data[
                        "id"
                    ].isin(  # cspell:disable-line
                        context_data[context_key]["id"]
                    )
                    context_data[context_key] = candidate_context_data
                else:
                    context_data[context_key]["in_context"] = True
        return (str(context_text), context_data)

    def _build_local_context(
        self,
        selected_entities: list[Entity],
        max_tokens: int = 8000,
        include_entity_rank: bool = False,
        rank_description: str = "relationship count",
        include_relationship_weight: bool = False,
        top_k_relationships: int = 10,
        relationship_ranking_attribute: str = "rank",
        return_candidate_context: bool = False,
        column_delimiter: str = "|",
    ) -> tuple[str, dict[str, pd.DataFrame]]:
        """Build data context for local search prompt combining entity/relationship/covariate tables."""
        # build entity context
        entity_context, entity_context_data = build_entity_context(
            selected_entities=selected_entities,
            token_encoder=self.token_encoder,
            max_tokens=max_tokens,
            column_delimiter=column_delimiter,
            include_entity_rank=include_entity_rank,
            rank_description=rank_description,
            context_name="Entities",
        )
        entity_tokens = num_tokens(entity_context, self.token_encoder)

        # build relationship-covariate context
        added_entities = []
        final_context = []
        final_context_data = {}

        # gradually add entities and associated metadata to the context until we reach limit
        # query all relationships
        selected_relationships, entity_relation_df = self._query_relationship(
            selected_entities
        )

        for entity in selected_entities:
            current_context = []
            current_context_data = {}
            added_entities.append(entity)

            # build relationship context
            (
                relationship_context,
                relationship_context_data,
            ) = build_relationship_context(
                selected_entities=added_entities,
                relationships=selected_relationships,
                token_encoder=self.token_encoder,
                max_tokens=max_tokens,
                column_delimiter=column_delimiter,
                top_k_relationships=top_k_relationships,
                include_relationship_weight=include_relationship_weight,
                relationship_ranking_attribute=relationship_ranking_attribute,
                context_name="Relationships",
            )
            current_context.append(relationship_context)
            current_context_data["relationships"] = relationship_context_data
            total_tokens = entity_tokens + num_tokens(
                relationship_context, self.token_encoder
            )

            # build covariate context
            # TODO: add covariates to graphdb
            for covariate in self.covariates:
                covariate_context, covariate_context_data = build_covariates_context(
                    selected_entities=added_entities,
                    covariates=self.covariates[covariate],
                    token_encoder=self.token_encoder,
                    max_tokens=max_tokens,
                    column_delimiter=column_delimiter,
                    context_name=covariate,
                )
                total_tokens += num_tokens(covariate_context, self.token_encoder)
                current_context.append(covariate_context)
                current_context_data[covariate.lower()] = covariate_context_data

            if total_tokens > max_tokens:
                log.info("Reached token limit - reverting to previous context state")
                break

            final_context = current_context
            final_context_data = current_context_data

        # attach entity context to final context
        final_context_text = entity_context + "\n\n" + "\n\n".join(final_context)
        final_context_data["entities"] = entity_context_data

        if return_candidate_context:
            # we return all the candidate entities/relationships/covariates (not only those that were fitted into the context window)
            # and add a tag to indicate which records were included in the context window
            # because everything is from graphdb, relationships already complete, only need other related entities based on
            # current selected relationships
            all_related_entities = self._process_entity_relations(entity_relation_df)

            candidate_context_data = get_candidate_context(
                selected_entities=selected_entities,
                entities=all_related_entities,
                relationships=selected_relationships,
                covariates=self.covariates,
                include_entity_rank=include_entity_rank,
                entity_rank_description=rank_description,
                include_relationship_weight=include_relationship_weight,
            )
            for key in candidate_context_data:
                candidate_df = candidate_context_data[key]
                if key not in final_context_data:
                    final_context_data[key] = candidate_df
                    final_context_data[key]["in_context"] = False
                else:
                    in_context_df = final_context_data[key]

                    if "id" in in_context_df.columns and "id" in candidate_df.columns:
                        candidate_df["in_context"] = candidate_df[
                            "id"
                        ].isin(  # cspell:disable-line
                            in_context_df["id"]
                        )
                        final_context_data[key] = candidate_df
                    else:
                        final_context_data[key]["in_context"] = True
        else:
            for key in final_context_data:
                final_context_data[key]["in_context"] = True
        return (final_context_text, final_context_data)

    def _query_relationship(self, selected_entities):
        relationship_cypher = """
            MATCH (e:Entity)-[r:RELATED]->(ee:Entity)
            WHERE e.id IN $entity_ids OR ee.id IN $entity_ids
            RETURN r.*, e.*, ee.*
        """
        db_query_result = self.conn.execute(
            relationship_cypher, {"entity_ids": [e.id for e in selected_entities]}
        )
        entity_relation_df = db_query_result.get_as_df()
        # Create a dictionary for renaming r.* columns
        rename_dict = {
            col: col.replace("r.", "")
            for col in entity_relation_df.columns
            if col.startswith("r.")
        }
        # Apply the renaming
        entity_relation_df = entity_relation_df.rename(columns=rename_dict)
        selected_relationships = read_relationships(entity_relation_df)
        return selected_relationships, entity_relation_df

    def _process_entity_relations(entity_relation_df: pd.DataFrame) -> List[Entity]:
        # Split into two dataframes
        df_1 = entity_relation_df[
            [col for col in entity_relation_df.columns if col.startswith("e.")]
        ].copy()
        df_2 = entity_relation_df[
            [col for col in entity_relation_df.columns if col.startswith("ee.")]
        ].copy()

        # Rename columns to remove the prefix
        df_1.columns = [col.replace("e.", "") for col in df_1.columns]
        df_2.columns = [col.replace("ee.", "") for col in df_2.columns]

        # Read entities from both dataframes
        entities_1 = read_entities(df=df_1)
        entities_2 = read_entities(df=df_2)

        # Merge and deduplicate entities
        all_entities = entities_1 + entities_2
        unique_entities = {entity.id: entity for entity in all_entities}

        return list(unique_entities.values())

    def _build_text_unit_context(
        self,
        selected_entities: list[Entity],
        max_tokens: int = 8000,
        return_candidate_context: bool = False,
        column_delimiter: str = "|",
        context_name: str = "Sources",
    ) -> tuple[str, dict[str, pd.DataFrame]]:
        """Rank matching text units and add them to the context window until it hits the max_tokens limit."""
        if not selected_entities:
            return ("", {context_name.lower(): pd.DataFrame()})
        selected_text_units = []
        text_unit_ids_set = set()

        unit_info_list = []

        # find all text unit related to selected entities
        text_unit_cypher = """
            MATCH (c:Chunk)-[has:HAS_ENTITY]->(e:Entity)
            WHERE e.id IN $entity_ids
            RETURN c.*
        """
        db_query_result = self.conn.execute(
            text_unit_cypher, {"entity_ids": [e.id for e in selected_entities]}
        )
        text_unit_df = db_query_result.get_as_df()
        # Create a dictionary for renaming r.* columns
        rename_dict = {
            col: col.replace("c.", "")
            for col in text_unit_df.columns
            if col.startswith("c.")
        }
        # Apply the renaming
        text_unit_df = text_unit_df.rename(columns=rename_dict)
        related_text_units = read_text_units(text_unit_df, covariates_col=None)
        related_text_units_map = {unit.id: unit for unit in related_text_units}
        relationship_values, _ = self._query_relationship(selected_entities)

        for index, entity in enumerate(selected_entities):
            # get matching relationships
            entity_relationships = [
                rel
                for rel in relationship_values
                if rel.source == entity.title or rel.target == entity.title
            ]

            for text_id in entity.text_unit_ids or []:
                if (
                    text_id not in text_unit_ids_set
                    and text_id in related_text_units_map
                ):
                    selected_unit = deepcopy(related_text_units_map[text_id])
                    num_relationships = count_relationships(
                        entity_relationships, selected_unit
                    )
                    unit_info_list.append((selected_unit, index, num_relationships))

        # sort by entity_order and the number of relationships desc
        unit_info_list.sort(key=lambda x: (x[1], -x[2]))

        selected_text_units = [unit[0] for unit in unit_info_list]

        context_text, context_data = build_text_unit_context(
            text_units=selected_text_units,
            token_encoder=self.token_encoder,
            max_tokens=max_tokens,
            shuffle_data=False,
            context_name=context_name,
            column_delimiter=column_delimiter,
        )

        if return_candidate_context:
            candidate_context_data = get_candidate_text_units(
                selected_entities=selected_entities,
                text_units=related_text_units,
            )
            context_key = context_name.lower()
            if context_key not in context_data:
                candidate_context_data["in_context"] = False
                context_data[context_key] = candidate_context_data
            else:
                if (
                    "id" in candidate_context_data.columns
                    and "id" in context_data[context_key].columns
                ):
                    candidate_context_data["in_context"] = candidate_context_data[
                        "id"
                    ].isin(context_data[context_key]["id"])
                    context_data[context_key] = candidate_context_data
                else:
                    context_data[context_key]["in_context"] = True

        return (str(context_text), context_data)
