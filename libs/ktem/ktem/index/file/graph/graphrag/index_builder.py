# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
Indexing API for GraphRAG.

WARNING: This API is under development and may undergo changes in future releases.
Backwards compatibility is not guaranteed at this time.
"""

from datashaper import WorkflowCallbacks

from graphrag.cache.noop_pipeline_cache import NoopPipelineCache
from graphrag.callbacks.factory import create_pipeline_reporter
from graphrag.config.enums import CacheType
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.create_pipeline_config import create_pipeline_config
from graphrag.index.run import run_pipeline_with_config
from graphrag.index.typing import PipelineRunResult
from graphrag.logger.base import ProgressLogger
from graphrag.index.config.workflow import PipelineWorkflowReference


async def build_index(
    config: GraphRagConfig,
    run_id: str = "",
    is_resume_run: bool = False,
    memory_profile: bool = False,
    callbacks: list[WorkflowCallbacks] | None = None,
    progress_logger: ProgressLogger | None = None,
) -> list[PipelineRunResult]:
    """Run the pipeline with the given configuration.

    Parameters
    ----------
    config : GraphRagConfig
        The configuration.
    run_id : str
        The run id. Creates a output directory with this name.
    is_resume_run : bool default=False
        Whether to resume a previous index run.
    memory_profile : bool
        Whether to enable memory profiling.
    callbacks : list[WorkflowCallbacks] | None default=None
        A list of callbacks to register.
    progress_logger : ProgressLogger | None default=None
        The progress logger.

    Returns
    -------
    list[PipelineRunResult]
        The list of pipeline run results
    """
    is_update_run = bool(config.update_index_storage)

    print("build_index called hahahahahahahahah###########")

    if is_resume_run and is_update_run:
        msg = "Cannot resume and update a run at the same time."
        raise ValueError(msg)

    pipeline_config = create_pipeline_config(config)
    pipeline_cache = (
        NoopPipelineCache() if config.cache.type == CacheType.none is None else None
    )

    ### update workflow of graph to support more attributes
    # Find the workflow reference in pipeline_config.workflows which name is "create_base_entity_graph"
    # for workflow in pipeline_config.workflows:
    #     if workflow.name == "create_base_entity_graph":
    #         workflow.config["graph_merge_operations"] = {
    #             "nodes": {
    #                 "source_id": {
    #                     "operation": "concat",
    #                     "delimiter": ", ",
    #                     "distinct": True,
    #                 },
    #                 "description": (
    #                     {
    #                         "operation": "concat",
    #                         "separator": "\n",
    #                         "distinct": False,
    #                     }
    #                 ),
    #             },
    #             "edges": {
    #                 "source_id": {
    #                     "operation": "concat",
    #                     "delimiter": ", ",
    #                     "distinct": True,
    #                 },
    #                 "description": (
    #                     {
    #                         "operation": "concat",
    #                         "separator": "\n",
    #                         "distinct": False,
    #                     }
    #                 ),
    #                 "edge_name_timestamp": {
    #                     "operation": "concat",
    #                     "delimiter": ", ",
    #                     "distinct": True,
    #                 },
    #                 "weight": "sum",
    #             },
    #         }
    #         break

    ## add cusomized workflow to pipline configured workflows
    from .verbs import create_final_relationships_multi_version

    pipeline_config.workflows.append(
        PipelineWorkflowReference(
            name="create_final_relationships_multi_version",
            steps=[
                {
                    "verb": "create_final_relationships_multi_version",
                    "args": {},
                    "input": {"source": "workflow:extract_graph"},
                }
            ],
            config={},
        )
    )

    # create a pipeline reporter and add to any additional callbacks
    # TODO: remove the type ignore once the new config engine has been refactored
    callbacks = callbacks or []
    callbacks.append(create_pipeline_reporter(config.reporting, None))  # type: ignore
    outputs: list[PipelineRunResult] = []
    async for output in run_pipeline_with_config(
        pipeline_config,
        run_id=run_id,
        memory_profile=memory_profile,
        cache=pipeline_cache,
        callbacks=callbacks,
        logger=progress_logger,
        is_resume_run=is_resume_run,
        is_update_run=is_update_run,
    ):
        outputs.append(output)
        if progress_logger:
            if output.errors and len(output.errors) > 0:
                progress_logger.error(output.workflow)
            else:
                progress_logger.success(output.workflow)
            progress_logger.info(str(output.result))
    return outputs
