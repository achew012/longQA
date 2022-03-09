from clearml import PipelineController, Task

PIPELINE_PROJECT_NAME = "TextExtraction"
PIPELINE_NAME = "DocumentClf-longQA"

DENOISER_TASK_PROJECT_NAME = "DocumentClf"
DENOISER_TASK_NAME = "LED-Denoiser-train"

DATASET_AUDIT_PROJECT_NAME = "DataAudit"


pipe = PipelineController(
    project=PIPELINE_PROJECT_NAME,
    name=PIPELINE_NAME,
    version="1.2",
    add_pipeline_tags=True,
)
pipe.set_default_execution_queue("compute")  # set to queue with GPU

pipe.add_step(
    name="DocumentClf",
    base_task_project="datasets/c4",
    base_task_name="dataset_store_c4_dataset",
)

# pipe.add_step(
#     name="dataset_etl",
#     base_task_project=ETL_TASK_PROJECT_NAME,
#     base_task_name="dataset_load_csv",
# )

denoiser_task = Task.get_task(
    project_name=DENOISER_TASK_PROJECT_NAME,
    task_name=DENOISER_TASK_NAME,
    task_filter={"status": ["published"]},
)
pipe.add_step(
    name="denoiser",
    base_task_id=denoiser_task.id,
    parameter_override={
        # "General/clearml_dataset_project_name": ${dataset_etl.artifacts.data.url}
        "General/clearml_dataset_project_name": ETL_TASK_PROJECT_NAME,
        "General/clearml_dataset_name": "dataset_load_csv",
    },
)

# pipe.add_step(
#     name="dataset_audit",
#     base_task_project=DATASET_AUDIT_PROJECT_NAME,
#     base_task_name="dataset_audit",
#     parameter_override={
#         "General/prediction_path": ${denoiser.artifacts.generated_text.url},
#     },
# )

# Starting the pipeline (in the background)
pipe.start()

print("done")
