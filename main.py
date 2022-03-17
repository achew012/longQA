from clearml import PipelineController, Task

PIPELINE_PROJECT_NAME = "TextExtraction"
PIPELINE_NAME = "DocumentClf-longQA-Extraction"

CLF_TASK_PROJECT_NAME = "DocumentClf"
CLF_TASK_NAME = "Classifier_train"

QA_PROJECT_NAME = "LongQA"
QA_TASK_NAME = "LongQA-NER-train"


pipe = PipelineController(
    project=PIPELINE_PROJECT_NAME,
    name=PIPELINE_NAME,
    version="1.2",
    add_pipeline_tags=True,
)
pipe.set_default_execution_queue("compute")  # set to queue with GPU

clf_base_task = Task.get_task(
    project_name=CLF_TASK_PROJECT_NAME,
    task_name=CLF_TASK_NAME,
    task_filter={"status": ["published"]},
)

# qg_base_task = Task.get_task(
#     project_name=CLF_TASK_PROJECT_NAME,
#     task_name=CLF_TASK_NAME,
#     task_filter={"status": ["published"]},
# )

qa_base_task = Task.get_task(
    project_name=QA_PROJECT_NAME,
    task_name=QA_TASK_NAME,
    task_filter={"status": ["published"]},
)

pipe.add_step(
    name="document_clf",
    base_task_id=clf_base_task.id,
)

pipe.add_step(
    name="qaqg",
    base_task_id=qa_base_task.id,
    # parameter_override={
        # "General/clearml_dataset_project_name": ${qg.artifacts.predictions.url}
        # "General/clearml_dataset_project_name": ,
        # "General/clearml_dataset_name": "dataset_load_csv",
        # "General/event_type: ${document_clf.artifacts.data.url}"
    },
)

# Starting the pipeline (in the background)
pipe.start()

print("done")
