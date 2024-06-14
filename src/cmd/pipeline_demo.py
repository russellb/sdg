import pprint
import sys

import instructlab.sdg.pipeline as sdg_pipeline


def print_line():
    print('-' * 80)


def test_single_stage(endpoint_url, model):
    '''Demo of a single stage of a pipeline.

    Each stage is represented by a SDGPipelienStage object.

    This object takes some configuration as seen below. In this example we only
    provide the required fields.

    A stage has a run() method that takes a pipeline context as input and returns
    an updated pipeline context. The pipeline context is a Python dictionary.

    Imagine that the template would be something much more extensive, with fields
    to be filled in by the pipeline context. The template is a Jinja2 template.

    When the stage is run, the template is filled in with the pipeline context and
    then sent to the configured model endpoint. The output of the model is stored
    in the pipeline context under the key specified in the 'output' field.
    '''
    cfg = {
        'endpoint_url': endpoint_url,
        'model': model,
        'template': 'Reply with only "single_stage" and no other words.',
        'output': 'test_stage1',
    }
    stage = sdg_pipeline.SDGPipelineStage(cfg)
    print_line()
    print("Running test stage ...")
    result = stage.run({})
    pprint.PrettyPrinter().pprint(result)
    print_line()


def test_sample_pipeline(endpoint_url, model):
    '''Demo of a 3-stage pipeline.

    This pipeline is hardcoded in code, but the same thing could be loaded
    from a configuration file (YAML, for example).

    The execution is very similar to the first example, but now we pass the
    pipeline context through a list of stages.

    An example yaml configuration for this pipeline would be:

    ---
    stages:
        - endpoint_url: http://localhost:8000/v1
            model: models/merlinite-7b-lab-Q4_K_M.gguf
            template: Reply with "stage1" and no other words.
            output: stage1_output

        - endpoint_url: http://localhost:8000/v1
            model: models/merlinite-7b-lab-Q4_K_M.gguf
            template: Reply with "stage2" and no other words.
            output: stage2_output

        - endpoint_url: http://localhost:8000/v1
            model: models/merlinite-7b-lab-Q4_K_M.gguf
            template: Reply with "stage3" and no other words.
            output: stage3_output
    ---

    Embedding the template directly in the config file may be problematic
    as the template becomes larger. An alternative would be to have a
    config item called `template_file` which points to a file with the
    template content.

    While the templating is not yet fully implemented, here is an example
    of how it would work:

    ---
    stages:
        - endpoint_url: http://localhost:8000/v1
            model: models/merlinite-7b-lab-Q4_K_M.gguf
            template: Reply with "stage1" and no other words.
            output: stage1_output

        - endpoint_url: http://localhost:8000/v1
            model: models/merlinite-7b-lab-Q4_K_M.gguf
            template: Reply with only "Stage 1 output was {{ stage1_output }}".
            output: stage2_output
    '''
    pipeline_cfg = [
        {
            'endpoint_url': endpoint_url,
            'model': model,
            'template': 'Reply with "stage1" and no other words.',
            'output': 'stage1_output',
        },
        {
            'endpoint_url': endpoint_url,
            'model': model,
            'template': 'Reply with "stage2" and no other words.',
            'output': 'stage2_output',
        },
        {
            'endpoint_url': endpoint_url,
            'model': model,
            'template': 'Reply with "stage3" and no other words.',
            'output': 'stage3_output',
        },
    ]
    pipeline = sdg_pipeline.SDGPipeline(pipeline_cfg)
    print_line()
    print("Running sample 3-stage pipeline ...")
    result = pipeline.run({})
    pprint.PrettyPrinter().pprint(result)
    print_line()


def main(argv=sys.argv):
    endpoint_url = 'http://localhost:8000/v1'
    model = 'models/merlinite-7b-lab-Q4_K_M.gguf'

    test_single_stage(endpoint_url, model)
    test_sample_pipeline(endpoint_url, model)


if __name__ == '__main__':
    sys.exit(main())
