# SPDX-License-Identifier: Apache-2.0

import typing

import instructlab.utils
import openai
import yaml

import instructlab.sdg.utils as sdg_utils

class PipelineException(Exception):
    pass


class SDGPipeline(object):
    def __init__(self, config: typing.List[typing.Dict[str, str]]):
        self.stages = [SDGPipelineStage(stage) for stage in config]

    def run(self, pipeline_ctx: typing.Dict[str, str]):
        for s in self.stages:
            pipeline_ctx = s.run(pipeline_ctx)
        return pipeline_ctx


class SDGPipelineStage(object):
    def __init__(self, config: typing.Dict[str, str]):
        '''Initialize a custom SDG Pipeline Stage.

        config:
          - endpoint_url (required) - OpenAI API compatible endpoint URL
          - model (required) - model name
          - template (required) - Prompt template, jinja2 formatted
          - output (required) - key to set with the output of this stage

          - inputs (optional) - list of input keys to the template, used for validation

          - api_key (optional) - API key for endpoint_url
          - tls_insecure (optional)
          - tls_client_cert (optional)
          - tls_client_key (optional)
          - tls_client_passwd (optional)
        '''
        self.config = config
        required = (
            'endpoint_url',
            'model',
            'output',
            'template'
        )
        missing = [c for c in required if c not in config]
        if len(missing) > 0:
            raise PipelineException(f"Required config item(s) missing: {missing}")

    def run(self, pipeline_ctx: typing.Dict[str, str]):
        '''Run this pipeline stage.

        pipeline_ctx - context of the pipeline
        '''
        missing_inputs = [i for i in self.config.get('inputs', ()) if i not in pipeline_ctx]
        if len(missing_inputs) > 0:
            raise PipelineException(f"Required input(s) missing from pipeline context: {missing_inputs}")

        # TODO - process template, j2, using pipeline_ctx
        prompt = self.config['template']

        decoding_args = sdg_utils.OpenAIDecodingArguments(
            temperature=1.0,
            n=1,
            # Hard-coded to maximize length.
            # Requests will be automatically adjusted.
            max_tokens=3072,
            top_p=1.0,
            stop=["* Task 5"],
        )

        # TODO - handle failures gracefully with retries
        result = sdg_utils.openai_completion(
            api_base=self.config['endpoint_url'],
            tls_insecure=self.config.get('tls_insecure', False),
            tls_client_cert=self.config.get('tls_client_cert'),
            tls_client_key=self.config.get('tls_client_key'),
            tls_client_passwd=self.config.get('tls_client_passwd'),
            prompts=[prompt],
            decoding_args=decoding_args,
            model_name=self.config['model'],
            return_text=True,
            api_key=self.config.get('api_key', 'default_api_key'),
        )
        pipeline_ctx[self.config['output']] = result

        return pipeline_ctx
