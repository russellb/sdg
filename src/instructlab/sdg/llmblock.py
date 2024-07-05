# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict
import re

# Third Party
from datasets import Dataset

# Local
from .block import Block
from .logger_config import setup_logger

logger = setup_logger(__name__)

MODEL_FAMILY_MIXTRAL = "mixtral"
MODEL_FAMILY_MERLINITE = "merlinite"

_MODEL_PROMPT_MIXTRAL = "<s> [INST] {prompt} [/INST]"
_MODEL_PROMPT_MERLINITE = "'<|system|>\nYou are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.\n<|user|>\n{prompt}\n<|assistant|>\n'"

_MODEL_PROMPTS = {
    MODEL_FAMILY_MIXTRAL: _MODEL_PROMPT_MIXTRAL,
    MODEL_FAMILY_MERLINITE: _MODEL_PROMPT_MERLINITE,
}


def _get_model_prompt(model_family):
    if model_family not in _MODEL_PROMPTS:
        raise ValueError(f"Unknown model family: {model_family}")
    return _MODEL_PROMPTS[model_family]


# pylint: disable=dangerous-default-value
class LLMBlock(Block):
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        ctx,
        block_name,
        config_path,
        output_cols,
        parser_kwargs={},
        **batch_kwargs,
    ) -> None:
        super().__init__(ctx, block_name)
        self.block_config = self._load_config(config_path)
        self.prompt_struct = (
            """{system}\n{introduction}\n{principles}\n{examples}\n{generation}"""
        )
        self.prompt_template = self.prompt_struct.format(**self.block_config)
        self.model_prompt = _get_model_prompt(self.ctx.model_family)
        self.output_cols = output_cols
        self.batch_params = batch_kwargs.get("batch_kwargs", {})
        self.batch_params["batched"] = self.ctx.batched
        self.batch_params["num_procs"] = self.ctx.num_procs
        self.parser_name = parser_kwargs.get("parser_name", None)
        self.parsing_pattern = parser_kwargs.get("parsing_pattern", None)
        self.parser_cleanup_tags = parser_kwargs.get("parser_cleanup_tags", None)
        self.defaults = {
            "model": self.ctx.model_id,
            "temperature": 0,
            "max_tokens": 12000,
        }

    def _parse(self, generated_string) -> dict:
        matches = {}

        if self.parser_name is not None and self.parser_name == "custom":
            pattern = re.compile(self.parsing_pattern, re.DOTALL)
            all_matches = pattern.findall(generated_string)
            matches = {column_name: [] for column_name in self.output_cols}
            if all_matches and isinstance(all_matches[0], tuple):
                for match in all_matches:
                    for column_name, value in zip(self.output_cols, match):
                        value = value.strip()
                        for clean_tag in self.parser_cleanup_tags:
                            value = value.replace(clean_tag, "")
                        matches[column_name].append(value)
            else:
                matches[self.output_cols[0]] = (
                    [match.strip() for match in all_matches] if all_matches else []
                )
        else:
            for start_tag, end_tag, output_col in zip(
                self.block_config.get("start_tags", []),
                self.block_config.get("end_tags", []),
                self.output_cols,
            ):
                if not start_tag and not end_tag:
                    matches[output_col] = [
                        generated_string.strip() if generated_string else None
                    ]
                else:
                    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
                    all_matches = re.findall(pattern, generated_string, re.DOTALL)
                    matches[output_col] = (
                        [match.strip() for match in all_matches] if all_matches else []
                    )

        return matches

    def _gen_kwargs(self, **gen_kwargs):
        gen_kwargs = {**self.defaults, **gen_kwargs}
        if "max_tokens" in gen_kwargs:
            gen_kwargs["max_tokens"] = int(gen_kwargs["max_tokens"])
        if "temperature" in gen_kwargs:
            gen_kwargs["temperature"] = float(gen_kwargs["temperature"])
        return gen_kwargs

    def _generate(self, samples, **gen_kwargs) -> list:
        prompts = [
            self.model_prompt.format(
                prompt=self.prompt_template.format(**sample).strip()
            )
            for sample in samples
        ]
        response = self.ctx.client.completions.create(
            prompt=prompts,
            **self._gen_kwargs(**gen_kwargs),
        )
        return [choice.text.strip() for choice in response.choices]

    def generate(self, samples, **gen_kwargs) -> Dataset:
        """
        Generate the output from the block. This method should first validate the input data,
        then generate the output, and finally parse the generated output before returning it.

        :return: The parsed output after generation.
        """
        num_samples = self.batch_params.get("num_samples", None)
        batched = self.batch_params.get("batched", False)
        logger.debug("Generating outputs for {} samples".format(len(samples)))

        if (num_samples is not None) and ("num_samples" not in samples.column_names):
            samples = samples.add_column("num_samples", [num_samples] * len(samples))

        # validate each sample
        for sample in samples:
            if not self._validate(self.prompt_template, sample):
                return None

        # generate the output
        outputs = []
        if batched:
            outputs = self._generate(samples, **gen_kwargs)
        else:
            outputs = [self._generate([sample], **gen_kwargs)[0] for sample in samples]
        logger.debug("Generated outputs: {}".format(outputs))

        new_data = []
        for sample, output in zip(samples, outputs):
            parsed_outputs = self._parse(output)
            # pylint: disable=consider-using-generator
            max_length = max([len(value) for value in parsed_outputs.values()])
            for values in zip(*(lst[:max_length] for lst in parsed_outputs.values())):
                new_data.append({**sample, **dict(zip(parsed_outputs.keys(), values))})

        return Dataset.from_list(new_data)


class ConditionalLLMBlock(LLMBlock):
    def __init__(
        self,
        ctx,
        block_name,
        config_paths,
        output_cols,
        selector_column_name,
        parser_kwargs={},
        **batch_kwargs,
    ) -> None:
        super().__init__(
            ctx,
            block_name,
            config_paths[0][0],
            output_cols,
            parser_kwargs=parser_kwargs,
            **batch_kwargs,
        )
        self.selector_column_name = selector_column_name
        self.prompt_template = {}
        if len(config_paths) == 1 and config_paths[0][1] == "All":
            self.prompt_template = self.prompt_struct.format(**self.block_config)
        else:
            for config, config_key in config_paths:
                self.prompt_template[config_key] = self.prompt_struct.format(
                    **self._load_config(config)
                )

    def _generate(self, samples, **gen_kwargs) -> str:
        if isinstance(self.prompt_template, dict):
            prompts = [
                self.model_prompt.format(
                    prompt=self.prompt_template[sample[self.selector_column_name]]
                    .format(**sample)
                    .strip()
                )
                for sample in samples
            ]
        else:
            prompts = [
                self.model_prompt.format(
                    prompt=self.prompt_template.format(**sample).strip()
                )
                for sample in samples
            ]
        response = self.ctx.client.completions.create(
            prompt=prompts,
            **self._gen_kwargs(**gen_kwargs),
        )
        return [choice.text.strip() for choice in response.choices]

    def validate(self, prompt_template: str, input_dict: Dict[str, Any]) -> bool:
        if isinstance(prompt_template, dict):
            prompt_template = prompt_template[input_dict[self.selector_column_name]]
        return super()._validate(prompt_template, input_dict)
