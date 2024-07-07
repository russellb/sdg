# SPDX-License-Identifier: Apache-2.0
# Third Party
from datasets import Dataset

# Local
from .block import Block
from .logger_config import setup_logger

logger = setup_logger(__name__)


class SamplePopulatorBlock(Block):
    def __init__(self, ctx, config_paths, column_name, post_fix="") -> None:
        super().__init__(
            ctx, block_name=self.__class__.__name__
        )  # Call the base class's __init__
        self.configs = {}
        for config in config_paths:
            if post_fix:
                config_name = config.replace(".yaml", f"_{post_fix}.yaml")
            else:
                config_name = config
            config_key = config.split("/")[-1].split(".")[0]
            self.configs[config_key] = self._load_config(config_name)
        self.column_name = column_name

    def _generate(self, sample) -> dict:
        sample = {**sample, **self.configs[sample[self.column_name]]}
        return sample

    def generate(self, samples) -> Dataset:
        samples = samples.map(self._generate, num_proc=self.ctx.num_procs)
        return samples


class SelectorBlock(Block):
    def __init__(self, ctx, choice_map, choice_col, output_col) -> None:
        super().__init__(ctx, block_name=self.__class__.__name__)
        self.choice_map = choice_map
        self.choice_col = choice_col
        self.output_col = output_col

    def _generate(self, sample) -> dict:
        sample[self.output_col] = sample[self.choice_map[sample[self.choice_col]]]
        return sample

    def generate(self, samples: Dataset) -> Dataset:
        samples = samples.map(self._generate, num_proc=self.ctx.num_procs)
        return samples


class CombineColumnsBlock(Block):
    def __init__(self, ctx, columns, output_col, separator="\n\n") -> None:
        super().__init__(ctx, block_name=self.__class__.__name__)
        self.columns = columns
        self.output_col = output_col
        self.separator = separator

    def _generate(self, sample) -> dict:
        sample[self.output_col] = self.separator.join(
            [sample[col] for col in self.columns]
        )
        return sample

    def generate(self, samples: Dataset) -> Dataset:
        samples = samples.map(self._generate, num_proc=self.ctx.num_procs)
        return samples
