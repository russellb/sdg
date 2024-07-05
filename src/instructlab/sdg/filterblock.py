# SPDX-License-Identifier: Apache-2.0
# Standard
import operator

# Third Party
from datasets import Dataset

# Local
from .block import Block
from .logger_config import setup_logger

logger = setup_logger(__name__)


class FilterByValueBlockError(Exception):
    """An exception raised by the FilterByValue block."""


def _get_operator_func(op):
    if not op in dir(operator):
        raise FilterByValueBlockError("Unknown FilterByValueBlock operation '{op}'")
    return getattr(operator, op)


def _get_convert_dtype(convert_dtype):
    if not convert_dtype:
        return None

    type_mapping = {
        "int": int,
        "float": float,
        "bool": bool,
    }

    if not convert_dtype in type_mapping:
        raise FilterByValueBlockError(
            "Unknown FilterByValueBlock convert_dtype '{convert_dtype}'"
        )

    return type_mapping[convert_dtype]


class FilterByValueBlock(Block):
    def __init__(
        self, ctx, filter_column, filter_value, operation, convert_dtype=None
    ) -> None:
        super().__init__(ctx, block_name=self.__class__.__name__)
        self.value = filter_value
        self.column_name = filter_column
        self.operation = _get_operator_func(operation)
        self.convert_dtype = convert_dtype
        if self.convert_dtype:
            self.value = self.convert_dtype(self.value)

    def _convert_dtype(self, sample):
        try:
            sample[self.column_name] = self.convert_dtype(sample[self.column_name])
        except ValueError as e:
            logger.error(
                "Error converting dtype: %s, filling with None to be filtered later", e
            )
            sample[self.column_name] = None
        return sample

    def generate(self, samples) -> Dataset:
        if self.convert_dtype:
            samples = samples.map(
                lambda x: {
                    **x,
                    self.column_name: self.convert_dtype(x[self.column_name]),
                },
                num_proc=self.num_procs,
            )

        return samples.filter(
            lambda x: self.operation(x[self.column_name], self.value),
            num_proc=self.ctx.num_procs,
        )
