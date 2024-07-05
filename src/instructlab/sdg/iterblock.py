# Third Party
from datasets import Dataset

# Local
from .block import Block
from .logger_config import setup_logger

logger = setup_logger(__name__)


class IterBlock(Block):
    def __init__(self, ctx, block_name, block_type, block_kwargs, **kwargs):
        super().__init__(ctx, block_name)
        self.block = block_type(**block_kwargs)
        self.gen_kwargs = kwargs.get("gen_kwargs", {})
        self.gen_kwargs = kwargs.get("gen_kwargs", {})

    def generate(self, samples, **gen_kwargs) -> Dataset:
        generated_samples = []
        for _ in range(self.ctx.num_iters):
            batch_generated = self.block.generate(
                samples, **{**self.gen_kwargs, **gen_kwargs}
            )
            generated_samples.extend(batch_generated)

        return Dataset.from_list(generated_samples)
