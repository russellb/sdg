import yaml
from collections import ChainMap
from abc import ABC, abstractmethod
from typing import Any, Dict, Union
from .logger_config import setup_logger

from datasets import Dataset

logger = setup_logger(__name__)


class Block(ABC):
    def __init__(self, block_name: str) -> None:
        self.block_name = block_name

    @staticmethod
    def _validate(prompt_template: str, input_dict: Dict[str, Any]) -> bool:
        """
        Validate the input data for this block. This method should be implemented by subclasses
        to define how the block validates its input data.
        
        :return: True if the input data is valid, False otherwise.
        """
        class Default(dict):
            def __missing__(self, key: str) -> None:
                raise KeyError(key)

        try:
            prompt_template.format_map(ChainMap(input_dict, Default()))
            return True
        except KeyError as e:
            logger.error("Missing key: {}".format(e))
            return False

    def _parse(self, output: str) -> str:
        """
        Parse the output generated by this block.
        This method should process the generated output and return the parsed result.
        
        :param output: The raw output generated by the block.
        :return: The parsed result.
        """
        raise NotImplementedError("The '_parse' method must be implemented by subclasses.")

    @abstractmethod
    def generate(self) -> Dataset:
        """
        Generate the output from the block. This method should first validate the input data,
        then generate the output, and finally parse the generated output before returning it.
        
        :return: The parsed output after generation.
        """
        pass

    def _generate(self) -> Dataset:
        """
        The core generation logic for this block. This method should be implemented by subclasses
        to define how the block generates its output.
        
        :return: The raw output generated by the block.
        """
        raise NotImplementedError("The '_generate' method must be implemented by subclasses.")

    def _load_config(self, config_path: str) -> Union[Dict[str, Any], None]:
        """
        Load the configuration file for this block.
        
        :param config_path: The path to the configuration file.
        :return: The loaded configuration.
        """
        with open(config_path, 'r', encoding="utf-8") as config_file:
            return yaml.safe_load(config_file)