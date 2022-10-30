
from typing import *
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt import Template

class ManualTemplate(Template):
    registered_inputflag_names = ["loss_ids", 'shortenable_ids']

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 text: Optional[str] = None,
                 placeholder_mapping: dict = {'<text_a>':'text_a','<text_b>':'text_b'},
                ):
        super().__init__(tokenizer=tokenizer,
                         placeholder_mapping=placeholder_mapping)
        self.text = text

    def on_text_set(self):
        """
        when template text was set

        1. parse text
        """

        self.text = self.parse_text(self.text)