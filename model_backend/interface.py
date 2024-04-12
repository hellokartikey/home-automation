import random
from transformers import T5Tokenizer, T5ForConditionalGeneration

class Model:
    def __init__(self, name: str = "google/flan-t5-base") -> None:
        self.name = name
        self.tokenizer = T5Tokenizer.from_pretrained(self.name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.name)

    def infer(self, message: str) -> str:
        input_ids = self.tokenizer(message, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids)

        return self.tokenizer.decode(outputs[0])

    def set_context(self, context: str) -> None:
        self.context = context

    def context(self) -> str:
        return ""
