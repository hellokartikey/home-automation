import random
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

CONTEXT = """Reply as a home assistant with plain text without tags.
When asked to turn on lights say LIGHTSON.
When asked to turn off lights say LIGHTSOFF.
When asked to turn on fan say FANON.
When asked to turn off fan say FANOFF.

User: """

LIGHTSON = "LIGHTSON"
LIGHTSOFF = "LIGHTSOFF"
FANON = "FANON"
FANOFF = "FANOFF"

class Model:
    def __init__(self, name: str = "google/flan-t5-large") -> None:
        self.name = name
        self.tokenizer = T5Tokenizer.from_pretrained(self.name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.name)
        self.context = CONTEXT

    def infer(self, message: str) -> str:
        input_ids = self.tokenizer(
                self.context + message, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids)

        output_str = self.tokenizer.decode(outputs[0])

        if FANON in output_str:
            st.success("Turning on fan!")

        if FANOFF in output_str:
            st.success("Turning off fan!")

        if LIGHTSON in output_str:
            st.success("Turning on lights!")

        if LIGHTSOFF in output_str:
            st.success("Turning off lights!")

        return output_str

    def set_context(self, context: str) -> None:
        self.context = context

    def context(self) -> str:
        return self.context


model = Model()

