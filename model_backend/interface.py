import random

class Model:
    def __init__(self, name: str) -> None:
        self.name = name

    def infer(self, message: str) -> str:
        response = random.choice([
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ])

        return response

    def set_context(self, context: str) -> None:
        self.context = context

    def context(self) -> str:
        return ""
