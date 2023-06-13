from steamship_langchain.llms import OpenAI
from steamship.invocable import PackageService, post

class JokeWizard(PackageService):

    @post("generate")
    def generate(self) -> str:
        llm = OpenAI(
            client=self.client,
            model_name="text-ada-001", 
            n=2, best_of=2, 
            temperature=0.9)
        
        return llm("Tell me a joke")
