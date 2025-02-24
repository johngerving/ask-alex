from rag_pipeline import RagPipeline

from ray import serve
from starlette.requests import Request

# from dotenv import load_dotenv
# load_dotenv()

@serve.deployment
class HaystackQA:
    def __init__(self):
        self.pipeline = RagPipeline()

    async def __call__(self, request: Request) -> str:
        query = (await request.body()).decode('UTF-8')

        # Run the pipeline with the user's query
        res = self.pipeline.run(query)

        # Return different reply based on whether chat route or RAG route was followed
        if "rag_llm" in res:
            replies = res["rag_llm"]["replies"]
        elif "chat_llm" in res:
            replies = res["chat_llm"]["replies"]
        else:
            raise Exception("No LLM output found")

        if replies:
            return replies[0].text

        return ""

haystack_deployment = HaystackQA.bind()
# query = "What are the impacts of ammonium phosphate-based fire retardants on cyanobacteria growth?"
