from typing import Optional, List

from haystack import component
from haystack.dataclasses import ChatMessage

@component
class RAGClassifier:
    '''
    A custom Haystack component that determines whether a response to a user message should utilize RAG or not.
    '''

    def __init__(self):
        # The number of times the component has been run
        self.iteration_counter = 0

    # Define component output
    @component.output_types(valid_reply=str, invalid_replies=Optional[List[str]], error_message=Optional[str])
    def run(self, replies: List[ChatMessage]):
        self.iteration_counter += 1

        ## Try to parse the LLM's reply ##
        # If the LLM's reply is either 'chat' or 'rag', return '"valid_replies"'
        try:
            output = replies[0].text.lower()

            # Make sure the LLM's response has 'chat' or 'rag', but not both
            if 'chat' in output and 'rag' in output:
                raise ValueError("Reply cannot contain both 'chat' and 'rag'")
            if 'chat' not in output and 'rag' not in output:
                raise ValueError("Reply must contain either 'chat' or 'rag'")

            if 'chat' in output:
                output = 'chat'
            
            if 'rag' in output:
                output = 'rag'

            return {"valid_reply": output}
        # If the LLM's reply is corrupted or not valid, return "invalid_reply" and "error_message" for the LLM to try again
        except ValueError as e:
            return {"invalid_replies": replies, "error_message": str(e)}