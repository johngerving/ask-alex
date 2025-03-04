from typing import Optional, Dict, List 

from haystack import component
from haystack.dataclasses import ChatMessage

@component
class IntentClassifier:
    '''
    A custom Haystack component that parses LLM output to classify user intent.
    '''

    def __init__(self, classes: Dict[str, str]):
        # The number of times the component has been run
        self.iteration_counter = 0

        # Available intent classes
        self.classes = classes

    # Define component output
    @component.output_types(valid_reply=str, invalid_replies=Optional[List[str]], error_message=Optional[str])
    def run(self, replies: List[ChatMessage]):
        self.iteration_counter += 1

        ## Try to parse the LLM's reply ##
        # If the LLM's reply contains one intent class, return '"valid_replies"'
        try:
            output = replies[0].text.lower()

            # Get number of intent classes found in response
            matches = [x for x in self.classes.keys() if x in output]

            if len(matches) == 1:
                output = matches[0]
            elif len(matches) < 1:
                raise ValueError(f"Reply does not contain any intent classes. Must contain at least one of: {', '.join(self.classes.keys())}")
            elif len(matches) > 1:
                raise ValueError(f"Reply contains more than one intent class. Must contain no more than one of: {', '.join(self.classes.keys())}")

            return {"valid_reply": output}
        # If the LLM's reply is corrupted or not valid, return "invalid_reply" and "error_message" for the LLM to try again
        except ValueError as e:
            return {"invalid_replies": replies, "error_message": str(e)}