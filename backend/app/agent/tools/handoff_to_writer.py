from llama_index.core.tools import FunctionTool


def handoff_to_writer():
    """Handoff to another agent to write the final answer.

    This function takes no arguments.
    """
    return "handoff_to_writer"


tool = FunctionTool.from_defaults(
    fn=handoff_to_writer,
    return_direct=True,
)
