from src.langgraph_agentic_ai.state.state import State
from langchain_core.messages import HumanMessage

class ChatbotWithToolNode:
    """
    Chatbot logic enhanced with tool integration.
    """
    def __init__(self, model):
        self.llm = model

    def process(self, state: State) -> dict:
        """
        Processes the input state and generates a response with tool integration.
        """
        user_input = state["messages"][-1] if state["messages"] else ""
        llm_response = self.llm.invoke([{"role": "user", "content": user_input}])

        # Simulate tool-specific logic
        tools_response = f"Tool integration for: '{user_input}'"

        return {"messages": [llm_response, tools_response]}
    

    def create_chatbot(self, tools):
        """
        Returns a chatbot node function that can call tools.
        """
        llm_with_tools = self.llm.bind_tools(tools)

        def chatbot_node(state: State):
            """
            Chatbot logic for processing the input state and returning a response.
            Invokes the LLM with tool bindings.
            """
            messages = state["messages"]
            
            # If the first message is a string, convert it to HumanMessage
            if messages and isinstance(messages[0], str):
                messages = [HumanMessage(content=messages[0])]
            
            # Invoke the LLM with tools
            response = llm_with_tools.invoke(messages)
            
            return {"messages": [response]}

        return chatbot_node

