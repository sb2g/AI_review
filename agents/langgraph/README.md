### LangGraph
- When to use LangGraph?
    - Design a flow of actions based on the output of each action, and decide what to execute next accordingly
    - Some scenarios:
        - Multi-step reasoning processes that need explicit control on the flow
        - Applications requiring persistence of state between steps
        - Systems that combine deterministic logic with AI capabilities
        - Workflows that need human-in-the-loop interventions
        - Complex agent architectures with multiple components working together
    - Additional features on top of vanilla python based agent:
        - It includes states, visualization, logging (traces), built-in human-in-the-loop, and more

- How does it work?
    - LangGraph uses a directed graph structure to define the flow of your application
    - Building blocks
        - Nodes: 
            - Represent individual processing steps
            - Can be python functions or agents. Each node:
                - Takes the state as input, Performs some operation, Returns updates to the state
                - Can contain:
                    - LLM calls: Generate text or make decisions
                    - Tool calls: Interact with external systems
                    - Conditional logic: Determine next steps
                    - Human intervention: Get input from users
            - Special nodes: START (entrypoint), END (end of task)
        - Edges: 
            - Connect nodes and define the possible paths through your graph
        - Conditional edges
            - Decisions
        - State: 
            - Is user defined and maintained and passed between nodes during execution
            - When deciding which node to target next, this is the current state that we look at
        - StateGraph: 
            - Is the container that holds your entire agent workflow  
            - Add nodes & edges to it 
            - Compile it 
            - Invoke it 
    - Workflows
        - ReAct pattern  

### References:
- https://huggingface.co/learn/agents-course/unit1/what-are-agents
- https://huggingface.co/docs/smolagents/conceptual_guides/intro_agents
- https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/  