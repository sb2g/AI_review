# Overview:
- This part of repo goes over what are Agents & their components / workflows.  
    - It shows a basic agent from scratch in pytorch.   
    - It then aims to go over the following Agent frameworks:   
        - smolagents  
        - LlamaIndex  
        - LangGraph  
        - CrewAI   
    - Use `environment_mlenv2a.yaml` to define the Conda env to run the code shown here.  
        - `mlenv2a`:  `conda env create -f environment_mlenv2a.yml`

# Overview of Agents: 
- Use AI model(s) as reasoning engine along with tools to interact with environment and perform a given task 
- Capabilities:
    - Understand natural language (using AI models) 
    - Reason & Plan (using AI models) 
    - Interact with environment: Retrieve info, Execute actions & Observe results (using Tools & AI models) 
- Examples:
    - Personal virtual assistants: Google assistant, Alexa, Siri
    - Customer service chatbots
- Key components: 
    - AI models: 
        - Can be LLMs or VLMs 
        - Used for natural language understanding, reasoning, planning 
    - Tools / functions: 
        - Can be Custom tools/functions or General purpose tools / external resources
        - Used to perform actions that AI models cant do themselves. These are some types of actions 
            - Gather information: Web search, Query db, Retrieve docs
            - Use tools: Make API calls, Run calculations, Execute code
            - Interact with env: Control devices
            - Communicate: with Users, Agents
        - Tools available to agents can have great impact on the results 
- Some high level items needed in an agent 
    - Thought 
    - Action 
    - Observation 
    - Tools available
    - Parser to extract tool calls 
    - System prompt to call tools
    - Memory
    - Error logging & Retry mechanism
- Workflows
    - ReAct pattern:
        - Reason about the task
        - Act using appropriate tools
        - Observe the results
        - Repeat as necessary until task is complete
    - Thought, Action, Observation:
        - Thought
            - For e.g. Read the prompt and decide what tool to call with what parameters, or if final answer should be generated.        - 
            - Approaches: 
                - ReAct approach for Internal reasoning:
                    - Zero-shot Chain of Thought (as opposed to Few-shot Chain of Thought)
                    - Add this to the prompt: "Let's think step by step". 
                    - Its only a prompting approach, not a training approach like reasoning models do it.           
        - Action
            - For e.g. call weather API
            - Types of agents based on how action is specified:
                - JSON agent: Action written in JSON format
                    - Function calling agent: Action written as function call
                - Code agent: Action written as a code block
            - Approaches:
                - Stop and parse approach
                    - Output is in pre-determined format (JSON/code). Stop generating tokens.
                    - JSON agent: Parser reads the action & calls tool with required parameters
                    - Code agent: Executes the high-level language code externally in its interpreter. (e.g. Python)
        - Observation
            - For e.g. retrieve response from API and add it to prompt
            - Types of observations
                - System feedback: Errors, notifiations, status codes
                - Data updates: Db, file system
                - Env data: Resource usage, System metrics, etc. 
                - Respose analysis: Responses from API, query, etc. 
                - Time-based events: Task completed, Deadline
            - Approach
                - Collect feedback from env: Data, Error, System logs, etc.
                - Append these results into existing context / prompt (i.e. update memory)
                - Use this updated info for subsequent thought & action iteration

### Some high level types of agents:
- Simple processor: Process AI model output. No impact on program flow
- Router: AI model output determines basic program control flow 
- Tool/Function caller: AI model output calls required functions
- Multi-step agent: AI model output controls program iteration/continuation
- Multi-agent: AI model agent workflow can start another AI model agent workflow
- Code agents: AI model codes its own tools / agents

### When to use agents and not direct coding/automation? 
- Pre-determined workfow is not enough. For e.g. Plan a trip based on complex natural language input => Check weather, maps, availability, etc.
- 

## Trade-off: Control (vs) Freedom
When designing AI applications, there is a trade-off between control and freedom:
- Control allows you to ensure predictable behavior and maintain guardrails.
- Freedom gives AI model to be more room to be creative and tackle unexpected problems.

### Some frameworks for AI agents
- smolagents
    - Focus on "Freedom" of execution of agent
    - Focuses on codeAgent
    - Can call multiple tools in a single action step, create their own tools, etc. However, this behavior can make them less predictable and less controllable than a regular Agent working with JSON
- LlamaIndex
- LangGraph
    - Focuses on "Controlling" the execution of agent
    - LangGraph provides the structure you need
    - Useful if your application involves a series of steps that need to be orchestrated in a specific way, with decisions being made at each junction point
- CrewAI

### References:
- https://huggingface.co/learn/agents-course/unit1/what-are-agents
- https://huggingface.co/docs/smolagents/conceptual_guides/intro_agents
