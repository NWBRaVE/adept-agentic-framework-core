# Chapter 03: LLM Sandbox and Multi-Agent Capabilities

## Getting Started

To run this chapter, navigate to the current directory and execute the lifecycle script. This script will handle all the Docker services for you.

```bash
./start-chapter-resources.sh
```

---

This chapter introduces two major features to the ADEPT framework: a sandboxed environment for secure code execution and a multi-agent system for tackling complex tasks.

## Key Additions

- **Sandbox MCP Server**: A new, highly specialized MCP server is added to execute arbitrary Python, JavaScript, and shell code in a secure, isolated environment using the `llm-sandbox` library. This allows the agent to perform dynamic calculations, data manipulation, and other tasks that can be solved with code, without compromising the host system.
- **Multi-Agent Tool**: The main MCP server is updated with a `multi_agent_tool`. This tool allows for the creation and management of teams of AI agents. A "Planner" agent can generate a plan, which is then executed by a "Supervisor" agent that delegates tasks to worker agents with specific roles (e.g., "chemist", "bioinformatician"). This enables the framework to handle much more complex, multi-step workflows that require different areas of expertise.
- **Multi-Agent ID**: A `multi_agent_id` is introduced for context tracking within these complex, multi-agent sessions.

## Key Technologies

- [llm-sandbox](https://github.com/gofastmcp/llm-sandbox): A library for securely executing code in an isolated environment.
- [Multi-Agent Systems](https://en.wikipedia.org/wiki/Multi-agent_system): A computerized system composed of multiple interacting intelligent agents.
- [LangGraph](https://langchain-ai.github.io/langgraph/): A library for building stateful, multi-agent applications with LLMs.

## Demonstration Scenarios

This section provides sample queries to demonstrate the advanced multi-agent orchestration capabilities of this chapter, including how they interact with other tools and handle plotting.

### Scenario 1: Router Mode - Multi-Phase Scientific Analysis with Plotting

This scenario demonstrates a structured, multi-phase analysis using the 'router' mode, where a plan is generated and executed step-by-step.

1.  **Create Multi-Agent Session:**
    *   **User Query:** "Create a multi-agent session in 'router' mode. The team should include a 'Bioinformatician', a 'Chemist', and a 'Software Engineer'. The overall goal is to perform a competitive analysis for a new drug candidate targeting protein P01112."
    *   **Expected Agent Response:** The agent will confirm session creation and provide a `multi_agent_session_id`.

2.  **Generate and Review Plan:**
    *   **User Query:** "Generate a detailed, multi-phase plan for the session I just created, focusing on analyzing P01112's sequence, finding existing patents, and identifying similar compounds. Ensure the plan uses appropriate MCP tools and includes Python code execution steps for plotting where relevant. For Python plotting, remember to explicitly create figure objects (e.g., `fig, ax = plt.subplots()`) and avoid `plt.show()`. The figure object should be the last expression in the code block for capture. Also, ensure any Python code for plotting includes in-script package installations (e.g., using `subprocess`)."
    *   **Expected Agent Response:** The agent will present the generated plan and ask for your approval.

3.  **Approve and Execute Plan:**
    *   **User Query:** "The plan looks good. Approve and execute it."
    *   **Expected Agent Behavior:** The agent will execute the plan. If a plotting step is encountered, the `ExecuteCode` tool runs the Python code, captures the plot, and the agent includes the plot URL in its response, which should then render in the UI.

<!---

### Scenario 2: Graph Mode - Dynamic Problem Solving with Plotting

This scenario demonstrates a more dynamic problem-solving approach using the 'graph' mode, where the supervisor agent intelligently routes tasks without a rigid pre-defined plan.

1.  **Create Multi-Agent Session:**
    *   **User Query:** "Create a multi-agent session in 'graph' mode. The team should include a 'Biologist', and a 'Chemist'."
    *   **Expected Agent Response:** The agent will confirm session creation and provide a `multi_agent_session_id`.

2.  **Execute Dynamic Task:**
    *   **User Query:** "Using the graph-based session, find the human KRAS protein sequence from UniProt, then perform a BLAST search against SwissProt, and finally identify chemical compounds in PubChem known to inhibit it. Include a plot of the BLAST results if possible, remembering the plotting guidelines for Python code."
    *   **Expected Agent Behavior:** The supervisor agent dynamically routes tasks to the 'Biologist' (for UniProt and BLAST) and 'Chemist' (for PubChem). The final response should synthesize findings and include any generated plot, which should then render correctly.

-->

---

**Important Note for Plot Rendering:**

For plot rendering to work correctly in your local Docker Compose environment, ensure you have set the `SANDBOX_MCP_SERVER_PUBLIC_URL` environment variable in your `.env` file (e.g., `SANDBOX_MCP_SERVER_PUBLIC_URL=http://localhost:8082`). This variable tells the sandbox server the public URL it should use when generating plot links.

## References

### How It Works: Sandbox and Multi-Agent Systems

**Example User Query:** "Write a Python script to calculate the average of the numbers [10, 20, 30] and tell me if the result is greater than 15."

1.  **Sandbox MCP Server**: The agent generates a Python script to perform the calculation and comparison. It then calls the `ExecuteCode` tool, sending the script to the Sandbox MCP server. The server runs the code in a secure, isolated environment and returns the script's output (e.g., "The average is 20, which is greater than 15.") to the agent, who then relays it to the user.

2.  **Multi-Agent Orchestration**: For more complex tasks, the agent can use the `CreateMultiAgentSession` tool to assemble a team of specialized agents. For example, a query like "Find a potential drug candidate for protein P01112" would cause the agent to create a team of a bioinformatician and a chemist. The bioinformatician would analyze the protein's sequence, and the chemist would then use that information to search for related compounds, demonstrating a collaborative, multi-step workflow.

- [llm-sandbox Documentation](https://github.com/gofastmcp/llm-sandbox)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

## Additional Resources

### Sandboxing and Secure Code Execution

- **YouTube**: [Sandboxing explained](https://www.youtube.com/watch?v=kn32PHG2wcU)
- **Blog Post**: [What is a Sandbox?](https://www.kaspersky.com/resource-center/definitions/what-is-a-sandbox)

### Multi-Agent Systems

- **Google A2A Protocol**:
    - [Google's A2A Protocol: Enabling AI Agents to Collaborate](https://a2aprotocol.ai/)
- **Amazon Bedrock**:
    - [Build generative AI applications with agents for Amazon Bedrock](https://aws.amazon.com/bedrock/agents/)
- **Docker**:
    - [Docker AI Agent](https://www.docker.com/blog/docker-ai-a-new-ai-powered-product-to-boost-developer-productivity/)
- **Coursera**: [Multi-Agent Reinforcement Learning](https://www.coursera.org/learn/multi-agent-reinforcement-learning-in-robotics)
- **Udemy**: [Multi-Agent Systems with Python](https://www.udemy.com/course/multi-agent-systems-with-python/)
- **YouTube**: [Introduction to Multi-Agent Systems](https://www.youtube.com/watch?v=eHEHE2fpnWQ)
