# Chapter 05: OpenWebUI Integration

## Getting Started

To run this chapter, navigate to the current directory and execute the lifecycle script. This script will handle all the Docker services for you.

```bash
./start-chapter-resources.sh
```

---

This chapter integrates the ADEPT framework with OpenWebUI, providing a polished, production-grade user interface for interacting with the scientific workflow agent. It also introduces a new backend service to make this integration possible.

## Key Additions

- **OpenWebUI Backend Service**: A new service, `openwebui_backend`, is introduced. It is defined in `Dockerfile.openwebui_backend` and orchestrated via `docker-compose-openwebui.yaml`. This service exposes an OpenAI-compatible API endpoint (`/v1/chat/completions`), allowing the `ScientificWorkflowAgent` to act as a backend model for any OpenAI-compatible frontend.

- **Agent as a Backend Model**: The new backend service enables the agent to be seamlessly integrated into OpenWebUI. Users can select the "Agentic Framework CoT" model from the dropdown in OpenWebUI to interact directly with the full suite of scientific and computational tools.

- **Frontend/Backend Separation**: This chapter emphasizes a clear architectural separation between the user interface and the agentic backend. The `docker-compose-openwebui.yaml` file manages the backend services, while the OpenWebUI container runs as a separate frontend, communicating with the backend via the standardized API. This modularity allows for independent development, scaling, and maintenance of the UI and the agent framework.

## Key Technologies

- [OpenWebUI](https://open-webui.com/): A user-friendly and feature-rich web interface for interacting with local and remote large language models.
- [OpenAI-compatible API](https://platform.openai.com/docs/api-reference/chat): A standardized API specification for chat-based language models, which allows for interoperability between different frontends and backends.
- [Docker Compose](https://docs.docker.com/compose/): A tool for defining and running multi-container Docker applications.

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

### Scenario 2: Graph Mode - Dynamic Problem Solving with Plotting

This scenario demonstrates a more dynamic problem-solving approach using the 'graph' mode, where the supervisor agent intelligently routes tasks without a rigid pre-defined plan.

1.  **Create Multi-Agent Session:**
    *   **User Query:** "Create a multi-agent session in 'graph' mode. The team should include a 'Biologist' and a 'Chemist'."
    *   **Expected Agent Response:** The agent will confirm session creation and provide a `multi_agent_session_id`.

2.  **Execute Dynamic Task:**
    *   **User Query:** "Using the graph-based session, find the human KRAS protein sequence from UniProt, then perform a BLAST search against SwissProt, and finally identify chemical compounds in PubChem known to inhibit it. Include a plot of the BLAST results if possible, remembering the plotting guidelines for Python code."
    *   **Expected Agent Behavior:** The supervisor agent dynamically routes tasks to the 'Biologist' (for UniProt and BLAST) and 'Chemist' (for PubChem). The final response should synthesize findings and include any generated plot, which should then render correctly.

---

**Important Note for Plot Rendering:**

For plot rendering to work correctly in your local Docker Compose environment, ensure you have set the `SANDBOX_MCP_SERVER_PUBLIC_URL` environment variable in your `.env` file (e.g., `SANDBOX_MCP_SERVER_PUBLIC_URL=http://localhost:8082`). This variable tells the sandbox server the public URL it should use when generating plot links.

## References

### How It Works: OpenWebUI Integration                                                                                                               

This chapter focuses on providing a polished and user-friendly interface for the agentic framework by integrating it with OpenWebUI.                  

**Example Scenario:** A user typing "Hello" into the OpenWebUI chat interface.                                                                       

1.  **Frontend Request**: The user types "Hello" into the OpenWebUI chat interface and hits send. OpenWebUI formats this into a standard,            
OpenAI-compatible API request.                                                                                                                       

**Backend Service**: The request is sent to the `openwebui_backend` service. This service acts as a bridge, receiving the OpenAI-formatted   
request.                                                                                                                                     

3.  **Agent Invocation**: The backend service extracts the user's message and passes it to the `ScientificWorkflowAgent`.                    

4.  **Response Streaming**: The agent processes the message (in this case, likely just generating a greeting) and streams the response back to the
backend service, which in turn streams it to the OpenWebUI frontend for a real-time, interactive chat experience.                                 


- [OpenWebUI Documentation](https://docs.openwebui.com/getting-started/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

## Additional Resources

### Web Development and APIs

- **Coursera**: [Full-Stack Web Development with React Specialization](https://www.coursera.org/specializations/full-stack-react)
- **Udemy**: [The Complete 2023 Web Development Bootcamp](https://www.udemy.com/course/the-complete-web-development-bootcamp/)
- **YouTube**: [REST API Crash Course](https://www.youtube.com/watch?v=-mN3VyJuCjM)
- **YouTube**: [What is an API?](https://www.youtube.com/watch?v=bxuYDT-BWaI)

### Docker

- **Coursera**: [Docker for Absolute Beginners](https://www.coursera.org/projects/docker-for-absolute-beginners)
- **Udemy**: [Docker for the Absolute Beginner - Hands On - DevOps](https://www.udemy.com/course/learn-docker/)
- **YouTube**: [Docker Tutorial for Beginners](https://www.youtube.com/watch?v=3c-iBn73dDE)
