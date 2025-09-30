# Chapter 04: Cloud-Native Deployment

## Getting Started

To run this chapter, navigate to the current directory and execute the lifecycle script. This script will handle all the Docker services for you.

```bash
./start-chapter-resources.sh
```

---

This chapter transitions the ADEPT framework from a local Docker Compose setup to a cloud-native deployment model, targeting Kubernetes. It introduces the necessary Infrastructure as Code (IaC) and CI/CD components to automate the deployment and management of the entire application stack on both Microsoft Azure and Amazon Web Services (AWS).

## Key Additions

- **Kubernetes Deployment with Helm**: A comprehensive Helm chart is introduced in `infra/helm/` to define, configure, and deploy all the framework's services (MCP servers, Streamlit UI, etc.) as Kubernetes resources. This enables consistent and repeatable deployments across different Kubernetes environments.

- **Azure Infrastructure & CI/CD**:
    - **Pulumi for Azure**: The `infra/azure/pulumi/` directory contains a Pulumi project to programmatically provision the core Azure infrastructure, including an Azure Kubernetes Service (AKS) cluster and an Azure Container Registry (ACR).
    - **Azure Pipelines**: An `azure-pipelines.yml` file defines a full CI/CD pipeline. This pipeline automates the process of building all service Docker images, pushing them to ACR, and deploying the Helm chart to the AKS cluster.

- **AWS Infrastructure & CI/CD**:
    - **AWS CDK**: The `infra/aws/cdk/` directory provides an AWS Cloud Development Kit (CDK) application to provision the necessary AWS infrastructure. This includes Amazon ECR repositories for container images and an Amazon ECS cluster for running the services.
    - **AWS CodePipeline**: The CDK app also sets up a complete CI/CD pipeline using AWS CodePipeline and AWS CodeBuild, which automates the building, pushing, and deployment of the services to ECS.

Note thatthe IaC deployment has only been fully tested within k8s cluster at PNNL's EMSL Computing Facility. This is a very experimental branch with cloud deployments not yet fully tested. However, the reader is encouraged to learn more and play with the k8s deployment on their development machine, e.g., Docker Desktop with its local Kubernetes Engine enabled. Please refer to `scripts/helm-manage_local_k8s.sh` and `infra/helm/README.md` for additional details.  

## Key Technologies

- [Kubernetes](https://kubernetes.io/): An open-source system for automating deployment, scaling, and management of containerized applications.
- [Helm](https://helm.sh/): The package manager for Kubernetes, used to define, install, and upgrade even the most complex Kubernetes applications.
- [Docker](https://www.docker.com/): A platform for developing, shipping, and running applications in containers.
- [Pulumi](https://www.pulumi.com/): An infrastructure as code platform that allows you to use familiar programming languages to provision and manage cloud resources.
- [Azure](https://azure.microsoft.com/): A cloud computing service created by Microsoft for building, testing, deploying, and managing applications and services through Microsoft-managed data centers.
- [Amazon Web Services (AWS)](https://aws.amazon.com/): A comprehensive and broadly adopted cloud platform, offering over 200 fully featured services from data centers globally.
- [CI/CD](https://aws.amazon.com/devops/what-is-cicd/): Continuous Integration and Continuous Delivery/Deployment is a method to frequently deliver apps to customers by introducing automation into the stages of app development.



## Demonstration Scenarios

This section provides sample queries to demonstrate the advanced multi-agent orchestration capabilities of this chapter, including how they interact with other tools and handle plotting.

### Scenario 1: Router Mode - Multi-Phase Scientific Analysis with Plotting

This scenario demonstrates a structured, multi-phase analysis using the 'router' mode, where a plan is generated and executed step-by-step.

1.  **Create Multi-Agent Session:**
    *   **User Query:** "Create a multi-agent session in 'router' mode. The team should include a 'Bioinformatician', a 'Chemist', and a 'Software Engineer'. The overall goal is to perform a competitive analysis for a new drug candidate targeting protein P01112."
    *   **Expected Agent Response:** The agent will confirm session creation and provide a `multi_agent_session_id`.

2.  **Generate and Review Plan:**
    *   **User Query:** "Generate a detailed, multi-phase plan for the session I just created, focusing on analyzing P01112's sequence, finding existing patents, and identifying similar compounds. Ensure the plan uses appropriate MCP tools and includes Python code execution steps for plotting where relevant. For Python plotting, remember to explicitly create figure objects (e.g., `fig, ax = plt.subplots()`) and avoid `plt.show()`. The figure object should be the last expression in your code block for it to be captured. Also, ensure any Python code for plotting includes in-script package installations (e.g., using `subprocess`)."
    *   **Expected Agent Response:** The agent will present the generated plan and ask for your approval.

3.  **Approve and Execute Plan:**
    *   **User Query:** "The plan looks good. Approve and execute it."
    *   **Expected Agent Behavior:** The agent will execute the plan. If a plotting step is encountered, the `ExecuteCode` tool runs the Python code, captures the plot, and the agent includes the plot URL in its response, which should then render in the UI.

### Scenario 2: Graph Mode - Dynamic Problem Solving with Plotting

This scenario demonstrates a more dynamic problem-solving approach using the 'graph' mode, where the supervisor agent intelligently routes tasks without a rigid pre-defined plan.

1.  **Create Multi-Agent Session:**
    *   **User Query:** "Create a multi-agent session in 'graph' mode. The team should include a 'Biologist', and a 'Chemist'."
    *   **Expected Agent Response:** The agent will confirm session creation and provide a `multi_agent_session_id`.

2.  **Execute Dynamic Task:**
    *   **User Query:** "Using the graph-based session, find the human KRAS protein sequence from UniProt, then perform a BLAST search against SwissProt, and finally identify chemical compounds in PubChem known to inhibit it. Include a plot of the BLAST results if possible, remembering the plotting guidelines for Python code."
    *   **Expected Agent Behavior:** The supervisor agent dynamically routes tasks to the 'Biologist' (for UniProt and BLAST) and 'Chemist' (for PubChem). The final response should synthesize findings and include any generated plot, which should then render correctly.

---

**Important Note for Plot Rendering:**

For plot rendering to work correctly in your local Docker Compose environment, ensure you have set the `SANDBOX_MCP_SERVER_PUBLIC_URL` environment variable in your `.env` file (e.g., `SANDBOX_MCP_SERVER_PUBLIC_URL=http://localhost:8082`). This variable tells the sandbox server the public URL it should use when generating plot links.

## References

- [Kubernetes Documentation](https://kubernetes.io/docs/home/)
- [Helm Documentation](https://helm.sh/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [Pulumi Documentation](https://www.pulumi.com/docs/)
- [Azure Documentation](https://docs.microsoft.com/en-us/azure/?product=popular)
- [AWS Documentation](https://docs.aws.amazon.com/)
- [Azure Pipelines Documentation](https://docs.microsoft.com/en-us/azure/devops/pipelines/?view=azure-devops)
- [AWS CDK Documentation](https://docs.aws.amazon.com/cdk/v2/guide/home.html)

## Additional Resources

### Kubernetes and Helm

- **Coursera**: [Google Kubernetes Engine Specialization](https://www.coursera.org/specializations/google-kubernetes-engine)
- **Udemy**: [Docker and Kubernetes: The Complete Guide](https://www.udemy.com/course/docker-and-kubernetes-the-complete-guide/)
- **YouTube**: [Kubernetes Full Course in 6 Hours | Kubernetes Tutorial for Beginners](https://www.youtube.com/watch?v=X48VuDVv0do)
- **YouTube**: [Full Helm Course for Beginners](https://www.youtube.com/watch?v=DQk8HOVlumI)

### Infrastructure as Code (IaC)

- **Coursera**: [Managing Cloud Infrastructure with Terraform](https://www.coursera.org/projects/managing-cloud-infrastructure-with-terraform) (Terraform is a similar tool to Pulumi and the concepts are transferable)
- **Udemy**: [Pulumi: The Complete Guide to Infrastructure as Code in C#](https://www.udemy.com/course/pulumi-the-complete-guide-to-infrastructure-as-code-in-csharp/)
- **YouTube**: [Infrastructure as Code Explained](https://www.youtube.com/watch?v=iHK65awT6zs)

### CI/CD

- **Coursera**: [IBM DevOps and Software Engineering Professional Certificate](https://www.coursera.org/professional-certificates/devops-and-software-engineering)
- **Udemy**: [The Complete CI/CD and DevOps Course](https://www.udemy.com/course/ci-cd-devops/)
- **YouTube**: [What is CI/CD?](https://www.youtube.com/watch?v=scEDHsr3APg)

