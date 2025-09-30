# Agentic Framework Helm Chart

This Helm chart deploys the complete Agentic Framework, a multi-service application designed for building and running AI-powered scientific workflows.

## Architecture

The Helm chart deploys the following core services:

-   **`mcp-server`**: The main MCP server that hosts general-purpose tools for data analysis, RAG, and interaction with scientific databases (UniProt, PubChem).
-   **`hpc-mcp-server`**: A dedicated MCP server for High-Performance Computing (HPC) tasks, such as running Nextflow pipelines for BLAST or video processing.
-   **`sandbox-mcp-server`**: A specialized MCP server for executing code in a secure, isolated environment. It can be configured to use a Docker-based or a Kubernetes-native backend.
-   **`streamlit-app`**: The main user interface, providing a chat-based harness for interacting with the Langchain agent.

## Prerequisites

-   A running Kubernetes cluster (e.g., Docker Desktop, Minikube, or a cloud/on-prem provider).
-   `kubectl` configured to communicate with your cluster.
-   `helm` (version 3+) installed.
-   If deploying to a cluster with a private container registry, you must have credentials and an `imagePullSecret`.

## Configuration

The deployment is configured primarily through the `infra/helm/agentic-framework/values.yaml` file. For specific environments, `values-local.yaml` (for local Docker Desktop) and `values-azure.yaml` are provided as overrides.

### Key Configuration Parameters (`values.yaml`)

| Parameter                               | Description                                                                                                                                                           | Default                                                                    |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `namespace`                             | The Kubernetes namespace for all resources.                                                                                                                           | `class`                                                                    |
| `global.image.pullPolicy`               | The image pull policy. Use `Never` for local development with locally built images, `Always` for production.                                                          | `Always`                                                                   |
| `imagePullSecrets`                      | A list of secrets required to pull images from a private registry.                                                                                                    | `[{ name: "my-registry-secret" }]`                                          |
| `secrets.create`                        | If `true`, a new secret is created from `secrets.data`. If `false`, `secrets.existingSecretName` must be provided.                                                    | `true`                                                                     |
| `secrets.existingSecretName`            | The name of a pre-existing secret containing required API keys and credentials.                                                                                       | `""`                                                                       |
| `sandbox_mcp_server.backend`            | The execution backend for the sandbox. `kubernetes` (recommended for clusters) creates ephemeral pods. `docker` (for local dev) uses Docker-in-Docker.                 | `kubernetes`                                                               |
| `persistence.*.storageClassName`        | The `StorageClass` for PersistentVolumeClaims. For on-prem/cloud, this must be set to a valid provisioner (e.g., `mscmsc-nfs`). For local dev, it can be empty.         | `mscmsc-nfs`                                                               |
| `ingress.enabled`                       | If `true`, an Ingress resource is created to expose the Streamlit UI.                                                                                                   | `true`                                                                     |
| `ingress.hosts[0].host`                 | The hostname for the Ingress rule.                                                                                                                                    | `agentic.emsl.pnl.gov`                                                     |
| `global.proxy.*`                        | Configuration for outbound HTTP/HTTPS proxies if required by the cluster environment.                                                                                 | (pre-configured for PNNL)                                                  |

---

## Deployment

### 1. Prepare Secrets

The framework requires API keys and other credentials, which should be managed via Kubernetes Secrets.

**Option A: Create Secret from `.env` file (Recommended for local)**

Create a `.env` file in the project root. Then, create a secret in your target namespace:

```bash
kubectl create secret generic my-app-env-secret --from-env-file=.env --namespace <your-namespace>
```

In your `values.yaml` or `values-local.yaml`, set:

```yaml
secrets:
  create: false
  existingSecretName: "my-app-env-secret"
```

**Option B: Create Secret for Private Registry**

If your container images are in a private registry, create an `imagePullSecret`:

```bash
kubectl create secret docker-registry my-registry-secret \
  --docker-server=<your-registry-server> \
  --docker-username=<your-username> \
  --docker-password=<your-password> \
  --namespace <your-namespace>
```

Ensure your `values.yaml` references this secret:

```yaml
imagePullSecrets:
  - name: my-registry-secret
```

### 2. Install the Helm Chart

Navigate to the project root directory.

**For Local Deployment (Docker Desktop):**

Use the `values-local.yaml` file, which is pre-configured for local development (e.g., `imagePullPolicy: Never`, `sandbox_mcp_server.backend: docker`).

```bash
# The helm-manage_local_k8s.sh script can automate this process.
# Or, run manually:
helm install my-agentic-release ./infra/helm/agentic-framework/ \
  --namespace agentic-framework \
  --create-namespace \
  -f ./infra/helm/agentic-framework/values-local.yaml
```

**For On-Prem/Cloud Deployment (e.g., EMSL RZR Cluster):**

Use the default `values.yaml`, which is configured for the `class` namespace and private registry. Ensure your `kubectl` context is correctly configured.

```bash
# The helm-manage_emsl_k8s.sh script can automate this process.
# Or, run manually:
helm install my-agentic-release ./infra/helm/agentic-framework/ \
  --namespace class \
  -f ./infra/helm/agentic-framework/values.yaml
```

### 3. Verify the Deployment

Check the status of the pods and persistent volume claims:

```bash
kubectl get pods,pvc --namespace <your-namespace>
```

All pods should eventually be in the `Running` state, and all PVCs should be `Bound`.

---

## Accessing the Application

**Using `port-forward` (for local or direct access):**

The most reliable way to access the Streamlit UI is by forwarding a local port to the service in the cluster. The command is provided in the Helm chart's notes after a successful installation.

```bash
# Get the service name (adjust if you used a different release name)
export SERVICE_NAME=my-agentic-release-agentic-framework-streamlit-app

# Forward the port
kubectl port-forward --namespace <your-namespace> svc/$SERVICE_NAME 8501:8501
```

Now, open your web browser and navigate to **http://127.0.0.1:8501**.

**Using Ingress:**

If you enabled the Ingress (`ingress.enabled=true`) and have an Ingress controller running in your cluster, you can access the application at the host you configured in `values.yaml` (e.g., `http://agentic.emsl.pnl.gov`).

---

## Debugging

**Tailing Logs:**

To view the logs from all services at once, use a label selector. Open a new terminal and run:

```bash
kubectl logs -f --namespace <your-namespace> -l app.kubernetes.io/instance=my-agentic-release --all-containers=true --tail=100
```

**Describing a Pod:**

If a pod is stuck in a state like `Pending` or `CrashLoopBackOff`, use `describe` to get detailed events and find the root cause:

```bash
kubectl describe pod <full-pod-name> --namespace <your-namespace>
```

---

## Cleanup

To uninstall the Helm release and delete all associated resources (Deployments, Services, PVCs, etc.), run:

```bash
helm uninstall my-agentic-release --namespace <your-namespace>
```
