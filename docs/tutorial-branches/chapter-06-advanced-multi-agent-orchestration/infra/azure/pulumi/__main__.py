import pulumi
import pulumi_azure_native as azure_native
from pulumi_azure_native import resources, containerservice, authorization

# --- Configuration ---
config = pulumi.Config()

# --- Resource Group ---
resource_group = resources.ResourceGroup("agentic-rg")

# --- Azure Container Registry (ACR) ---
container_registry = azure_native.containerregistry.Registry("agenticAcr",
    resource_group_name=resource_group.name,
    sku=azure_native.containerregistry.SkuArgs(name="Basic"),
    admin_user_enabled=True,
)

# --- AKS Cluster ---

# Create a system-assigned managed identity for the AKS cluster
aks_identity = azure_native.managedidentity.UserAssignedIdentity("aksIdentity",
    resource_group_name=resource_group.name,
)

# Grant the AKS identity the AcrPull role on the container registry
# This allows nodes in the AKS cluster to pull images from ACR
acr_pull_assignment = authorization.RoleAssignment("acrPullAssignment",
    principal_id=aks_identity.principal_id,
    principal_type=authorization.PrincipalType.SERVICE_PRINCIPAL,
    role_definition_id=pulumi.Output.concat(
        "/subscriptions/", authorization.get_subscription().subscription_id,
        "/providers/Microsoft.Authorization/roleDefinitions/7f951a00-0792-461c-b1e5-72e06c714de2" # AcrPull role
    ),
    scope=container_registry.id,
)

aks_cluster = containerservice.ManagedCluster("agenticAksCluster",
    resource_group_name=resource_group.name,
    # Define the cluster properties
    agent_pool_profiles=[{
        "count": 2,  # Number of nodes
        "max_pods": 110,
        "mode": "System",
        "name": "agentpool",
        "os_disk_size_gb": 30,
        "os_type": "Linux",
        "type": "VirtualMachineScaleSets",
        "vm_size": "Standard_DS2_v2",
    }],
    dns_prefix=resource_group.name,
    enable_rbac=True,
    kubernetes_version="1.28.5",
    # Assign the managed identity to the cluster
    identity=containerservice.ManagedClusterIdentityArgs(
        type=containerservice.ResourceIdentityType.USER_ASSIGNED,
        user_assigned_identities={
            aks_identity.id: {},
        },
    ),
    # Ensure the role assignment is created before the cluster
    opts=pulumi.ResourceOptions(depends_on=[acr_pull_assignment]),
)

# --- Exports ---

# Export the kubeconfig to access the cluster
creds = pulumi.Output.all(resource_group.name, aks_cluster.name).apply(
    lambda args: containerservice.list_managed_cluster_user_credentials(
        resource_group_name=args[0],
        resource_name=args[1],
    )
)
kubeconfig = creds.kubeconfigs[0].value.apply(lambda enc: enc.decode("utf-8"))

pulumi.export("resourceGroupName", resource_group.name)
pulumi.export("acrLoginServer", container_registry.login_server)
pulumi.export("aksClusterName", aks_cluster.name)
pulumi.export("kubeconfig", pulumi.Output.secret(kubeconfig))