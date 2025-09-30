from constructs import Construct
from aws_cdk import (
    Stack,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_efs as efs,
    aws_iam as iam,
    aws_elasticloadbalancingv2 as elbv2,
    aws_servicediscovery as servicediscovery,
    aws_route53 as route53,
    aws_route53_targets as route53_targets,
    aws_certificatemanager as acm,
    aws_autoscaling as autoscaling,
    RemovalPolicy,
    CfnParameter,
    CfnOutput,
    Duration,
)

class EcsStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, vpc_id: str, subnet_ids: list, ecr_repos: dict, domain_name: str, hosted_zone_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # --- Define CfnParameters to receive image URIs from CodeBuild ---
        mcp_server_image = CfnParameter(self, "mcp_server_image", type="String", description="MCP Server image URI").value_as_string
        streamlit_app_image = CfnParameter(self, "streamlit_app_image", type="String", description="Streamlit App image URI").value_as_string
        hpc_mcp_server_image = CfnParameter(self, "hpc_mcp_server_image", type="String", description="HPC MCP Server image URI").value_as_string
        sandbox_mcp_server_image = CfnParameter(self, "sandbox_mcp_server_image", type="String", description="Sandbox MCP Server image URI").value_as_string

        # 1. VPC and Cluster Setup
        vpc = ec2.Vpc.from_vpc_attributes(self, "ImportedVpc",
            vpc_id=vpc_id,
            availability_zones=self.availability_zones,
            private_subnet_ids=subnet_ids
        )

        # Create an IAM role for the EC2 instances that will join the cluster
        ec2_instance_role = iam.Role(self, "EcsInstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonEC2ContainerServiceforEC2Role")
            ]
        )

        # Create an Auto Scaling Group for EC2-based tasks (like the sandbox)
        asg = autoscaling.AutoScalingGroup(self, "EcsEc2Asg",
            vpc=vpc,
            instance_type=ec2.InstanceType("t3.large"),
            machine_image=ecs.EcsOptimizedImage.amazon_linux2(),
            role=ec2_instance_role,
            min_capacity=1,
            max_capacity=2,
        )

        # Create the ECS cluster and add the ASG as a capacity provider
        cluster = ecs.Cluster(self, "AgenticCluster", vpc=vpc)
        capacity_provider = ecs.AsgCapacityProvider(self, "AsgCapacityProvider",
            auto_scaling_group=asg
        )
        cluster.add_asg_capacity_provider(capacity_provider)

        # Add a Cloud Map namespace for service discovery
        namespace = servicediscovery.PrivateDnsNamespace(
            self, "DnsNamespace",
            name="agentic.local",
            vpc=vpc
        )

        # 2. Shared EFS File System for persistent data
        efs_filesystem = efs.FileSystem(
            self, "SharedEfs",
            vpc=vpc,
            performance_mode=efs.PerformanceMode.GENERAL_PURPOSE,
            throughput_mode=efs.ThroughputMode.BURSTING,
            removal_policy=RemovalPolicy.DESTROY
        )

        data_access_point = efs.AccessPoint(self, "DataAccessPoint", file_system=efs_filesystem, path="/data",
            create_acl=efs.Acl(owner_uid="1000", owner_gid="1000", permissions="755"),
            posix_user=efs.PosixUser(uid="1000", gid="1000"))

        blast_db_access_point = efs.AccessPoint(self, "BlastDbAccessPoint", file_system=efs_filesystem, path="/blast_databases",
            create_acl=efs.Acl(owner_uid="1000", owner_gid="1000", permissions="755"),
            posix_user=efs.PosixUser(uid="1000", gid="1000"))

        # 3. Security Groups
        alb_sg = ec2.SecurityGroup(self, "AlbSg", vpc=vpc, allow_all_outbound=True)
        alb_sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(80), "Allow HTTP from anywhere")
        alb_sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(443), "Allow HTTPS from anywhere")

        ecs_sg = ec2.SecurityGroup(self, "EcsSg", vpc=vpc, allow_all_outbound=True)
        ecs_sg.add_ingress_rule(alb_sg, ec2.Port.tcp(8501), "Allow ALB to access Streamlit")
        ecs_sg.add_ingress_rule(ecs_sg, ec2.Port.all_tcp(), "Allow intra-service communication")
        asg.add_security_group(ecs_sg)
        efs_filesystem.connections.allow_default_port_from(ecs_sg)

        # 4. IAM Roles
        task_execution_role = iam.Role(self, "EcsTaskExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonECSTaskExecutionRolePolicy")])

        # 5. Application Load Balancer
        alb = elbv2.ApplicationLoadBalancer(self, "Alb", vpc=vpc, internet_facing=True, security_group=alb_sg)
        #listener = alb.add_listener("HttpListener", port=80, open=True)
        CfnOutput(self, "LoadBalancerDNS", value=alb.load_balancer_dns_name)

        # --- Custom Domain and HTTPS Setup ---
        https_listener = None
        if domain_name and hosted_zone_id:
            # Look up the hosted zone in Route 53
            hosted_zone = route53.HostedZone.from_hosted_zone_attributes(self, "HostedZone",
                hosted_zone_id=hosted_zone_id,
                zone_name=domain_name.split('.', 1)[-1]
            )

            # Create an ACM certificate and validate it with DNS
            certificate = acm.Certificate(self, "SslCertificate",
                domain_name=domain_name,
                validation=acm.CertificateValidation.from_dns(hosted_zone)
            )

            # Create an HTTPS listener and attach the certificate
            https_listener = alb.add_listener("HttpsListener",
                port=443,
                certificates=[certificate],
                ssl_policy=elbv2.SslPolicy.RECOMMENDED_TLS
            )
            # Redirect all HTTP traffic to HTTPS
            alb.add_redirect(source_protocol=elbv2.ApplicationProtocol.HTTP, source_port=80, target_protocol=elbv2.ApplicationProtocol.HTTPS, target_port=443)

            # Create a Route 53 Alias record pointing to the ALB
            route53.ARecord(self, "AliasRecord", zone=hosted_zone, record_name=domain_name, target=route53.RecordTarget.from_alias(route53_targets.LoadBalancerTarget(alb)))
            CfnOutput(self, "AppURL", value=f"https://{domain_name}")

        # --- Service Definitions ---
        services_config = {
            "mcp_server": {"image": mcp_server_image, "port": 8080, "cpu": 1024, "memory": 2048, "volumes": [{"name": "data", "access_point": data_access_point, "container_path": "/app/data"}]},
            "streamlit_app": {"image": streamlit_app_image, "port": 8501, "cpu": 1024, "memory": 2048, "volumes": [{"name": "data", "access_point": data_access_point, "container_path": "/app/data"}], "environment": {"MCP_SERVER_URL": "http://mcp-server.agentic.local:8080/mcp", "HPC_MCP_SERVER_URL": "http://hpc-mcp-server.agentic.local:8081", "SANDBOX_MCP_SERVER_URL": "http://sandbox-mcp-server.agentic.local:8082/mcp", "MCP_SERVER_URL_FOR_LANGCHAIN": "http://mcp-server.agentic.local:8080/mcp", "SANDBOX_MCP_SERVER_URL_FOR_LANGCHAIN": "http://sandbox-mcp-server.agentic.local:8082/mcp"}},
            "hpc_mcp_server": {"image": hpc_mcp_server_image, "port": 8081, "cpu": 2048, "memory": 4096, "volumes": [{"name": "data", "access_point": data_access_point, "container_path": "/app/data"}, {"name": "blastdb", "access_point": blast_db_access_point, "container_path": "/blast_databases"}], "environment": {"BLASTDB": "/blast_databases", "HPC_MCP_SERVER_PORT": "8081"}},
            "sandbox_mcp_server": {"image": sandbox_mcp_server_image, "port": 8082, "cpu": 1024, "memory": 2048, "volumes": [{"name": "data", "access_point": data_access_point, "container_path": "/app/data"}, {"name": "docker_sock", "host_path": "/var/run/docker.sock", "container_path": "/var/run/docker.sock"}], "environment": {"SANDBOX_MCP_SERVER_PORT": "8082"}, "ec2_required": True}
        }

        for name, config in services_config.items():
            task_role = iam.Role(self, f"{name}TaskRole", assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"))
            # In a real app, add specific permissions to S3, etc. here
            # ecr_repos[name].grant_pull(task_role)

            cloud_map_options = ecs.CloudMapOptions(
                name=name.replace("_", "-"),
                cloud_map_namespace=namespace
            )

            if config.get("ec2_required"):
                # --- EC2 Task Definition and Service ---
                task_definition = ecs.Ec2TaskDefinition(self, f"{name}TaskDef",
                    execution_role=task_execution_role,
                    task_role=task_role,
                    network_mode=ecs.NetworkMode.AWS_VPC)

                for vol_info in config.get("volumes", []):
                    if "host_path" in vol_info:
                        task_definition.add_volume(name=vol_info["name"], host=ecs.Host(source_path=vol_info["host_path"]))
                    elif "access_point" in vol_info:
                        task_definition.add_volume(name=vol_info["name"], efs_volume_configuration=efs.EfsVolumeConfiguration(
                            file_system_id=efs_filesystem.file_system_id,
                            transit_encryption="ENABLED",
                            access_point_id=vol_info["access_point"].access_point_id
                        ))

                container = task_definition.add_container(f"{name}Container",
                    image=ecs.ContainerImage.from_registry(config["image"]),
                    logging=ecs.LogDrivers.aws_logs(stream_prefix=name),
                    environment=config.get("environment", {}),
                    port_mappings=[ecs.PortMapping(container_port=config["port"])],
                    memory_limit_mib=config["memory"],
                    cpu=config["cpu"],
                    privileged=True
                )

                for vol_info in config.get("volumes", []):
                    container.add_mount_points(ecs.MountPoint(
                        container_path=vol_info["container_path"],
                        source_volume=vol_info["name"],
                        read_only=False
                    ))

                service = ecs.Ec2Service(self, f"{name}Service",
                    cluster=cluster,
                    task_definition=task_definition,
                    cloud_map_options=cloud_map_options,
                    desired_count=1,
                    capacity_provider_strategies=[ecs.CapacityProviderStrategy(
                        capacity_provider=capacity_provider.capacity_provider_name,
                        weight=1
                    )]
                )
            else:
                # --- Fargate Task Definition and Service ---
                task_volumes = []
                container_mount_points = []
                if "volumes" in config:
                    for vol_info in config["volumes"]:
                        efs_volume_config = efs.EfsVolumeConfiguration(
                            file_system_id=efs_filesystem.file_system_id,
                            transit_encryption="ENABLED",
                            access_point_id=vol_info["access_point"].access_point_id
                        )
                        task_volumes.append(ecs.Volume(name=vol_info["name"], efs_volume_configuration=efs_volume_config))
                        container_mount_points.append(ecs.MountPoint(
                            container_path=vol_info["container_path"],
                            source_volume=vol_info["name"],
                            read_only=False
                        ))

                task_definition = ecs.FargateTaskDefinition(self, f"{name}TaskDef",
                    cpu=config["cpu"],
                    memory_limit_mib=config["memory"],
                    execution_role=task_execution_role,
                    task_role=task_role,
                    volumes=task_volumes
                )

                container = task_definition.add_container(f"{name}Container",
                    image=ecs.ContainerImage.from_registry(config["image"]),
                    logging=ecs.LogDrivers.aws_logs(stream_prefix=name),
                    environment=config.get("environment", {}),
                    port_mappings=[ecs.PortMapping(container_port=config["port"])]
                )
                container.add_mount_points(*container_mount_points)

                service = ecs.FargateService(self, f"{name}Service",
                    cluster=cluster,
                    task_definition=task_definition,
                    vpc_subnets=ec2.SubnetSelection(subnets=vpc.private_subnets),
                    security_groups=[ecs_sg],
                    cloud_map_options=cloud_map_options
                )

                if name == "streamlit_app":
                    # If HTTPS is configured, add target to HTTPS listener, otherwise create a default HTTP listener
                    listener_to_use = https_listener if https_listener else alb.add_listener("HttpListener", port=80)

                    listener_to_use.add_targets("StreamlitTarget",
                            port=8501,
                        targets=[service],
                        health_check=elbv2.HealthCheck(
                            path="/_stcore/health",
                            interval=Duration.seconds(30),
                            timeout=Duration.seconds(5),
                        )
                    )


