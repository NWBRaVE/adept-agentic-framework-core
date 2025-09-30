#!/usr/bin/env python3
import os
import aws_cdk as cdk
from dotenv import load_dotenv

from stacks.cicd_stack import CicdStack
from stacks.ecs_stack import EcsStack

# Load environment variables from .env file
load_dotenv()

app = cdk.App()

# --- Configuration ---
project_name = os.getenv("PROJECT_NAME", "AgenticFramework")
aws_env = cdk.Environment(
    account=os.getenv("AWS_ACCOUNT"),
    region=os.getenv("AWS_REGION")
)

vpc_id = os.getenv("VPC_ID")
private_subnet_ids = os.getenv("PRIVATE_SUBNET_IDS", "").split(',')
domain_name = os.getenv("DOMAIN_NAME")
hosted_zone_id = os.getenv("HOSTED_ZONE_ID")

if not all([aws_env.account, aws_env.region, vpc_id, private_subnet_ids[0]]):
    raise ValueError(
        "Please set AWS_ACCOUNT, AWS_REGION, VPC_ID, and PRIVATE_SUBNET_IDS in your .env file."
    )

# --- Stack Definitions ---

# 1. CI/CD Stack: Creates ECR repos and the pipeline to build/push images.
cicd_stack = CicdStack(
    app,
    f"{project_name}CicdStack",
    env=aws_env
)

# 2. ECS Application Stack: Deploys the services to ECS Fargate.
EcsStack(
    app,
    f"{project_name}EcsStack",
    vpc_id=vpc_id,
    subnet_ids=private_subnet_ids,
    ecr_repos=cicd_stack.ecr_repos,
    domain_name=domain_name,
    hosted_zone_id=hosted_zone_id,
    env=aws_env
)

app.synth()

