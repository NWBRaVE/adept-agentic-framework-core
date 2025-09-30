# Running the Test Suite

This directory contains the test suites for the services in this chapter, primarily focused on the `agent_gateway`.

## Prerequisites

- Docker must be installed and running on your system.

## Running Tests via Docker

The provided `Dockerfile.tests` creates a self-contained environment with all the necessary dependencies to run the test suite.

### 1. Build the Test Image

First, build the Docker image that will be used to run the tests. Execute this command from the root of the `chapter-06-advanced-multi-agent-orchestration` directory:

```bash
docker build -t agentic-framework-tests:latest -f Dockerfile.tests .
```

This command builds the `Dockerfile.tests` file and tags the resulting image as `agentic-framework-tests:latest`.

### 2. Run the Test Container

Once the image is built, run the tests by executing the following command:

```bash
docker run --rm agentic-framework-tests:latest
```

This command starts a container from the image you just built. The `CMD` instruction in the `Dockerfile.tests` is configured to run `pytest`. The container will execute all the tests, print the output to your terminal, and then automatically remove itself upon completion (`--rm`).

You should see the output from `pytest`, indicating the status of each test (e.g., passed, failed, skipped).


### New Test Cases

In addition to the existing tests, the test suite now includes:

    -   `test_tutorial_http_tool_registration`: Verifies the successful registration of an HTTP-based tool and simulates its invocation by the agent, ensuring the
     gateway correctly handles HTTP requests.
