#!/bin/bash

# ==============================================================================
# ADEPT Tutorial Lifecycle Management Script
#
# This script provides a safe and interactive way to manage the Docker
# resources for each tutorial chapter. It handles startup, graceful
# shutdown, and cleanup of containers and networks.
# ==============================================================================

# --- Configuration and Setup ---
# Color codes for better output
BLUE='\033[1;34m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
NC='\033[0m' # No Color
COMPOSE_BAKE=true # Bake time

# Function to print a formatted header
print_header() {
    echo -e "${BLUE}=======================================================${NC}"
    echo -e "${BLUE}  ADEPT Chapter Resource Manager (${PWD##*/}) ${NC}"
    echo -e "${BLUE}=======================================================${NC}"
}

# --- Auto-detect Docker Compose files ---
BASE_COMPOSE_FILE="docker-compose.yaml"
# List of optional overlay files to check for in order
OVERLAY_FILES=(
    "docker-compose-jupyterlab.yaml"
    "docker-compose-openwebui.yaml"
    "docker-compose-n8n.yaml"
    "docker-compose-redis.yaml"
)
COMPOSE_CMD="docker compose"

# Start with the base file, which is required
if [ -f "$BASE_COMPOSE_FILE" ]; then
    COMPOSE_CMD+=" -f $BASE_COMPOSE_FILE"
else
    echo -e "${RED}Error: Base file '$BASE_COMPOSE_FILE' not found. Exiting.${NC}"
    exit 1
fi

# Loop through potential overlay files and add them if they exist
for overlay_file in "${OVERLAY_FILES[@]}"; do
    if [ -f "$overlay_file" ]; then
        echo -e "${GREEN}Info: Overlay file '$overlay_file' found and will be used.${NC}"
        COMPOSE_CMD+=" -f $overlay_file"
    fi
done

echo -e "${GREEN}Using command: ${YELLOW}${COMPOSE_CMD}${NC}"
echo ""

# --- Cleanup Function ---
# This function is called when the script exits or is interrupted.
cleanup() {
    echo ""
    echo -e "${YELLOW}-------------------------------------------------------${NC}"
    echo -e "${YELLOW}Caught exit signal. Shutting down and cleaning up...${NC}"
    echo -e "${YELLOW}-------------------------------------------------------${NC}"

    echo -e "Stopping and removing containers..."
    $COMPOSE_CMD down --remove-orphans
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to bring down containers. Please check Docker status.${NC}"
    else
        echo -e "${GREEN}Containers stopped and removed successfully.${NC}"
    fi

    # Ask before pruning networks
    read -p "$(echo -e ${YELLOW}"Do you want to prune unused Docker networks? (y/N) "${NC})" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Pruning unused networks..."
        docker network prune -f
        echo -e "${GREEN}Network prune complete.${NC}"
    fi

    # Ask before pruning images (this is a global, aggressive action)
    read -p "$(echo -e ${RED}"DANGER: Prune ALL unused Docker images? This affects your entire system. (y/N) "${NC})" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Pruning ALL unused images..."
        docker image prune -a -f
        echo -e "${GREEN}Image prune complete.${NC}"
    fi

    echo -e "${GREEN}Cleanup finished. Goodbye!${NC}"
    # Restore cursor
    tput cnorm
}

# --- Trap Exit Signal ---
# The 'trap' command sets up a command to be executed when the script
# receives a specific signal. Here, we catch EXIT, SIGINT (Ctrl+C), and SIGTERM.
trap cleanup EXIT SIGINT SIGTERM

# --- Main Script Logic ---
print_header

# Check for existing containers for this project
echo "Checking for existing resources for this project..."
# Use 'docker compose ps' which is project-aware
RUNNING_CONTAINERS=$($COMPOSE_CMD ps -q)

if [ -n "$RUNNING_CONTAINERS" ]; then
    echo -e "${YELLOW}Warning: Found existing containers for this chapter.${NC}"
    $COMPOSE_CMD ps
    read -p "$(echo -e ${YELLOW}"Do you want to tear down these existing resources before starting? (y/N) "${NC})" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Tearing down existing resources..."
        $COMPOSE_CMD down --remove-orphans
        echo -e "${GREEN}Teardown complete.${NC}"
    else
        echo -e "${RED}Aborted by user. Please manually manage existing resources before running this script again.${NC}"
        # Untrap the cleanup function before exiting to prevent it from running
        trap - EXIT SIGINT SIGTERM
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}Starting services. Press ${YELLOW}Ctrl+C${GREEN} to stop and clean up.${NC}"
echo -e "Building images if necessary and starting containers..."
echo ""

# Hide cursor for cleaner log output
tput civis

# Run docker-compose in the foreground. The script will wait here until
# the user presses Ctrl+C, which will be caught by our trap.
$COMPOSE_CMD up --build --remove-orphans

# The script will only reach here if 'docker compose up' exits on its own,
# which is unlikely for server processes. The trap handles the main exit path.
# The cleanup function will be called automatically due to the EXIT trap.
