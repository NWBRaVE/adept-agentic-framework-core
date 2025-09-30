import sys
import json

def main():
    input_data = json.loads(sys.stdin.read())
    message = input_data.get("message", "")
    response = {"status": "success", "echo": f"Docker tool received: {message}"}
    sys.stdout.write(json.dumps(response))

if __name__ == "__main__":
    main()