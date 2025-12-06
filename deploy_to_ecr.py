import subprocess
import sys

# --- Configuration ---
REGION = "ca-central-1"
ACCOUNT_ID = "503561425608"
REPO_NAME = "severed/optics/strategy-1"  # Local image name
ECR_REPO_NAME = "severed/optics/strategy-1" # Remote ECR repo name

# Derived URLs
ECR_REGISTRY = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com"
FULL_IMAGE_URI = f"{ECR_REGISTRY}/{ECR_REPO_NAME}:latest"

def run_command(command, capture_output=False, input_text=None):
    """Helper to run shell commands with error handling."""
    try:
        print(f"\n[EXEC] {command}")

        if input_text:
            # Pass input to stdin (used for docker login)
            subprocess.run(
                command,
                input=input_text.encode(),
                shell=True,
                check=True
            )
        elif capture_output:
            # Capture output (used to get the password)
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True
            )
            return result.stdout.decode().strip()
        else:
            # Stream output to console (used for build/push)
            subprocess.run(command, shell=True, check=True)

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed with exit code {e.returncode}")
        sys.exit(1)

def main():
    print("=" * 60)
    print(f"DEPLOYING TO ECR: {FULL_IMAGE_URI}")
    print("=" * 60)

    # 1. Retrieve Authentication Token
    print("\n1. Retrieving ECR Login Password...")
    password_cmd = f"aws ecr get-login-password --region {REGION}"
    password = run_command(password_cmd, capture_output=True)

    # 2. Authenticate Docker Client
    print("\n2. Logging into Docker...")
    login_cmd = f"docker login --username AWS --password-stdin {ECR_REGISTRY}"
    run_command(login_cmd, input_text=password)

    # 3. Build Docker Image
    print("\n3. Building Docker Image (this may take a while)...")
    # We use --platform linux/amd64 to ensure compatibility with Fargate
    build_cmd = f"docker build --platform linux/amd64 -t {REPO_NAME} ."
    run_command(build_cmd)

    # 4. Tag Image
    print("\n4. Tagging Image...")
    tag_cmd = f"docker tag {REPO_NAME}:latest {FULL_IMAGE_URI}"
    run_command(tag_cmd)

    # 5. Push to ECR
    print("\n5. Pushing to ECR...")
    push_cmd = f"docker push {FULL_IMAGE_URI}"
    run_command(push_cmd)

    print("\n" + "=" * 60)
    print("SUCCESS: Image deployed to ECR")
    print("=" * 60)

if __name__ == "__main__":
    # Check dependencies
    subprocess.run("docker --version", shell=True, check=True, stdout=subprocess.DEVNULL)
    subprocess.run("aws --version", shell=True, check=True, stdout=subprocess.DEVNULL)

    main()
