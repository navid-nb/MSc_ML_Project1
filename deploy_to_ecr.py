import subprocess
import sys

# --- Configuration ---
REGION = "ca-central-1"
ACCOUNT_ID = "503561425608"
REPO_NAME = "severed/optics/strategy-1"  # Local image name (used for reference)
ECR_REPO_NAME = "severed/optics/strategy-1" # Remote ECR repo name

# Derived URLs
ECR_REGISTRY = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com"
FULL_IMAGE_URI = f"{ECR_REGISTRY}/{ECR_REPO_NAME}:latest"

def run_command(command, capture_output=False, input_text=None, allow_fail=False):
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
            subprocess.run(command, shell=True, check=not allow_fail)

    except subprocess.CalledProcessError as e:
        if not allow_fail:
            print(f"\n[ERROR] Command failed with exit code {e.returncode}")
            sys.exit(1)

def main():
    print("=" * 60)
    print(f"DEPLOYING TO ECR (MULTI-ARCH): {FULL_IMAGE_URI}")
    print("=" * 60)

    # 1. Retrieve Authentication Token
    print("\n1. Retrieving ECR Login Password...")
    password_cmd = f"aws ecr get-login-password --region {REGION}"
    password = run_command(password_cmd, capture_output=True)

    # 2. Authenticate Docker Client
    print("\n2. Logging into Docker...")
    login_cmd = f"docker login --username AWS --password-stdin {ECR_REGISTRY}"
    run_command(login_cmd, input_text=password)

    # 3. Setup Buildx (New Step)
    print("\n3. Setting up Docker Buildx...")
    # We attempt to create a new builder. If one already exists or is in use,
    # we allow it to fail gracefully and proceed, assuming the environment is ready.
    subprocess.run("docker buildx create --use", shell=True)

    # 4. Build and Push (Combined Step)
    print("\n4. Building and Pushing Multi-Arch Image (amd64 + arm64)...")
    print("   Note: This pushes directly to ECR. It will not show up in 'docker images' locally.")

    buildx_cmd = (
        f"docker buildx build "
        f"--platform linux/amd64,linux/arm64 "
        f"-t {FULL_IMAGE_URI} "
        f"--push ."
    )
    run_command(buildx_cmd)

    print("\n" + "=" * 60)
    print("SUCCESS: Multi-architecture image deployed to ECR")
    print("=" * 60)

if __name__ == "__main__":
    # Check dependencies
    try:
        subprocess.run("docker --version", shell=True, check=True, stdout=subprocess.DEVNULL)
        subprocess.run("aws --version", shell=True, check=True, stdout=subprocess.DEVNULL)
    except Exception:
        print("[ERROR] Please ensure 'docker' and 'aws-cli' are installed.")
        sys.exit(1)

    main()