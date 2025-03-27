import subprocess
import os

def run_pytest():
    project_root = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(project_root, "test")

    print("Running tests with pytest...\n")
    subprocess.run(["pytest", test_dir, "-v", "--disable-warnings"])

if __name__ == "__main__":
    run_pytest()
