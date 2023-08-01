import os, sys
import subprocess
from pathlib import Path

MESHSIZES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
original_paths = [Path(f"../data/original/x{i}y{i}/test_data_new.json") for i in MESHSIZES]
sides_path = Path(f"../data/orderrouting_sides_long/test_data.json")
full_path = Path(f"../data/orderrouting_long/test_data.json")

def main():
    extension = ".exe" if os.name == "nt" else ""
    rust_router = Path(f"../rust/o-routing/target/release/o-routing{extension}")
    for path in original_paths:
        subprocess.run([rust_router, path, '-c', f"{os.cpu_count()}"])
    subprocess.run([rust_router, sides_path, '-c', f"{os.cpu_count()}"])
    subprocess.run([rust_router, full_path, '-c', f"{os.cpu_count()}"])

if __name__ == "__main__":
    main()