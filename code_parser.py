import os

# Set the root directory of the repo and the output file
repo_path = os.path.abspath('')
output_file = "all_code_combined.txt"

# Extensions to include (you can customize this)
code_extensions = {'.py', '.kts', '.txt', '.bat', '.git', '.md', '.gradle.kts', '.json'}

def is_code_file(filename):
    return any(filename.endswith(ext) for ext in code_extensions)

with open(output_file, 'w', encoding='utf-8') as outfile:
    for root, _, files in os.walk(repo_path):
        for file in files:
            if is_code_file(file):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        outfile.write(f"\n\n--- {full_path} ---\n\n")
                        outfile.write(f.read())
                except Exception as e:
                    print(f"Could not read {full_path}: {e}")
