import os

def fix_labels(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r") as file:
                lines = file.readlines()
            
            with open(file_path, "w") as file:
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        parts[0] = "0"
                        file.write(" ".join(parts) + "\n")
            print(f"Fixed label: {filename}")

label_dir = "./datasetMOT20/valid/labels"
fix_labels(label_dir)
