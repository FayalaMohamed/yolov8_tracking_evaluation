import os

def count_non_empty_lines_in_directory(directory_path):
    total_non_empty_lines = 0
    
    if not os.path.isdir(directory_path):
        print(f"The directory '{directory_path}' does not exist.")
        return

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"): 
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                non_empty_lines = sum(1 for line in file if line.strip())
                total_non_empty_lines += non_empty_lines
                print(f"{filename}: {non_empty_lines} non-empty lines")
    
    print(f"\nTotal non-empty lines in all .txt files: {total_non_empty_lines}")

directory_path = '/mnt/gammarus/20210608-Mathilde/ToBeLabelled'
count_non_empty_lines_in_directory(directory_path)
