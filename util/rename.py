import os

def rename_files_in_directory(directory_path):
    if not os.path.isdir(directory_path):
        print(f"The directory '{directory_path}' does not exist.")
        return

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        if not os.path.isfile(file_path):
            continue

        new_filename = '05-' + filename

        new_file_path = os.path.join(directory_path, new_filename)

        os.rename(file_path, new_file_path)
        print(f"Renamed: {filename} -> {new_filename}")


rename_files_in_directory("./MOT20/train/MOT20-05/img1")
rename_files_in_directory("./MOT20/train/MOT20-05/labels")
