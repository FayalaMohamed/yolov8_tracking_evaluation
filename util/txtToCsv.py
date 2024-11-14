def get_max_widths(lines):
    # Split each line into columns and find the maximum width for each column
    columns = [line.split() for line in lines]
    max_widths = [max(len(col[i]) for col in columns if len(col) > i) for i in range(len(columns[0]))]
    return max_widths

with open('TrackEval/data/trackers/mot_challenge/MOT20-train/YOLO11x-ft/pedestrian_summary.txt', 'r') as f:
    lines = f.readlines()

max_widths = get_max_widths(lines)

# Write the contents to a .csv file with aligned columns
with open('TrackEval/data/trackers/mot_challenge/MOT20-train/YOLO11x-ft/pedestrian_summary.csv', 'w') as f:
    for line in lines:
        columns = line.split()
        padded_columns = [col.ljust(max_widths[i]) for i, col in enumerate(columns)]
        csv_line = ','.join(padded_columns)
        f.write(csv_line + '\n')