import os

def last_four_lines_of_files(directory, output_file):
    with open(output_file, 'w') as out_f:
        for filename in os.listdir(directory):
            if filename.startswith("output"):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as in_f:
                        lines = in_f.readlines()
                        if len(lines) >= 4:
                            out_f.write(filename + '\n')
                            out_f.write(''.join(lines[-4:]))
                            out_f.write('\n')  # Separate entries
                except UnicodeDecodeError:
                    print(f"Skipping file {filename} due to encoding issues.")

directory = '.'  # Current directory
output_file = 'result.txt'
last_four_lines_of_files(directory, output_file)

