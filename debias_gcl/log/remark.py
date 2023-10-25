import os

# 路径到你的目录
directory_path = './'
# 输出文件
output_file = 'output.txt'

with open(output_file, 'w') as out_f:
    # 遍历目录下的所有文件
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        # 检查是否为文件
        if os.path.isfile(file_path):
            with open(file_path, 'r') as in_f:
                lines = in_f.readlines()
                # 获取最后一行
                last_line = lines[-1] if lines else ''
                # 写入输出文件
                out_f.write(filename + '\n')
                out_f.write(last_line + '\n')

print(f"Processed all files in {directory_path} and saved to {output_file}.")

