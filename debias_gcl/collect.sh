#!/bin/bash

# 清空或创建result.txt文件
echo "" > result.txt

# 遍历当前目录下的所有以env开头的文件
for file in env*; do
    # 检查该条目是否为文件
    if [ -f "$file" ]; then
        # 打印文件名并添加到result.txt文件中
        echo "$file:" >> result.txt
        # 打印文件的最后4行内容并添加到result.txt文件中
        tail -n 4 "$file" >> result.txt
        # 在不同文件内容之间添加一个空行，方便阅读
        echo "" >> result.txt
    fi
done
