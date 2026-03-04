#!/bin/bash
# 切换到工作目录

#Get script path and go there (in case script is lauched from another dir)
#Get script path and go there (in case script is lauched from another dir)
SWD="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
echo -e "\e[34m=== 调试信息 ===\e[0m"
echo -e "\e[32m当前作业目录是:\e[0m $SWD"
echo -e "\e[32m脚本路径是:\e[0m $0"
echo -e "\e[32m切换前所在目录:\e[0m $(pwd)"
cd "$SWD"
echo -e "\e[32m切换后所在目录:\e[0m $(pwd)"
echo -e "\e[34m================\e[0m"