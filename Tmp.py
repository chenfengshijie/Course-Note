import os
import shutil


def copy_folders_with_keywords(source_folder, destination_folder, keywords):
    # 遍历源文件夹中的所有子文件夹
    for root, dirs, files in os.walk(source_folder):
        # 检查关键字是否存在于当前文件夹的路径中
        if any(keyword in root for keyword in keywords):
            # 构建目标文件夹的路径
            destination_path = os.path.join(
                destination_folder, os.path.relpath(root, source_folder)
            )
            # 复制文件夹到目标位置
            shutil.copytree(root, destination_path)


# 源文件夹路径
source_folder = "此电脑\Redmi K30i 5G\内部存储设备\微云保存的文件"
# 目标文件夹路径
destination_folder = "I:\hhhh\白丝\森萝财团.7z"
# 关键字列表
keywords = ["森萝财团"]

# 调用函数复制文件夹
copy_folders_with_keywords(source_folder, destination_folder, keywords)
