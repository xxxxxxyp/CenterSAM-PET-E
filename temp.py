import os
import re

def rename_nii_gz_files(target_dir):
    """
    遍历指定目录下的所有nii.gz文件，重命名为仅保留前四位数字
    
    Args:
        target_dir (str): 目标目录路径
    """
    # 检查目录是否存在
    if not os.path.isdir(target_dir):
        print(f"错误：目录 {target_dir} 不存在！")
        return
    
    # 遍历目录中的所有文件
    for filename in os.listdir(target_dir):
        # 只处理.nii.gz文件
        if filename.endswith('.nii.gz'):
            # 获取文件完整路径
            old_path = os.path.join(target_dir, filename)
            
            # 提取文件名中的所有数字
            digits = re.findall(r'\d', filename)
            
            # 检查是否有至少四位数字
            if len(digits) >= 4:
                # 取前四位数字作为新文件名
                new_name = ''.join(digits[:4]) + '.nii.gz'
                new_path = os.path.join(target_dir, new_name)
                
                # 避免重名覆盖
                if os.path.exists(new_path):
                    print(f"警告：文件 {new_path} 已存在，跳过重命名 {filename}")
                    continue
                
                # 执行重命名
                try:
                    os.rename(old_path, new_path)
                    print(f"成功：{filename} -> {new_name}")
                except Exception as e:
                    print(f"错误：重命名 {filename} 失败 - {str(e)}")
            else:
                print(f"警告：{filename} 中的数字不足四位，跳过")

# 主程序
if __name__ == "__main__":
    # 请修改为你的目标目录路径
    # Windows示例: r"C:\Users\YourName\data"
    # Linux/Mac示例: "/home/yourname/data"
    target_directory = r"C:\Projects\CenterSAM-PET-E\data\raw\images"
    
    # 调用函数执行重命名
    rename_nii_gz_files(target_directory)