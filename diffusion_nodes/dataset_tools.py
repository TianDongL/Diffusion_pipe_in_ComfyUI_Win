import os

def normalize_windows_path(path):
    """
    规范化Windows环境下的路径
    将WSL路径转换为Windows路径，或保持Windows路径不变
    """
    if not path:
        return path
        
    # 将路径转换为Windows格式
    path = str(path).replace('/', '\\')
    
    # 处理WSL格式的路径转换为Windows路径
    if path.startswith('\\mnt\\'):
        # /mnt/c/path -> C:\path
        parts = path.split('\\')
        if len(parts) >= 3:
            drive_letter = parts[2].upper()
            rest_path = '\\'.join(parts[3:])
            return f"{drive_letter}:\\{rest_path}"
    
    # 如果路径以\开头但不是UNC路径，可能是WSL路径
    if path.startswith('\\') and not path.startswith('\\\\'):
        # 假设是根目录下的路径，映射到当前工作目录
        current_dir = os.getcwd()
        return os.path.join(current_dir, path.lstrip('\\'))
    
    # 规范化路径
    return os.path.normpath(path)

class GeneralDatasetPathNode:
    """
    连接通用数据集配置节点的额外参数
    
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "数据集文件夹路径"
                }),
            }
        }
    
    RETURN_TYPES = ("input_path",)
    RETURN_NAMES = ("input_path",)
    FUNCTION = "get_dataset_path"
    CATEGORY = "Diffusion-Pipe/dataset"
    
    def get_dataset_path(self, dataset_path):
        """
        返回数据集路径
        """
        # 规范化路径用于验证
        normalized_path = normalize_windows_path(dataset_path)
        
        # 验证路径是否存在
        if not os.path.exists(normalized_path):
            print(f"警告: 路径不存在: {normalized_path}")
        
        return (dataset_path,)

class ArBucketsNode:
    """
    宽高比分桶配置节点
    用于配置训练数据集的宽高比分桶设置
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ar_buckets": ("STRING", {
                    "default": "[[512, 512], [448, 576]]",
                    "multiline": False,
                    "tooltip": "宽高比分桶配置，可选"
                }),
            }
        }
    
    RETURN_TYPES = ("ar_buckets",)
    RETURN_NAMES = ("ar_buckets",)
    FUNCTION = "process_ar_buckets"
    CATEGORY = "Diffusion-Pipe/dataset"
    
    def process_ar_buckets(self, ar_buckets):
        """
        直接返回用户输入的宽高比分桶配置字符串，保持原格式
        """
        try:
            # 清理输入字符串（只去除首尾空白)
            ar_buckets_str = ar_buckets.strip()
            
            if not ar_buckets_str:
                print("警告: 宽高比分桶配置为空，使用默认值")
                return ("[[512, 512], [448, 576]]",)
            
            # 直接返回用户输入的字符串，保持原格式
            print(f"宽高比分桶配置: {ar_buckets_str}")
            return (ar_buckets_str,)
            
        except Exception as e:
            print(f"处理宽高比分桶配置时出错: {str(e)}")
            print("使用默认宽高比分桶配置: [[512, 512], [448, 576]]")
            return ("[[512, 512], [448, 576]]",)

class EditModelDatasetPathNode:
    """
    编辑模型数据集路径节点
    配置target路径和control路径
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "生成图像路径 - 模型要学习生成的图像"
                }),
                "control_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "原图像路径 - 与目标图像对应的控制图像"
                }),
            }
        }
    
    RETURN_TYPES = ("input_path",)
    RETURN_NAMES = ("input_path",)
    FUNCTION = "get_edit_dataset_paths"
    CATEGORY = "Diffusion-Pipe/dataset"
    
    def get_edit_dataset_paths(self, target_path, control_path):
        """
        返回编辑模型数据集路径配置
        """
        # 规范化路径用于验证
        normalized_target_path = normalize_windows_path(target_path)
        normalized_control_path = normalize_windows_path(control_path)
        
        # 验证路径是否存在
        if not os.path.exists(normalized_target_path):
            print(f"警告: 目标路径不存在: {normalized_target_path}")
        
        if not os.path.exists(normalized_control_path):
            print(f"警告: 控制路径不存在: {normalized_control_path}")
        
        # 返回包含两个路径的字典
        dataset_config = {
            "path": target_path,
            "control_path": control_path
        }
        
        return (dataset_config,)


class FrameBucketsNode:
    """
    帧桶配置节点
    用于配置视频训练中的帧数分桶设置
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_buckets": ("STRING", {
                    "default": "[1, 33, 81, 97]",
                    "multiline": False,
                    "tooltip": "帧数分桶配置，专用于视频模型训练，格式：[1, 33, 81, 97]"
                }),
            }
        }
    
    RETURN_TYPES = ("frame_buckets",)
    RETURN_NAMES = ("frame_buckets",)
    FUNCTION = "process_frame_buckets"
    CATEGORY = "Diffusion-Pipe/dataset"
    
    def process_frame_buckets(self, frame_buckets):
        """
        直接返回用户输入的帧桶配置字符串，保持原格式
        """
        try:
            # 清理输入字符串（只去除首尾空白）
            frame_buckets_str = frame_buckets.strip()
            
       
            if not frame_buckets_str:
                print("警告: 帧桶配置为空，使用默认值")
                return ("[1, 33, 65, 97]",)
            
            # 直接返回用户输入的字符串，保持原格式
            print(f"帧桶配置: {frame_buckets_str}")
            return (frame_buckets_str,)
            
        except Exception as e:
            print(f"处理帧桶配置时出错: {str(e)}")
            print("使用默认帧桶配置: [1, 33, 65, 97]")
            return ("[1, 33, 65, 97]",)

 