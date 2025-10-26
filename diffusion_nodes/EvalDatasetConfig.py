import json
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

class EvalDatasetConfig:
    """
    评估数据集配置节点
    用于配置评估数据集的各种参数
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 输入路径
                "input_path": ("input_path", {
                    "tooltip": "评估数据集输入路径，必选，根据不同训练目的，选择不同节点"
                }),
                
                # 基础分辨率设置
                "resolutions": ("STRING", {
                    "default": "[512]",
                    "multiline": False,
                    "tooltip": "评估分辨率，可以是单个数值（正方形）或 [宽度, 高度] 对,例如: [1280, 720]"
                }),
                
                # 宽高比分桶设置
                "enable_ar_bucket": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否启用宽高比分桶设置"
                    }),
                    
                "min_ar": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "最小宽高比"
                }),
                "max_ar": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "最大宽高比"
                }),
                "num_ar_buckets": ("INT", {
                    "default": 7,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "宽高比分桶数量"
                }),
                
                # 数据集重复次数
                "num_repeats": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "评估数据集重复次数"
                }),
            },
            "optional": {
                # 帧分桶设置（视频训练） 
                "frame_buckets": ("frame_buckets",{
                "tooltip": "帧分桶设置，例如: [1, 33] 或 [1, 33, 65, 97]，专用与视频模型训练"                            
            }),
                "ar_buckets": ("ar_buckets",{
                    "tooltip": "宽高比分桶设置，例如:[[512, 512], [448, 576]]",
            }),            
            }
        }
    
    RETURN_TYPES = ("EVAL_DATASET_CONFIG",)
    RETURN_NAMES = ("eval_dataset_config",)
    FUNCTION = "generate_config"
    CATEGORY = "Diffusion-Pipe/dataset"
    
    def generate_config(self, input_path, resolutions, enable_ar_bucket, min_ar, max_ar, 
                       num_ar_buckets, num_repeats, frame_buckets=None, ar_buckets=None):
        """
        生成评估数据集配置文件内容
        """
        try:
            # 处理input_path参数
            dataset_path = None
            control_path = None
            is_edit_model = False
            
            if isinstance(input_path, dict):
                # 如果是字典（来自EditModelDatasetPathNode）
                dataset_path = input_path.get("path")
                control_path = input_path.get("control_path")
                is_edit_model = True
            elif isinstance(input_path, str):
                # 如果是字符串（来自GeneralDatasetPathNode）
                dataset_path = input_path
            
            # 验证模型类型兼容性
            if is_edit_model and frame_buckets is not None and frame_buckets.strip():
                raise ValueError(
                    "error，you can't use frame_buckets and edit_model at the same time"
                 
                )
            
            # 解析分辨率设置
            resolutions_list = self._parse_list_input(resolutions, "resolutions")
            
            # 解析帧分桶设置（如果提供）
            frame_buckets_list = None
            if frame_buckets is not None and frame_buckets.strip():
                frame_buckets_list = self._parse_list_input(frame_buckets, "frame_buckets")
            
            # 解析宽高比分桶设置（如果提供）
            ar_buckets_list = None
            if ar_buckets is not None and ar_buckets.strip():
                ar_buckets_list = self._parse_list_input(ar_buckets, "ar_buckets")
            
            # 构建配置内容
            config_lines = []
            
            # 分辨率配置
            if len(resolutions_list) == 1 and isinstance(resolutions_list[0], (int, float)):
                config_lines.append(f"resolutions = [{int(resolutions_list[0])}]")
            else:
                config_lines.append(f"resolutions = {resolutions_list}")
            
            # 宽高比分桶配置
            config_lines.append(f"enable_ar_bucket = {str(enable_ar_bucket).lower()}")
            
            if enable_ar_bucket:
                config_lines.extend([
                    f"min_ar = {min_ar}",
                    f"max_ar = {max_ar}",
                    f"num_ar_buckets = {num_ar_buckets}",
                ])
            
            # 宽高比分桶配置（如果提供）
            if ar_buckets_list is not None:
                config_lines.append(f"ar_buckets = {ar_buckets_list}")
            
            # 帧分桶配置（仅在提供时添加）
            if frame_buckets_list is not None:
                config_lines.append(f"frame_buckets = {frame_buckets_list}")
            
            if control_path:

                normalized_dataset_path = normalize_windows_path(dataset_path) if dataset_path else "C:\\path\\to\\target\\images"
                normalized_control_path = normalize_windows_path(control_path) if control_path else "C:\\path\\to\\control\\images"
                config_lines.extend([
                    "[[directory]]",
                    f"path = '{normalized_dataset_path}'",
                    f"control_path = '{normalized_control_path}'",
                    f"num_repeats = {num_repeats}",
                ])
            else:
                normalized_dataset_path = normalize_windows_path(dataset_path) if dataset_path else "C:\\path\\to\\your\\eval\\dataset"
                config_lines.extend([
                    "[[directory]]",
                    f"path = '{normalized_dataset_path}'",
                    f"num_repeats = {num_repeats}",
                ])
            
            config_content = "\n".join(config_lines)
            
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                output_path = os.path.join(current_dir, "..", "dataset", "evaldataset.toml")
                normalized_output_path = os.path.normpath(output_path)
                
                output_dir = os.path.dirname(normalized_output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                
                import datetime
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                current_cwd = os.getcwd()
                print("\n" + "="*80)
                print(f"[评估数据集配置] 配置文件已保存到: {normalized_output_path}")
                print(f"[评估数据集配置] 生成时间: {current_time}")
                print(f"[评估数据集配置] 当前工作目录: {current_cwd}")
                print("="*80 + "\n")
                
                with open(normalized_output_path, 'w', encoding='utf-8') as f:
                    f.write(config_content)
                
                display_path = normalized_output_path.replace('\\', '/')
                print("="*80)
                print(f"评估数据集配置已保存到: {display_path}")
                print("="*80)
                
                # 返回纯配置内容（不带路径注释）
                return (config_content,)
            except Exception as e:
                print(f"保存评估配置文件失败: {str(e)}")
                return (config_content,)
            
        except Exception as e:
            error_msg = f"生成评估数据集配置失败: {str(e)}"
            return (error_msg,)
    
    def _parse_list_input(self, input_str, param_name):
        """
        解析列表输入字符串
        """
        try:
            # 移除空白字符
            input_str = input_str.strip()
            
            if not input_str:
                raise ValueError(f"{param_name} 不能为空")
            
            # 尝试解析为 JSON
            if input_str.startswith('[') and input_str.endswith(']'):
                parsed = json.loads(input_str)
                if isinstance(parsed, list):
                    return parsed
                else:
                    raise ValueError("必须是列表格式")
            else:
                # 尝试解析为单个数值
                try:
                    value = float(input_str)
                    return [int(value) if value.is_integer() else value]
                except:
                    raise ValueError("格式错误，应为数字或 [数字1, 数字2, ...] 格式")
                    
        except json.JSONDecodeError:
            raise ValueError(f"{param_name} JSON 格式错误")
        except Exception as e:
            raise ValueError(f"{param_name} 解析失败: {str(e)}") 
    


