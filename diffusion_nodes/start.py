import os
import sys
import subprocess
import threading
from datetime import datetime
import toml
import tempfile
import json
import time
import signal
import queue
from pathlib import Path
import platform

try:
    from ..utils.config_parser import ConfigParser
except ImportError:
    import os
    import sys
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    try:
        from utils.config_parser import ConfigParser
    except ImportError:
        class ConfigParser:
            @staticmethod
            def merge_configs(dataset_config, train_config):
                return {**dataset_config, **train_config}

class Train:
    def __init__(self):
        self.training_process = None
        self.log_queue = queue.Queue()
        self.is_training = False
        # 注册全局实例
        try:
            from .train_monitor import set_global_train_instance
            set_global_train_instance(self)
        except ImportError:
            pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_config": ("DATASET_CONFIG", {
                    "tooltip": "数据集配置（来自GeneralDatasetConfig节点）"
                }),
                "train_config": ("TRAIN_CONFIG", {
                    "tooltip": "训练配置（来自GeneralConfig节点）"
                }),
                "config_path": ("config_path", {
                    "tooltip": "配置文件路径（来自GeneralConfig节点）"
                }),
            },
            "optional": {
                "resume_from_checkpoint": ("STRING", {
                    "default": "",
                    "tooltip": "从指定检查点继续训练，例如：'20250212_07-06-40' 或留空表示不从检查点恢复"
                }),
                "reset_dataloader": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "从检查点恢复时：勾选重置数据加载器（仅加载优化器状态，数据集从头开始）"
                }),
                "regenerate_cache": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "强制重新生成缓存（数据集更改后使用）"
                }),
                "trust_cache": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "信任现有缓存（跳过验证，加速大数据集加载）"
                }),
                "cache_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "仅缓存模型输入然后退出（不进行训练），用于预处理数据集"
                }),
                "i_know_what_i_am_doing": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "跳过某些检查和覆盖（高级用户专用，可能导致训练失败）"
                }),
                "dump_dataset": ("STRING", {
                    "default": "",
                    "tooltip": "将数据集导出到指定路径（调试用，导出后立即退出）"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "log_output")
    FUNCTION = "execute"
    CATEGORY = "Diffusion-Pipe/Train"
    
    def execute(self, dataset_config, train_config, config_path, resume_from_checkpoint="", reset_dataloader=False, regenerate_cache=False, trust_cache=False, cache_only=False, i_know_what_i_am_doing=False, dump_dataset=""):
        """ComfyUI节点的执行入口"""
        return self.start_training(dataset_config, train_config, config_path, resume_from_checkpoint, reset_dataloader, regenerate_cache, trust_cache, cache_only, i_know_what_i_am_doing, dump_dataset)
    
    def normalize_windows_path(self, path):
        """规范化Windows环境下的路径"""
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
        
        # 如果路径以/开头，可能是WSL路径，尝试转换
        if path.startswith('\\') and not path.startswith('\\\\'):
            # 假设是根目录下的路径，可能需要映射到当前工作目录
            current_dir = os.getcwd()
            return os.path.join(current_dir, path.lstrip('\\'))
        
                # 规范化路径
        return os.path.normpath(path)

    def log_reader(self, stream, log_queue, prefix="", stream_name="stream"):
        try:
            buffer = ""
            last_was_progress = False
            
            while True:
                chunk = stream.read(256)
                if not chunk:
                    break
                    
                text = chunk.decode('utf-8', errors='ignore')
                
                for char in text:
                    if char == '\n':
                        if buffer.strip():
                            if last_was_progress:
                                print()
                            line = f"{prefix}{buffer}" if prefix else buffer
                            print(line)
                            log_queue.put(line)
                            last_was_progress = False
                        buffer = ""
                    elif char == '\r':
                        if buffer.strip():
                            is_progress = '%|' in buffer or '|/' in buffer or ('[' in buffer and ']' in buffer)
                            if is_progress:
                                print(f"\r{buffer}", end='', flush=True)
                                last_was_progress = True
                            else:
                                # Regular line: print with newline
                                if last_was_progress:
                                    print()
                                line = f"{prefix}{buffer}" if prefix else buffer
                                print(line)
                                log_queue.put(line)
                                last_was_progress = False
                        buffer = ""
                    else:
                        buffer += char
                    
            if buffer.strip():
                if last_was_progress:
                    print()
                line = f"{prefix}{buffer}" if prefix else buffer
                print(line)
                log_queue.put(line)
        except Exception as e:
            error_msg = f"ERROR reading {stream_name}: {str(e)}"
            print(error_msg)
            log_queue.put(error_msg)

    def start_training(self, dataset_config, train_config, config_path, resume_from_checkpoint="", reset_dataloader=False, regenerate_cache=False, trust_cache=False, cache_only=False, i_know_what_i_am_doing=False, dump_dataset=""):
        """启动训练进程"""
        try:
            # 检查是否已有训练进程在运行
            if self.training_process and self.training_process.poll() is None:
                message = "❗训练已经在进行中，请等待当前训练完成或手动停止后再启动新的训练"
                print(message)
                return "WARNING", message
            
            # 验证输入参数
            if not dataset_config:
                return "ERROR", "未提供数据集配置 (dataset_config)"
            
            if not train_config:
                return "ERROR", "未提供训练配置 (train_config)"
            
            # 处理配置参数
            if isinstance(dataset_config, str):
                try:
                    import json
                    dataset_config = json.loads(dataset_config)
                except:
                    try:
                        import toml
                        dataset_config = toml.loads(dataset_config)
                    except:
                        # 如果都失败，创建基础配置
                        dataset_config = {}
            
            if not isinstance(dataset_config, dict):
                dataset_config = {}
            
            if isinstance(train_config, str):
                try:
                    import json
                    train_config = json.loads(train_config)
                except:
                    try:
                        import toml
                        train_config = toml.loads(train_config)
                    except:
                        train_config = {}
            
            if not isinstance(train_config, dict):
                train_config = {}
            
            if not config_path:
                return "ERROR", "未指定配置文件保存路径 (config_path)"
            
            config_path = self.normalize_windows_path(config_path)
            
            if not os.path.exists(config_path):
                return "ERROR", f"配置文件不存在: {config_path}"
            
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            train_script = os.path.join(current_dir, "train.py")
            
            if not os.path.exists(train_script):
                return "ERROR", f"找不到训练脚本: {train_script}"
            

            # 智能查找Python解释器
            def find_python_interpreter():
                """智能查找可用的Python解释器"""
                root_dir = os.path.normpath(os.path.join(current_dir, "..", "..", ".."))
                
                # 方法1: 检测当前ComfyUI是否在Conda环境中运行
                conda_prefix = os.environ.get('CONDA_PREFIX')
                if conda_prefix:
                    conda_python = os.path.join(conda_prefix, "python.exe")
                    if os.path.exists(conda_python):
                        conda_env_name = os.environ.get('CONDA_DEFAULT_ENV', os.path.basename(conda_prefix))
                        print(f"✓ 检测到Conda环境: {conda_env_name}")
                        print(f"  Python路径: {conda_python}")
                        return conda_python
                
                # 方法2: 检测是否在虚拟环境中运行（venv/virtualenv）
                virtual_env = os.environ.get('VIRTUAL_ENV')
                if virtual_env:
                    venv_python = os.path.join(virtual_env, "Scripts", "python.exe")
                    if os.path.exists(venv_python):
                        print(f"✓ 检测到虚拟环境: {virtual_env}")
                        print(f"  Python路径: {venv_python}")
                        return venv_python
                
                # 方法3: 动态搜索根目录下所有包含python.exe的文件夹
                found_pythons = []
                
                try:
                    # 只搜索一级子目录，避免搜索过深
                    for item in os.listdir(root_dir):
                        item_path = os.path.join(root_dir, item)
                        if os.path.isdir(item_path):
                            # 检查常见的Python路径
                            possible_paths = [
                                os.path.join(item_path, "python.exe"),           # 根目录
                                os.path.join(item_path, "Scripts", "python.exe"), # venv结构
                                os.path.join(item_path, "bin", "python.exe"),    # Linux-like结构
                            ]
                            
                            for python_path in possible_paths:
                                if os.path.exists(python_path):
                                    # 验证是否真的是Python解释器
                                    try:
                                        import subprocess
                                        result = subprocess.run(
                                            [python_path, "--version"],
                                            capture_output=True,
                                            text=True,
                                            timeout=3
                                        )
                                        if result.returncode == 0:
                                            version = result.stdout.strip() or result.stderr.strip()
                                            found_pythons.append({
                                                'path': python_path,
                                                'env_name': item,
                                                'version': version
                                            })
                                            break  # 找到一个就跳出内层循环
                                    except:
                                        pass
                except Exception as e:
                    print(f"⚠ 搜索环境时出错: {e}")
                
                if found_pythons:
                    print(f"✓ 找到 {len(found_pythons)} 个Python环境:")
                    for idx, py in enumerate(found_pythons, 1):
                        print(f"  {idx}. {py['env_name']} - {py['version']}")
                    
                    for py in found_pythons:
                        if py['env_name'] == 'python_embeded_DP':
                            print(f" 选择专用训练环境: {py['env_name']}")
                            return py['path']
                    
                    priority_keywords = ['python_embeded', 'python', 'venv', 'env']
                    for keyword in priority_keywords:
                        for py in found_pythons:
                            if keyword in py['env_name'].lower():
                                print(f"→ 选择环境: {py['env_name']}")
                                return py['path']
                    
                    print(f"→ 选择环境: {found_pythons[0]['env_name']}")
                    return found_pythons[0]['path']
                
                print(f"⚠ 未找到专用环境，使用当前Python解释器")
                print(f"  路径: {sys.executable}")
                print(f"  版本: {sys.version.split()[0]}")
                print(f"  注意: 请确保当前环境已安装所有训练依赖包")
                return sys.executable
            
            python_exe = find_python_interpreter()
            
            if not python_exe or not os.path.exists(python_exe):
                error_msg = f"错误: 无法找到可用的Python解释器"
                print(error_msg)
                return "ERROR", error_msg
            
            cmd = [
                python_exe,
                train_script,
                "--config", config_path,
                "--deepspeed"
            ]
            
            if regenerate_cache or train_config.get('regenerate_cache', False):
                cmd.append("--regenerate_cache")
            
            if trust_cache or train_config.get('trust_cache', False):
                cmd.append("--trust_cache")
            
            # 添加从检查点恢复训练的参数
            if resume_from_checkpoint and resume_from_checkpoint.strip():
                cmd.extend(["--resume_from_checkpoint", resume_from_checkpoint.strip()])
            
            # 添加 reset_dataloader 参数
            if reset_dataloader:
                cmd.append("--reset_dataloader")
            
            # 添加 cache_only 参数
            if cache_only:
                cmd.append("--cache_only")
            
            # 添加 i_know_what_i_am_doing 参数
            if i_know_what_i_am_doing:
                cmd.append("--i_know_what_i_am_doing")
            
            # 添加 dump_dataset 参数（优先使用节点参数）
            if dump_dataset and dump_dataset.strip():
                cmd.extend(["--dump_dataset", dump_dataset.strip()])
            
            # 处理高级配置中的命令行参数（向后兼容，但节点参数优先）
            train_cmd_args = train_config.get('_train_cmd_args', {})
            if train_cmd_args:
                # resume_from_checkpoint 参数（仅在节点未提供时使用）
                if 'resume_from_checkpoint' in train_cmd_args and not resume_from_checkpoint:
                    resume_value = train_cmd_args['resume_from_checkpoint']
                    if isinstance(resume_value, bool) and resume_value:
                        cmd.append("--resume_from_checkpoint")
                    elif isinstance(resume_value, str):
                        cmd.extend(["--resume_from_checkpoint", resume_value])
                
                bool_args = ['reset_dataloader', 'cache_only', 'i_know_what_i_am_doing']
                for arg in bool_args:
                    if train_cmd_args.get(arg, False):
                        # 避免重复添加（节点参数已处理的情况）
                        if f"--{arg}" not in cmd:
                            cmd.append(f"--{arg}")
                
                # master_port 参数（仅从 _train_cmd_args 配置）
                if 'master_port' in train_cmd_args:
                    cmd.extend(["--master_port", str(train_cmd_args['master_port'])])
                
                # dump_dataset 参数（仅在节点未提供时使用）
                if 'dump_dataset' in train_cmd_args and not dump_dataset:
                    cmd.extend(["--dump_dataset", str(train_cmd_args['dump_dataset'])])
            
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            # Print command info (minimal)
            print(f"CMD: {' '.join(cmd)}")
            
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                bufsize=0,  # 无缓冲模式
                universal_newlines=False,
                shell=False,  
                cwd=current_dir  
            )
            
            stdout_thread = threading.Thread(
                target=self.log_reader,
                args=(self.training_process.stdout, self.log_queue, "", "stdout"),
                daemon=True
            )
            stderr_thread = threading.Thread(
                target=self.log_reader,
                args=(self.training_process.stderr, self.log_queue, "Training ", "stderr"),
                daemon=True
            )
            stdout_thread.start()
            stderr_thread.start()
            
            self.is_training = True
            
            time.sleep(2)
            
            if self.training_process.poll() is not None:
                # Process failed to start
                return_code = self.training_process.returncode
                error_msg = f"Process failed. Exit code: {return_code}"
                
                try:
                    stderr_output = self.training_process.stderr.read().decode('utf-8', errors='ignore')
                    if stderr_output:
                        error_msg += f"\n{stderr_output}"
                except:
                    pass
                
                self.is_training = False
                return "ERROR", error_msg
            
            # Collect initial logs
            initial_logs = []
            log_timeout = time.time() + 3
            
            while time.time() < log_timeout:
                try:
                    log_line = self.log_queue.get(timeout=0.1)
                    initial_logs.append(log_line)
                except queue.Empty:
                    continue
            
            # Drain any remaining logs
            while True:
                try:
                    log_line = self.log_queue.get_nowait()
                    initial_logs.append(log_line)
                except queue.Empty:
                    break
            
            log_output = "\n".join(initial_logs) if initial_logs else "Training started, initializing..."
            
            return "TRAINING_STARTED", f"PID: {self.training_process.pid}\nConfig: {config_path}\n\n{log_output}"
            
        except Exception as e:
            self.is_training = False
            error_msg = f"Error starting training: {str(e)}"
            print(error_msg)
            return "ERROR", error_msg
    
    def stop_training(self):
        """Stop training process"""
        if self.training_process and self.training_process.poll() is None:
            try:
                self.training_process.terminate()
                
                try:
                    self.training_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.training_process.kill()
                    self.training_process.wait()
                
                self.is_training = False
                return "STOPPED", "Training stopped"
                
            except Exception as e:
                return "ERROR", f"Error stopping training: {str(e)}"
        else:
            return "NOT_RUNNING", "No training process running"
    
    def get_training_status(self):
        """Get training status"""
        if not self.training_process:
            return "NOT_STARTED", "Training not started"
        
        if self.training_process.poll() is None:
            # Process still running - collect ALL logs
            logs = []
            try:
                while True:
                    log_line = self.log_queue.get_nowait()
                    logs.append(log_line)
            except queue.Empty:
                pass
            
            # Return ALL logs, not truncated
            log_output = "\n".join(logs) if logs else "Training in progress..."
            return "RUNNING", f"PID: {self.training_process.pid}\n\n{log_output}"
        else:
            # Process finished - collect all remaining logs
            logs = []
            try:
                while True:
                    log_line = self.log_queue.get_nowait()
                    logs.append(log_line)
            except queue.Empty:
                pass
            
            return_code = self.training_process.returncode
            self.is_training = False
            
            log_output = "\n".join(logs) if logs else ""
            
            if return_code == 0:
                return "COMPLETED", f"Exit code: {return_code}\n\n{log_output}"
            else:
                return "FAILED", f"Exit code: {return_code}\n\n{log_output}"
    
