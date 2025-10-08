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
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "log_output")
    FUNCTION = "execute"
    CATEGORY = "Diffusion-Pipe/Train"
    
    def execute(self, dataset_config, train_config, config_path, resume_from_checkpoint=""):
        """ComfyUI节点的执行入口"""
        return self.start_training(dataset_config, train_config, config_path, resume_from_checkpoint)
    
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

    def log_reader_stdout(self, process, log_queue):
        """Read stdout from training process and print to console"""
        try:
            for line in iter(process.stdout.readline, b''):
                if line:
                    decoded_line = line.decode('utf-8', errors='ignore').rstrip()
                    if decoded_line:
                        # 同时打印到控制台和放入队列
                        print(decoded_line)
                        log_queue.put(decoded_line)
        except Exception as e:
            error_msg = f"ERROR reading stdout: {str(e)}"
            print(error_msg)
            log_queue.put(error_msg)
    
    def log_reader_stderr(self, process, log_queue):
        """Read stderr from training process and print to console"""
        try:
            for line in iter(process.stderr.readline, b''):
                if line:
                    decoded_line = line.decode('utf-8', errors='ignore').rstrip()
                    if decoded_line:
                        # 同时打印到控制台（带标记）和放入队列
                        stderr_line = f"[STDERR] {decoded_line}"
                        print(stderr_line)
                        log_queue.put(stderr_line)
        except Exception as e:
            error_msg = f"ERROR reading stderr: {str(e)}"
            print(error_msg)
            log_queue.put(error_msg)

    def start_training(self, dataset_config, train_config, config_path, resume_from_checkpoint=""):
        """启动训练进程"""
        try:
            # 检查是否已有训练进程在运行
            if self.training_process and self.training_process.poll() is None:
                message = "训练已经在进行中，请等待当前训练完成或手动停止后再启动新的训练"
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
            
            # 确保dataset_config是字典类型
            if not isinstance(dataset_config, dict):
                dataset_config = {}
            
            # 处理训练配置
            if isinstance(train_config, str):
                try:
                    import json
                    train_config = json.loads(train_config)
                except:
                    try:
                        import toml
                        train_config = toml.loads(train_config)
                    except:
                        # 如果都失败，创建基础配置
                        train_config = {}
            
            # 确保train_config是字典类型
            if not isinstance(train_config, dict):
                train_config = {}
            
            # 检查配置文件路径
            if not config_path:
                return "ERROR", "未指定配置文件保存路径 (config_path)"
            
            # 规范化配置文件路径为Windows格式
            config_path = self.normalize_windows_path(config_path)
            
            # 检查配置文件是否存在
            if not os.path.exists(config_path):
                return "ERROR", f"配置文件不存在: {config_path}"
            
            # 获取训练脚本路径
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            train_script = os.path.join(current_dir, "train.py")
            
            if not os.path.exists(train_script):
                return "ERROR", f"找不到训练脚本: {train_script}"
            
            # 构建训练命令 - 针对Windows环境和ComfyUI便携包
            num_gpus = train_config.get('number_of_gpus', 1)
            
            # 使用ComfyUI便携包中的Python解释器
            # 基于当前文件位置计算到便携包Python的路径
            python_exe = os.path.join(current_dir, "..", "..", "..", "python_embeded_DP", "python.exe")
            python_exe = os.path.normpath(python_exe)
            
            # 检查Python解释器是否存在
            if not os.path.exists(python_exe):
                error_msg = f"错误: 未找到便携包Python解释器: {python_exe}"
                print(error_msg)
                return "ERROR", error_msg
            else:
                print(f"使用便携包Python解释器: {python_exe}")            
            
            cmd = [
                python_exe,
                train_script,
                "--config", config_path,
                "--deepspeed"
            ]
            
            if train_config.get('regenerate_cache', False):
                cmd.append("--regenerate_cache")
            
            if train_config.get('trust_cache', False):
                cmd.append("--trust_cache")
                
            if 'master_port' in train_config:
                cmd.extend(["--master_port", str(train_config['master_port'])])
            
            # 添加从检查点恢复训练的参数
            if resume_from_checkpoint and resume_from_checkpoint.strip():
                cmd.extend(["--resume_from_checkpoint", resume_from_checkpoint.strip()])
            
            # 处理高级配置中的命令行参数
            train_cmd_args = train_config.get('_train_cmd_args', {})
            if train_cmd_args:
                # resume_from_checkpoint 参数
                if 'resume_from_checkpoint' in train_cmd_args:
                    resume_value = train_cmd_args['resume_from_checkpoint']
                    if isinstance(resume_value, bool) and resume_value:
                        cmd.append("--resume_from_checkpoint")
                    elif isinstance(resume_value, str):
                        cmd.extend(["--resume_from_checkpoint", resume_value])
                
                # 布尔型参数
                bool_args = ['reset_dataloader', 'regenerate_cache', 'cache_only', 
                           'trust_cache', 'i_know_what_i_am_doing']
                for arg in bool_args:
                    if train_cmd_args.get(arg, False):
                        cmd.append(f"--{arg}")
                
                # master_port 参数（如果高级配置中有，优先使用）
                if 'master_port' in train_cmd_args:
                    # 移除之前添加的master_port参数
                    if "--master_port" in cmd:
                        idx = cmd.index("--master_port")
                        cmd.pop(idx)  # 移除 --master_port
                        cmd.pop(idx)  # 移除 端口值
                    cmd.extend(["--master_port", str(train_cmd_args['master_port'])])
                
                # dump_dataset 参数
                if 'dump_dataset' in train_cmd_args:
                    cmd.extend(["--dump_dataset", str(train_cmd_args['dump_dataset'])])
            
            # 设置环境变量
            env = os.environ.copy()
            
            # Windows环境下的CUDA设置
            if num_gpus > 1:
                env['WORLD_SIZE'] = str(num_gpus)
                env['RANK'] = '0'
                env['LOCAL_RANK'] = '0'
                env['MASTER_ADDR'] = 'localhost'
                env['MASTER_PORT'] = str(train_config.get('master_port', 29500))
            
            # Print command info (minimal)
            print(f"CMD: {' '.join(cmd)}")
            
            # 在Windows上使用shell=True可能更稳定
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                bufsize=1,
                universal_newlines=False,
                shell=False,  # Windows上通常不需要shell=True
                cwd=current_dir  # 设置工作目录
            )
            
            # Start separate threads for stdout and stderr
            stdout_thread = threading.Thread(
                target=self.log_reader_stdout,
                args=(self.training_process, self.log_queue),
                daemon=True
            )
            stderr_thread = threading.Thread(
                target=self.log_reader_stderr,
                args=(self.training_process, self.log_queue),
                daemon=True
            )
            stdout_thread.start()
            stderr_thread.start()
            
            self.is_training = True
            
            # 等待一小段时间检查进程是否正常启动
            time.sleep(2)
            
            if self.training_process.poll() is not None:
                # Process failed to start
                return_code = self.training_process.returncode
                error_msg = f"Process failed. Exit code: {return_code}"
                
                # Try to read error output
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
    
