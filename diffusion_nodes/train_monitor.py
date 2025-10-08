import os
import sys
import subprocess
import threading
import time
import queue
import psutil
from pathlib import Path

# 全局进程管理器
class TensorBoardProcessManager:
    _instance = None
    _processes = {}  # port -> process_info
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register_process(self, port, process, logdir, host):
        """注册TensorBoard进程"""
        self._processes[port] = {
            'process': process,
            'logdir': logdir,
            'host': host,
            'start_time': time.time()
        }
    
    def get_process(self, port):
        """获取指定端口的进程信息"""
        return self._processes.get(port)
    
    def remove_process(self, port):
        """移除进程记录"""
        if port in self._processes:
            del self._processes[port]
    
    def kill_process_on_port(self, port):
        """强制终止指定端口的TensorBoard进程"""
        try:
            # 1. 从注册表中查找
            if port in self._processes:
                process = self._processes[port]['process']
                if process and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                self.remove_process(port)
            
            # 2. 使用psutil查找并终止所有占用该端口的进程
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and ('tensorboard' in proc.info['name'].lower() or 'python' in proc.info['name'].lower()):
                        cmdline = proc.info['cmdline'] or []
                        cmdline_str = ' '.join(cmdline)
                        if f'--port={port}' in cmdline_str or f'--port {port}' in cmdline_str:
                            print(f"找到占用端口{port}的进程: PID {proc.info['pid']}")
                            proc.terminate()
                            try:
                                proc.wait(timeout=5)
                            except psutil.TimeoutExpired:
                                proc.kill()
                                print(f"强制终止进程 PID {proc.info['pid']}")
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            
            return True
        except Exception as e:
            print(f"终止端口{port}上的进程时出错: {e}")
            return False
    
    def is_port_in_use(self, port):
        """检查端口是否被占用"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and ('tensorboard' in proc.info['name'].lower() or 'python' in proc.info['name'].lower()):
                        cmdline = proc.info['cmdline'] or []
                        cmdline_str = ' '.join(cmdline)
                        if f'--port={port}' in cmdline_str or f'--port {port}' in cmdline_str:
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            return False
        except Exception:
            return False

class TensorBoardMonitor:
    def __init__(self):
        self.process_manager = TensorBoardProcessManager()
        self.tensorboard_process = None
        self.is_running = False
        self.log_queue = queue.Queue()
        # 监控信息
        self.current_logdir = None
        self.current_host = None
        self.current_port = None
        self.start_time = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_dir": ("STRING", {
                    "forceInput": True,
                    "tooltip": "训练输出目录（来自通用训练设置）"
                }),
                "port": ("INT", {
                    "default": 6006,
                    "min": 1024,
                    "max": 65535,
                    "tooltip": "TensorBoard服务端口"
                }),
                "host": ("STRING", {
                    "default": "localhost",
                    "tooltip": "TensorBoard服务主机地址"
                }),
                "is_new_training": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否开启新训练（开启时会延迟30秒等待训练文件生成）"
                })
            },
            "optional": {
                "action": (["start", "stop", "status", "kill_port"], {
                    "default": "start",
                    "tooltip": "操作类型：启动/停止/查看状态/强制清理端口"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("url", "status")
    FUNCTION = "execute"
    CATEGORY = "Diffusion-Pipe/Monitor"
    
    def execute(self, output_dir, port=6006, host="localhost", is_new_training=True, action="start"):
        """执行TensorBoard监控操作"""
        if action == "start":
            url_result = self.start_tensorboard(output_dir, port, host, is_new_training)
            status = self.get_current_status()
            url = url_result[0] if url_result and len(url_result) > 0 else ""
            return (url, status)
        elif action == "stop":
            result_tuple = self.stop_tensorboard()
            status = self.get_current_status()
            result = result_tuple[0] if result_tuple and len(result_tuple) > 0 else ""
            return (result, status)
        elif action == "status":
            status = self.get_current_status()
            url = f"http://{host}:{port}" if self.is_running else ""
            return (url, status)
        elif action == "kill_port":
            if self.process_manager.kill_process_on_port(port):
                result = f"成功清理端口{port}上的所有进程 (Successfully cleaned all processes on port {port})"
                print(result)
            else:
                result = f"清理端口{port}失败或该端口无进程 (Failed to clean port {port} or no processes on this port)"
                print(result)
            status = self.get_current_status()
            return ("", result)
        else:
            return ("", "未知操作 (Unknown operation)")
    
    def normalize_path(self, path):
        """规范化路径"""
        if path is None:
            return None
        if not path or path.strip() == "":
            return path
            
        try:
            path_obj = Path(path)
            
            if not path_obj.is_absolute():
                current_dir = Path(__file__).parent.parent
                path_obj = current_dir / path_obj
            
            path_obj = path_obj.resolve()
            return str(path_obj)
            
        except Exception as e:
            print(f"路径规范化失败: {e}")
            return str(Path(path).resolve()) if path else path
    
    def find_latest_training_dir(self, base_dir):
        """在基础目录中寻找最新的训练子目录"""
        try:
            print(f"正在扫描目录: {base_dir}")
            training_dirs = []
            
            if not os.path.exists(base_dir):
                print(f"基础目录不存在: {base_dir}")
                return None
                
            items = os.listdir(base_dir)
            print(f"找到 {len(items)} 个项目")
            
            for item in items:
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path):
                    print(f"检查子目录: {item}")
                    if self.has_tensorboard_files(item_path):
                        print(f"发现训练日志目录: {item}")
                        training_dirs.append((item_path, os.path.getmtime(item_path)))
            
            if training_dirs:
                training_dirs.sort(key=lambda x: x[1], reverse=True)
                latest_dir = training_dirs[0][0]
                print(f"选择最新的训练目录: {latest_dir}")
                return latest_dir
            
            print("未找到包含TensorBoard日志的子目录")
            return None
            
        except Exception as e:
            print(f"查找训练目录时出错: {e}")
            return None
    
    def has_tensorboard_files(self, directory):
        """检查目录是否包含TensorBoard日志文件"""
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.startswith('events.out.tfevents'):
                        return True
            return False
        except:
            return False
    

    
    def start_tensorboard(self, output_dir, port, host, is_new_training=True):
        """启动TensorBoard服务"""
        try:
            if output_dir is None:
                print("错误: output_dir 参数为 None")
                return ("",)
            
            # 清理端口
            if self.process_manager.is_port_in_use(port):
                print(f"检测到端口{port}被占用，正在清理...")
                if self.process_manager.kill_process_on_port(port):
                    print(f"成功清理端口{port}")
                    time.sleep(3)  # 等待端口释放
                else:
                    print(f"清理端口{port}失败")
                    return ("",)
            
            # 检查是否已经运行
            if self.is_running and self.tensorboard_process and self.tensorboard_process.poll() is None:
                url = f"http://{host}:{port}"
                return (url,)
            
            output_dir = self.normalize_path(output_dir)
            
            if not os.path.exists(output_dir):
                print(f"输出目录不存在: {output_dir}")
                return ("",)
            
            # 新训练模式等待
            if is_new_training:
                print("开始新训练模式：等待训练文件生成（30秒延迟）...")
                time.sleep(30)
                print("等待完成，开始查找最新训练目录")
            
            # 查找训练目录
            logdir = self.find_latest_training_dir(output_dir)
            
            if logdir and os.path.exists(logdir):
                print(f"使用训练日志目录: {logdir}")
                final_logdir = logdir
            else:
                print(f"在目录 {output_dir} 中未找到训练日志，使用基础目录")
                final_logdir = output_dir
            
            # 构建TensorBoard启动命令
            cmd = [
                sys.executable, "-m", "tensorboard.main",
                "--logdir", final_logdir,
                "--port", str(port),
                "--host", host,
                "--reload_interval", "30"
            ]
            
            print(f"启动TensorBoard命令: {' '.join(cmd)}")
            print(f"日志目录: {final_logdir}")
            print(f"访问地址: http://{host}:{port}")
            
            # 启动进程
            self.tensorboard_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            # 注册进程
            self.process_manager.register_process(port, self.tensorboard_process, final_logdir, host)
            
            # 启动日志读取线程
            log_thread = threading.Thread(
                target=self.log_reader,
                args=(self.tensorboard_process, self.log_queue),
                daemon=True
            )
            log_thread.start()
            
            self.is_running = True
            self.current_logdir = final_logdir
            self.current_host = host
            self.current_port = port
            self.start_time = time.time()
            
            # 等待启动
            time.sleep(5)
            
            # 检查启动状态
            if self.tensorboard_process.poll() is not None:
                return_code = self.tensorboard_process.returncode
                print(f"TensorBoard启动失败，返回码: {return_code}")
                
                try:
                    stderr_output = self.tensorboard_process.stderr.read()
                    if stderr_output:
                        print(f"错误信息: {stderr_output}")
                except:
                    pass
                
                self.is_running = False
                self.process_manager.remove_process(port)
                return ("",)
            
            url = f"http://{host}:{port}"
            print(f"TensorBoard成功启动! PID: {self.tensorboard_process.pid}")
            print(f"访问地址: {url}")
            
            return (url,)
            
        except FileNotFoundError:
            print("Python解释器路径错误或TensorBoard未安装")
            return ("",)
        except Exception as e:
            self.is_running = False
            print(f"启动TensorBoard时发生错误: {str(e)}")
            return ("",)
    
    def stop_tensorboard(self):
        """停止TensorBoard服务"""
        result = "TensorBoard未运行 (TensorBoard not running)"
        
        if self.current_port:
            if self.process_manager.kill_process_on_port(self.current_port):
                result = "TensorBoard已停止 (TensorBoard stopped)"
                print("TensorBoard已停止 (TensorBoard stopped)")
            else:
                result = "停止失败 (Stop failed)"
                print("停止TensorBoard时出现错误 (Error occurred while stopping TensorBoard)")
        
        if self.tensorboard_process:
            try:
                self.tensorboard_process.terminate()
                
                try:
                    self.tensorboard_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.tensorboard_process.kill()
                    self.tensorboard_process.wait()
                    print("TensorBoard进程被强制终止 (TensorBoard process was forcibly terminated)")
                
                if result == "TensorBoard未运行 (TensorBoard not running)":
                    result = "TensorBoard已停止 (TensorBoard stopped)"
                    print("TensorBoard已停止 (TensorBoard stopped)")
                
            except Exception as e:
                print(f"停止TensorBoard时出现错误 (Error stopping TensorBoard): {str(e)}")
                if result == "TensorBoard未运行 (TensorBoard not running)":
                    result = f"停止失败 (Stop failed): {str(e)}"
        

        
        self.tensorboard_process = None
        self.is_running = False
        self.current_logdir = None
        self.current_host = None
        self.current_port = None
        self.start_time = None
        
        return (result,)
    
    def get_current_status(self):
        """获取TensorBoard当前状态"""
        if not self.tensorboard_process:
            return "🔴 未启动 (Not Started)"
        
        if self.tensorboard_process.poll() is None:
            pid = self.tensorboard_process.pid
            
            if self.start_time:
                run_time = time.time() - self.start_time
                hours = int(run_time // 3600)
                minutes = int((run_time % 3600) // 60)
                seconds = int(run_time % 60)
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                time_str = "未知 (Unknown)"
            
            status_lines = [
                f"🟢 运行中 (Running) (PID: {pid})",
                f"⏱️  运行时间 (Runtime): {time_str}"
            ]
            
            if self.current_logdir:
                try:
                    rel_path = os.path.relpath(self.current_logdir)
                    if len(rel_path) > 80:
                        rel_path = "..." + rel_path[-77:]
                    status_lines.append(f"📁 监控目录 (Monitor Dir): {rel_path}")
                except:
                    status_lines.append(f"📁 监控目录 (Monitor Dir): {self.current_logdir}")
            
            if self.current_host and self.current_port:
                status_lines.append(f"🌐 访问地址 (Access URL): http://{self.current_host}:{self.current_port}")
            
            if self.current_logdir and os.path.exists(self.current_logdir):
                try:
                    event_files = []
                    for root, dirs, files in os.walk(self.current_logdir):
                        for file in files:
                            if file.startswith('events.out.tfevents'):
                                rel_file_path = os.path.relpath(os.path.join(root, file), self.current_logdir)
                                event_files.append(rel_file_path)
                    
                    if event_files:
                        status_lines.append(f"📊 发现 {len(event_files)} 个事件文件 (Found {len(event_files)} event files):")
                        for i, file in enumerate(event_files[:3]):
                            if len(file) > 60:
                                file = file[:57] + "..."
                            status_lines.append(f"   • {file}")
                        if len(event_files) > 3:
                            status_lines.append(f"   • ... 还有 {len(event_files) - 3} 个文件 (... {len(event_files) - 3} more files)")
                    else:
                        status_lines.append("⚠️  未找到TensorBoard事件文件 (No TensorBoard event files found)")
                        
                except Exception as e:
                    status_lines.append(f"⚠️  读取目录时出错 (Error reading directory): {str(e)}")
            
            return "\n".join(status_lines)
        else:
            return_code = self.tensorboard_process.returncode
            self.is_running = False
            
            if return_code == 0:
                return "🔴 已停止 (Stopped) (正常退出 Normal Exit)"
            else:
                return f"🔴 已停止 (Stopped) (异常退出 Abnormal Exit，返回码 Return Code: {return_code})"
    
    def log_reader(self, process, log_queue):
        """读取TensorBoard进程的输出日志"""
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    line = line.strip()
                    log_queue.put(line)
                    print(f"[TensorBoard] {line}")
            
            for line in iter(process.stderr.readline, ''):
                if line:
                    line = line.strip()
                    log_queue.put(f"ERROR: {line}")
                    print(f"[TensorBoard Error] {line}")
                    
        except Exception as e:
            log_queue.put(f"Log reader error: {str(e)}") 