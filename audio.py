import os
import sys
import re
import csv
import time
import json
import signal
import tempfile
import serial
from serial.tools import list_ports

# 💡 屏蔽 macOS 底层 Cocoa/AppKit 的调试警告输出
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"
os.environ["GLOG_minloglevel"] = "2"

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import pyqtgraph as pg

# 语音 & 阿里 DashScope 库
import pygame
import speech_recognition as sr
import dashscope
from dashscope import Generation
from dashscope.audio.asr import Recognition
from dashscope.audio.tts_v2 import SpeechSynthesizer

# 🔑 阿里百炼 API Key
DASHSCOPE_API_KEY = "sk-ws-H.ELPRYPY.3Da1.MEUCIH1HoF8_TVBnASi6tyiiIEB0c7eO-J3ae4pCrMWvTdeRAiEAiHckpw-7ALRQKiho_GCEY-eD14zz69oPnKfvekwWXUQ"
dashscope.api_key = DASHSCOPE_API_KEY


# ==========================================
# 1. 语音合成播报 (CosyVoice TTS)
# ==========================================
def speak_async(text):
    """【语音合成与播报】使用 CosyVoiceTTS 说话"""
    if not text:
        return

    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        synthesizer = SpeechSynthesizer(
            model="cosyvoice-v1",
            voice="longxiaochun"  # 自然女声
        )
        audio_bytes = synthesizer.call(text)

        if not audio_bytes:
            print("⚠️ 未获取到语音数据，请检查 API Key！")
            return

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio:
            tmp_audio.write(audio_bytes)
            tmp_audio_path = tmp_audio.name

        print(f"🔊 AI 正在回答: \"{text}\"")
        pygame.mixer.music.load(tmp_audio_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.music.unload()
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)

        time.sleep(0.5)

    except Exception as e:
        print(f"⚠️ 语音播报异常: {e}")


# ==========================================
# 2. 最新阿里千问语音控制线程 (ASR + LLM)
# ==========================================
class VoiceThread(QThread):
    voice_command_signal = pyqtSignal(str, str)  # 发送解析结果 (JSON, reply)
    chat_log_signal = pyqtSignal(str, str)       # 更新 GUI 聊天日志 (role, content)

    def __init__(self):
        super().__init__()
        self.running = False
        self.is_active = False  # 🚩 唤醒 Flag，默认处于待机状态
        self.recognizer = sr.Recognizer()

        self.recognizer.pause_threshold = 1.2
        self.recognizer.non_speaking_duration = 0.5
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True

    def get_working_microphone(self):
        """优先绑定 MacBook 本地内置麦克风"""
        try:
            mic_list = sr.Microphone.list_microphone_names()
            target_idx = None
            for index, name in enumerate(mic_list):
                if "MacBook" in name and "Microphone" in name:
                    target_idx = index

            if target_idx is not None:
                return sr.Microphone(device_index=target_idx)
            else:
                return sr.Microphone()
        except Exception as e:
            print(f"⚠️ 麦克风初始化异常: {e}")
            return sr.Microphone()

    def run(self):
        self.running = True
        self.chat_log_signal.emit("system", "🎙️ 语音控制线程已启动，等待唤醒...")

        mic = self.get_working_microphone()
        if not mic:
            self.chat_log_signal.emit("system", "❌ 未能绑定麦克风，线程结束")
            return

        with mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
            self.chat_log_signal.emit("system", "✅ 噪音校准完成！请输入唤醒词激活助手（如：'开始测试'）。")

        while self.running:
            try:
                with mic as source:
                    audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=8)

                # 1. Paraformer 语音识别
                user_text = self.transcribe_audio_free(audio)
                if not user_text:
                    continue

                self.chat_log_signal.emit("user", user_text)

                # 2. Flag 唤醒逻辑
                if not self.is_active:
                    wake_keywords = ["小k", "小k医生", "开始测试", "测试"]
                    if any(kw in user_text.lower() for kw in wake_keywords):
                        self.is_active = True
                        reply = "好的，请问刚度 K 和延迟 Delay 分别需要设置为什么值？"
                        self.chat_log_signal.emit("ai", reply)
                        speak_async(reply)
                    else:
                        print("🔒 [待机过滤] 未检测到唤醒词。")
                else:
                    exit_keywords = ["退出测试", "结束测试", "停止测试", "重新待机"]
                    if any(kw in user_text for kw in exit_keywords):
                        self.is_active = False
                        reply = "已退出测试模式，进入休眠待机状态。"
                        self.chat_log_signal.emit("ai", reply)
                        speak_async(reply)
                    else:
                        self.process_with_qwen_turbo(user_text)

            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"⚠️ 监听循环异常: {e}")
                time.sleep(0.5)

    def transcribe_audio_free(self, audio_data):
        tmp_file_path = None
        try:
            wav_bytes = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
            if len(wav_bytes) < 1000:
                return ""

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(wav_bytes)
                tmp_file_path = tmp_file.name

            recognition = Recognition(
                model="paraformer-realtime-v2",
                format="wav",
                sample_rate=16000,
                callback=None
            )
            result = recognition.call(tmp_file_path)

            if result.status_code == 200:
                sentence_list = result.get_sentence()
                if sentence_list:
                    return "".join([s.get("text", "") for s in sentence_list]).strip()
            return ""
        except Exception as e:
            print(f"⚠️ ASR 过程异常: {e}")
            return ""
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    def process_with_qwen_turbo(self, text):
        system_prompt = (
            "你是一个下肢外骨骼机器人的语音助手。"
            "请分析用户的输入，提取控制指令（例如设置刚度 K、延迟 Delay 等）。"
            "请严格以 JSON 格式输出，不要包含 Markdown 格式："
            '{"action": "K或Delay或none", "value": "数值或空", "reply": "给用户的简短口语回复"}'
            "例如："
            '用户：“把刚度设为 1.5” -> {"action": "K", "value": "1.5", "reply": "好的，已为您将刚度设为 1.5"}'
            '用户：“延迟设置为 200” -> {"action": "Delay", "value": "200", "reply": "好的，已将延迟设为 200"}'
        )

        try:
            response = Generation.call(
                model="qwen-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                result_format="message"
            )

            if response.status_code == 200:
                content = response.output.choices[0].message.content.strip()
                if content.startswith("```"):
                    lines = content.splitlines()
                    content = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])

                reply_text = "收到指令"
                try:
                    res_json = json.loads(content)
                    reply_text = res_json.get("reply", "好的，已收到指令。")
                except json.JSONDecodeError:
                    reply_text = content

                self.chat_log_signal.emit("ai", reply_text)
                self.voice_command_signal.emit(content, reply_text)
                speak_async(reply_text)
            else:
                print(f"⚠️ qwen-turbo 调用失败: {response.message}")

        except Exception as e:
            print(f"⚠️ 处理对话异常: {e}")

    def stop(self):
        self.running = False
        self.wait()


# ==========================================
# 3. IMU 数据解析与全局变量
# ==========================================
LINE_PATTERN = re.compile(
    r"LEFT:\s*"
    r"([-+]?\d*\.?\d+)\s*,\s*"
    r"([-+]?\d*\.?\d+)\s*,\s*"
    r"([-+]?\d*\.?\d+)"
    r"\s*\|\s*Gyro:\s*"
    r"([-+]?\d*\.?\d+)\s*,\s*"
    r"([-+]?\d*\.?\d+)\s*,\s*"
    r"([-+]?\d*\.?\d+)"
    r"\s*\|\|\s*RIGHT:\s*"
    r"([-+]?\d*\.?\d+)\s*,\s*"
    r"([-+]?\d*\.?\d+)\s*,\s*"
    r"([-+]?\d*\.?\d+)"
    r"\s*\|\s*Gyro:\s*"
    r"([-+]?\d*\.?\d+)\s*,\s*"
    r"([-+]?\d*\.?\d+)\s*,\s*"
    r"([-+]?\d*\.?\d+)"
)

def parse_imu_line(line):
    match = LINE_PATTERN.search(line)
    if match is None:
        return None
    return tuple(float(x) for x in match.groups())

def find_available_ports():  
    ports = list(list_ports.comports())
    return [p.device for p in ports]

win_size = 150 
t_buffer                = list([0.0] * win_size)
L_IMU_buffer            = t_buffer.copy()
R_IMU_buffer            = t_buffer.copy()
L_motor_torque_buffer   = t_buffer.copy()
R_motor_torque_buffer   = t_buffer.copy()
L_motor_torque_d_buffer = t_buffer.copy()
R_motor_torque_d_buffer = t_buffer.copy()
L_motor_angpos_buffer   = t_buffer.copy()  
R_motor_angpos_buffer   = t_buffer.copy()  
L_motor_angpos_a_buffer = t_buffer.copy()  
R_motor_angpos_a_buffer = t_buffer.copy()  

L_leg_IMU_angle = 0.0
R_leg_IMU_angle = 0.0
L_motor_torque = 0.0
R_motor_torque = 0.0
L_motor_torque_desired = 0.0
R_motor_torque_desired = 0.0
L_motor_angpos = 0.0
R_motor_angpos = 0.0
L_motor_angpos_a = 0.0  
R_motor_angpos_a = 0.0  

red  = pg.mkPen(color=(255, 0, 0), width=2)
blue = pg.mkPen(color=(0, 0, 255), width=2)

Connection_Flag   = False
LogginButton_Flag = False
t_0 = 0
ser = None
csv_file_name = ""
DataHeaders = ["time", "L_IMU", "R_IMU", "L_Torque_d", "L_Torque", "L_AngPos", "R_Torque_d", "R_Torque", "R_AngPos"]


# ==========================================
# 4. 后台数据读取线程
# ==========================================
class DataThread(QThread):
    imu_signal = pyqtSignal(tuple, float)

    def __init__(self):
        super().__init__()
        self.running = False

    def run(self):
        global ser
        self.running = True
        start_time = time.perf_counter()

        while self.running and ser and ser.is_open:
            try:
                if ser.in_waiting > 0:
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue

                    values = parse_imu_line(line)
                    if values is not None:
                        pc_time = time.perf_counter() - start_time
                        self.imu_signal.emit(values, pc_time)
            except Exception as e:
                print(f"Read Exception: {e}")
                break
            time.sleep(0.001)

    def stop(self):
        self.running = False
        self.wait()


# ==========================================
# 5. GUI 主界面
# ==========================================
class MainWindow(QWidget): 

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        global ConnectButton, LoggingButton, SerialComboBox
                
        self.setWindowTitle('Hip Exoskeleton Software v2.0 (With Voice Control)')
        self.setWindowIcon(QtGui.QIcon('BIRO_logo.png'))
        
        connected_ports = find_available_ports()

        # 主布局设计
        MainLayout    = QHBoxLayout() 
        LP_Layout     = QVBoxLayout() 
        RTDD_Layout   = QHBoxLayout() 
        Comm_Layout   = QHBoxLayout() 
        Cmd_Layout    = QHBoxLayout() 
        LMotorLayout  = QVBoxLayout()
        RMotorLayout  = QVBoxLayout()

        self.setLayout(MainLayout)

        MainLayout.addLayout(LP_Layout, stretch=4)
        MainLayout.addLayout(RTDD_Layout, stretch=6)

        LP_Layout.addLayout(Comm_Layout)
        LP_Layout.addLayout(Cmd_Layout)

        RTDD_Layout.addLayout(LMotorLayout, stretch=5)
        RTDD_Layout.addLayout(RMotorLayout, stretch=5)

        # Plot 控件
        LnR_IMU_plot        = pg.PlotWidget() 
        L_Motor_TnTd_plot   = pg.PlotWidget()
        L_Motor_AngPos_plot = pg.PlotWidget()
        R_Motor_TnTd_plot   = pg.PlotWidget()
        R_Motor_AngPos_plot = pg.PlotWidget()

        # ------------------ 左侧控制面板 ------------------
        ConnectButton  = QPushButton("Connect")
        SerialComboBox = QComboBox()
        LoggingButton  = QPushButton("Data Logging")
        self.VoiceButton = QPushButton("Voice Ctrl: OFF")

        ConnectButton.clicked.connect(self.Connect_Clicked)
        LoggingButton.clicked.connect(self.LogginButton_Clicked)
        self.VoiceButton.clicked.connect(self.VoiceButton_Clicked)

        Comm_Layout.addWidget(QLabel("ComPort:"))  
        Comm_Layout.addWidget(SerialComboBox)
        SerialComboBox.addItems(connected_ports) 
        Comm_Layout.addWidget(ConnectButton)
        Comm_Layout.addWidget(LoggingButton)
        Comm_Layout.addWidget(self.VoiceButton)

        # K & Delay 输入框
        Cmd_Layout.addWidget(QLabel("K:")) 
        self.K_text = QLineEdit()  
        KButton = QPushButton("Send")
        KButton.clicked.connect(lambda: self.CmdButton_Clicked("K", self.K_text.text()))

        Cmd_Layout.addWidget(self.K_text)
        Cmd_Layout.addWidget(KButton)   

        Cmd_Layout.addWidget(QLabel("Delay:"))   
        self.Delay_text  = QLineEdit()
        DelayButton = QPushButton("Send")
        DelayButton.clicked.connect(lambda: self.CmdButton_Clicked("Delay", self.Delay_text.text()))

        Cmd_Layout.addWidget(self.Delay_text)
        Cmd_Layout.addWidget(DelayButton)

        # ------------------ 🎙️ 实时语音对话日志面板 ------------------
        voice_box = QGroupBox("🎙️ 实时语音控制 & 对话记录")
        voice_box_layout = QVBoxLayout()
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas, Monaco, monospace;")
        
        voice_box_layout.addWidget(self.chat_display)
        voice_box.setLayout(voice_box_layout)

        # 图表与语音日志放入左侧
        LP_Layout.addWidget(LnR_IMU_plot, stretch=4)
        LP_Layout.addWidget(voice_box, stretch=4)

        # ------------------ 右侧数据曲线 ------------------
        LMotorLayout.addWidget(L_Motor_TnTd_plot)  
        LMotorLayout.addWidget(L_Motor_AngPos_plot)   

        RMotorLayout.addWidget(R_Motor_TnTd_plot)  
        RMotorLayout.addWidget(R_Motor_AngPos_plot)  

        # 样式配置
        label_style = {"font-size": "14px"}
        title_style = {"color": "black", "font-size": "16px"}

        LnR_IMU_plot.setTitle("Thighs Angular Position", **title_style)
        LnR_IMU_plot.setLabel('left', "Angle [deg]", **label_style)
        LnR_IMU_plot.setLabel('bottom', "Time [s]", **label_style)
        LnR_IMU_plot.addLegend()
        LnR_IMU_plot.setBackground('w')
        LnR_IMU_plot.showGrid(x=True, y=True)
        self.L_IMU_line = LnR_IMU_plot.plot(t_buffer, L_IMU_buffer, name="Left", pen=red)
        self.R_IMU_line = LnR_IMU_plot.plot(t_buffer, R_IMU_buffer, name="Right", pen=blue)

        L_Motor_TnTd_plot.setTitle("Actuator 1 Torque (Left)", **title_style) 
        L_Motor_TnTd_plot.setBackground('w') 
        L_Motor_TnTd_plot.showGrid(x=True, y=True)
        self.L_Motor_Taud_line = L_Motor_TnTd_plot.plot(t_buffer, L_motor_torque_d_buffer, name="Command", pen=blue)
        self.L_Motor_Tau_line  = L_Motor_TnTd_plot.plot(t_buffer, L_motor_torque_buffer, name="Actual", pen=red)

        L_Motor_AngPos_plot.setTitle("Actuator 1 Position (Left)", **title_style)
        L_Motor_AngPos_plot.setBackground('w')   
        L_Motor_AngPos_plot.showGrid(x=True, y=True)   
        self.L_motor_angpos_line = L_Motor_AngPos_plot.plot(t_buffer, L_motor_angpos_buffer, name="Ref", pen=red)  
        self.L_motor_angpos_a_line = L_Motor_AngPos_plot.plot(t_buffer, L_motor_angpos_a_buffer, name="Actual", pen=blue)     

        R_Motor_TnTd_plot.setTitle("Actuator 2 Torque (Right)", **title_style)
        R_Motor_TnTd_plot.setBackground('w')
        R_Motor_TnTd_plot.showGrid(x=True, y=True)
        self.R_Motor_Taud_line = R_Motor_TnTd_plot.plot(t_buffer, R_motor_torque_d_buffer, name="Command", pen=blue)
        self.R_Motor_Tau_line  = R_Motor_TnTd_plot.plot(t_buffer, R_motor_torque_buffer, name="Actual", pen=red)

        R_Motor_AngPos_plot.setTitle("Actuator 2 Position (Right)", **title_style)
        R_Motor_AngPos_plot.setBackground('w')
        R_Motor_AngPos_plot.showGrid(x=True, y=True)
        self.R_motor_angpos_line = R_Motor_AngPos_plot.plot(t_buffer, R_motor_angpos_buffer, pen=red)

        # 串口数据线程
        self.data_thread = DataThread()
        self.data_thread.imu_signal.connect(self.on_imu_received)

        # 语音识别线程
        self.voice_thread = VoiceThread()
        self.voice_thread.voice_command_signal.connect(self.handle_voice_command)
        self.voice_thread.chat_log_signal.connect(self.append_chat_log)

        # 刷新定时器
        self.timer = QtCore.QTimer()
        self.timer.setInterval(20)
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()

    # 🗣️ 拼接实时日志输出到 GUI 面板
    def append_chat_log(self, role, message):
        timestamp = time.strftime("%H:%M:%S")
        if role == "user":
            formatted_text = f"<span style='color: #00ffff;'>[{timestamp}] 🗣️ 用户:</span> {message}"
        elif role == "ai":
            formatted_text = f"<span style='color: #ff00ff;'>[{timestamp}] 🤖 AI:</span> {message}"
        else:
            formatted_text = f"<span style='color: #ffff00;'>[{timestamp}] 💡 系统:</span> {message}"

        self.chat_display.append(formatted_text)

    # 🎤 核心：处理语音识别返回的数据并填入输入框
    def handle_voice_command(self, cmd_json_str, reply):
        try:
            cmd_json = json.loads(cmd_json_str)
            action = cmd_json.get("action", "none")
            value = str(cmd_json.get("value", "")).strip()

            if action == "K" and value:
                self.K_text.setText(value)
                self.append_chat_log("system", f"🎉 [语音触发] 已将 K 设为: {value}")
                self.CmdButton_Clicked("K", value)  # 自动下发串口

            elif action == "Delay" and value:
                self.Delay_text.setText(value)
                self.append_chat_log("system", f"🎉 [语音触发] 已将 Delay 设为: {value}")
                self.CmdButton_Clicked("Delay", value)  # 自动下发串口

        except Exception as e:
            print(f"⚠️ 指令解析异常: {e}")

    # 🎤 语音控制开关
    def VoiceButton_Clicked(self):
        if not self.voice_thread.isRunning():
            self.voice_thread.start()
            self.VoiceButton.setText("Voice Ctrl: ON")
            self.VoiceButton.setStyleSheet("background-color : green; color: white")
        else:
            self.voice_thread.stop()
            self.VoiceButton.setText("Voice Ctrl: OFF")
            self.VoiceButton.setStyleSheet("")

    def Connect_Clicked(self):
        global ser, Connection_Flag, ConnectButton, t_0

        if Connection_Flag:
            self.data_thread.stop()
            if ser and ser.is_open:
                ser.close()
            Connection_Flag = False
            ConnectButton.setText("Connect")
            ConnectButton.setStyleSheet("")
            return

        serial_port = SerialComboBox.currentText()
        if not serial_port:
            return

        try:
            ser = serial.Serial(port=serial_port, baudrate=1000000, timeout=0.2)
            time.sleep(0.5)
            ser.reset_input_buffer()

            if ser.is_open:
                Connection_Flag = True
                ConnectButton.setText("Receiving")
                ConnectButton.setStyleSheet("background-color : green; color: white")
                t_0 = time.time()
                self.data_thread.start()
        except Exception as e:
            print(f"Connect error: {e}")
            ConnectButton.setText("Error")
            ConnectButton.setStyleSheet("background-color : red")

    def on_imu_received(self, values, pc_time):
        global L_leg_IMU_angle, R_leg_IMU_angle, LogginButton_Flag, csv_file_name, DataHeaders
        (lax, lay, laz, lgx, lgy, lgz, rax, ray, raz, rgx, rgy, rgz) = values
        L_leg_IMU_angle = lax
        R_leg_IMU_angle = rax

        if LogginButton_Flag and csv_file_name:
            LoggedData = {
                "time": pc_time,
                "L_IMU": L_leg_IMU_angle,
                "R_IMU": R_leg_IMU_angle,
                "L_Torque_d": L_motor_torque_desired,
                "L_Torque": L_motor_torque,
                "L_AngPos": L_motor_angpos,
                "R_Torque_d": R_motor_torque_desired,
                "R_Torque": R_motor_torque,
                "R_AngPos": R_motor_angpos
            }
            try:
                with open(csv_file_name, mode="a", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=DataHeaders)
                    writer.writerow(LoggedData)
            except Exception as e:
                print(f"Log Error: {e}")

    def update_plot_data(self):
        global t_buffer, L_IMU_buffer, R_IMU_buffer, L_motor_torque_buffer, R_motor_torque_buffer,\
            L_motor_torque_d_buffer, R_motor_torque_d_buffer, L_motor_angpos_buffer, R_motor_angpos_buffer, \
            L_motor_angpos_a_buffer, R_motor_angpos_a_buffer, L_leg_IMU_angle, R_leg_IMU_angle, t_0, Connection_Flag

        if Connection_Flag:
            t = time.time() - t_0

            t_buffer = t_buffer[1:] + [t]
            L_IMU_buffer = L_IMU_buffer[1:] + [L_leg_IMU_angle]
            R_IMU_buffer = R_IMU_buffer[1:] + [R_leg_IMU_angle]

            L_motor_torque_d_buffer = L_motor_torque_d_buffer[1:] + [L_motor_torque_desired]
            L_motor_torque_buffer = L_motor_torque_buffer[1:] + [L_motor_torque]

            L_motor_angpos_buffer = L_motor_angpos_buffer[1:] + [L_motor_angpos]
            L_motor_angpos_a_buffer = L_motor_angpos_a_buffer[1:] + [L_motor_angpos_a]

            R_motor_torque_d_buffer = R_motor_torque_d_buffer[1:] + [R_motor_torque_desired]
            R_motor_torque_buffer = R_motor_torque_buffer[1:] + [R_motor_torque]

            R_motor_angpos_buffer = R_motor_angpos_buffer[1:] + [R_motor_angpos]

            self.L_IMU_line.setData(t_buffer, L_IMU_buffer)  
            self.R_IMU_line.setData(t_buffer, R_IMU_buffer)   

            self.L_Motor_Taud_line.setData(t_buffer, L_motor_torque_d_buffer)
            self.L_Motor_Tau_line.setData(t_buffer, L_motor_torque_buffer)

            self.L_motor_angpos_line.setData(t_buffer, L_motor_angpos_buffer)    
            self.L_motor_angpos_a_line.setData(t_buffer, L_motor_angpos_a_buffer)    

            self.R_Motor_Taud_line.setData(t_buffer, R_motor_torque_d_buffer)  
            self.R_Motor_Tau_line.setData(t_buffer, R_motor_torque_buffer)  

            self.R_motor_angpos_line.setData(t_buffer, R_motor_angpos_buffer)

    def CmdButton_Clicked(self, param_type, text_val):
        global ser
        text_val = text_val.strip()

        if not text_val:
            return

        try:
            cmd = float(text_val)
        except ValueError:
            print("⚠️ 输入格式有误，请输入数字！")
            return

        if ser and ser.is_open:
            msg = f"{param_type}:{cmd}\n".encode('utf-8')
            ser.write(msg)
            print(f"| {param_type} Command {cmd} sent |")
        else:
            print(f"❌ 串口未连接，指令 [{param_type}:{cmd}] 未下发")

    def LogginButton_Clicked(self):
        global LogginButton_Flag, LoggingButton, csv_file_name, DataHeaders, t_0

        if not LogginButton_Flag:
            LogginButton_Flag = True
            t_0 = time.time()
            LoggingButton.setText("Logging data")
            LoggingButton.setStyleSheet("background-color : blue; color: white")
            csv_file_name = "GUI_Logger_" + time.strftime("%Y-%m-%d_%H-%M-%S") + ".csv"

            with open(csv_file_name, mode="w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=DataHeaders)
                writer.writeheader()
        else:
            LogginButton_Flag = False
            LoggingButton.setText("Data Logging")
            LoggingButton.setStyleSheet("")

    def closeEvent(self, event):
        self.data_thread.stop()
        self.voice_thread.stop()
        if ser and ser.is_open:
            ser.close()
        event.accept()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QtWidgets.QApplication(sys.argv)
    Window = MainWindow()  
    Window.show()
    sys.exit(app.exec_())