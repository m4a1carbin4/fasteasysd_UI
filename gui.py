from fasteasySD import fasteasySD as fesd
import torch
import sys
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QSlider,
    QTabWidget,
    QSpacerItem,
    QSizePolicy,
    QComboBox,
    QCheckBox,
    QTextEdit,
    QFileDialog,
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import (
    QSize,
    pyqtSignal,
    pyqtSlot,
    QObject,
    QRunnable,
    QThreadPool,
    Qt,
)
from PIL.ImageQt import ImageQt
import traceback, sys
import os
from uuid import uuid4
import numpy as np

RESULTS_DIRECTORY = "results"

def get_lcm_diffusion_pipeline_path():
    main_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(
        main_path,
        "src",
        "models",
    )
    return file_path


def get_results_path():
    app_dir = os.path.dirname(__file__)
    config_path = os.path.join(app_dir, RESULTS_DIRECTORY)
    return config_path


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FastEasySD CPU")
        self.setFixedSize(QSize(530, 600))
        self.init_ui()
        self.threadpool = QThreadPool()
        self.output_path = get_results_path()
        self.seed_value.setEnabled(False)
        self.previous_width = 0
        self.previous_height = 0
        print(f"Output path : { self.output_path}")
        self.base_model.setEnabled(True)
        self.fesd = None
        self.model_changed = True
        self.use_lora = False
        self.previous_model = ""
        self.device_changed = True
        self.im = None

    def init_ui(self):
        self.create_main_tab()
        self.create_settings_tab()
        self.create_about_tab()
        self.show()

    def create_main_tab(self):
        self.img = QLabel("<<Image>>")
        self.img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        #self.img.setFixedSize(QSize(512, 512))

        self.prompt = QTextEdit()
        self.prompt.setPlaceholderText("add postive prompt : masterpiece, best quality, chamcham(twitch), hair bell, hair ribbon, multicolored hair, two-tone hair, 1girl, solo,")
        self.generate = QPushButton("Generate")
        self.generate.clicked.connect(self.text_to_image)
        self.prompt.setFixedHeight(35)

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.prompt)
        hlayout.addWidget(self.generate)
        
        self.n_prompt = QTextEdit()
        self.n_prompt.setPlaceholderText("add negative prompt : ex :) bad hand,text,watermark,low quality,medium quality")
        self.n_prompt.setFixedHeight(35)
        
        hnlayout = QHBoxLayout()
        hnlayout.addWidget(self.n_prompt)

        vlayout = QVBoxLayout()
        vlayout.addWidget(self.img)
        vlayout.addLayout(hlayout)
        vlayout.addLayout(hnlayout)

        self.tab_widget = QTabWidget(self)
        self.tab_main = QWidget()
        self.tab_settings = QWidget()
        self.tab_about = QWidget()
        self.tab_main.setLayout(vlayout)

        self.tab_widget.addTab(self.tab_main, "Text to Image")
        self.tab_widget.addTab(self.tab_settings, "Settings")
        self.tab_widget.addTab(self.tab_about, "About")

        self.setCentralWidget(self.tab_widget)
        self.use_seed = False

    def create_settings_tab(self):
        device_hlayout = QHBoxLayout()
        self.device_label = QLabel("Model Type:")
        self.device = QComboBox(self)
        self.device.addItem("cpu")
        self.device.addItem("cuda")
        self.device.currentIndexChanged.connect(self.device_update)
        
        device_hlayout.addWidget(self.device_label)
        device_hlayout.addWidget(self.device)
        
        model_type_hlayout = QHBoxLayout()
        self.model_type_label = QLabel("Model Type:")
        self.type = QComboBox(self)
        self.type.addItem("SD")
        self.type.addItem("SDXL")
        self.type.addItem("SSD-1B")
        self.type.addItem("LCM")
        
        model_type_hlayout.addWidget(self.model_type_label)
        model_type_hlayout.addWidget(self.type)
        
        model_hlayout = QHBoxLayout()
        self.base_model_label = QLabel("Base Model:")
        self.base_model = QLineEdit("runwayml/stable-diffusion-v1-5")
        self.find_model = QPushButton("find model")
        self.find_model.clicked.connect(self.find_model_src)
        model_hlayout.addWidget(self.base_model_label)
        model_hlayout.addWidget(self.base_model)
        model_hlayout.addWidget(self.find_model)
        
        lora_hlayout = QHBoxLayout()
        self.lora_check = QCheckBox("Use lora")
        self.lora_check.stateChanged.connect(self.lora_changed)
        self.lora_model_label = QLabel("lora:")
        self.lora_model = QLineEdit("./chamcham_new_train_lora_2-000001.safetensors")
        self.find_lora = QPushButton("find lora")
        self.find_lora.clicked.connect(self.find_lora_src)
        lora_hlayout.addWidget(self.lora_check)
        lora_hlayout.addWidget(self.lora_model_label)
        lora_hlayout.addWidget(self.lora_model)
        lora_hlayout.addWidget(self.find_lora)

        self.inference_steps_value = QLabel("Number of inference steps: 8")
        self.inference_steps = QSlider(orientation=Qt.Orientation.Horizontal)
        self.inference_steps.setMaximum(25)
        self.inference_steps.setMinimum(1)
        self.inference_steps.setValue(8)
        self.inference_steps.valueChanged.connect(self.update_label)

        self.guidance_value = QLabel("Guidance scale: 2")
        self.guidance = QSlider(orientation=Qt.Orientation.Horizontal)
        self.guidance.setMaximum(200)
        self.guidance.setMinimum(10)
        self.guidance.setValue(20)
        self.guidance.valueChanged.connect(self.update_guidance_label)

        self.width_value = QLabel("Width :")
        self.width = QComboBox(self)
        self.width.addItem("256")
        self.width.addItem("512")
        self.width.addItem("768")
        self.width.addItem("960")
        self.width.addItem("1024")
        self.width.setCurrentText("512")

        self.height_value = QLabel("Height :")
        self.height = QComboBox(self)
        self.height.addItem("256")
        self.height.addItem("512")
        self.height.addItem("768")
        self.height.addItem("960")
        self.height.addItem("1024")
        self.height.setCurrentText("512")

        self.seed_check = QCheckBox("Use seed")
        self.seed_check.stateChanged.connect(self.seed_changed)
        self.seed_value = QLineEdit()
        self.seed_value.setInputMask("9999999999")
        self.seed_value.setText("123123")

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.seed_check)
        hlayout.addWidget(self.seed_value)
        hspacer = QSpacerItem(20, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        slider_hspacer = QSpacerItem(20, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        vlayout = QVBoxLayout()
        vspacer = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        vlayout.addItem(hspacer)
        vlayout.addLayout(device_hlayout)
        vlayout.addLayout(model_type_hlayout)
        vlayout.addLayout(model_hlayout)
        vlayout.addLayout(lora_hlayout)
        vlayout.addItem(slider_hspacer)
        vlayout.addWidget(self.inference_steps_value)
        vlayout.addWidget(self.inference_steps)
        vlayout.addWidget(self.width_value)
        vlayout.addWidget(self.width)
        vlayout.addWidget(self.height_value)
        vlayout.addWidget(self.height)
        vlayout.addWidget(self.guidance_value)
        vlayout.addWidget(self.guidance)
        vlayout.addLayout(hlayout)
        vlayout.addItem(vspacer)
        self.tab_settings.setLayout(vlayout)

    def create_about_tab(self):
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setText(
            """<h1>FasteasySD_UI v0.0.1</h1> 
               <h3> GUI origin (c)2023 - Rupesh Sreeraman</h3>
               <h3> chamcham AI dev (c)2023 - WGNW_MGM</h3>
                <h3>Faster stable diffusion or stable diffusion XL on CPU</h3>
                 <h3>Based on Latent Consistency Models</h3>
                <h3>GitHub : https://github.com/rupeshs/fastsdcpu/</h3>"""
        )

        vlayout = QVBoxLayout()
        vlayout.addWidget(self.label)
        self.tab_about.setLayout(vlayout)

    def update_label(self, value):
        self.inference_steps_value.setText(f"Number of inference steps: {value}")

    def update_guidance_label(self, value):
        val = round(int(value) / 10, 1)
        self.guidance_value.setText(f"Guidance scale: {val}")

    def seed_changed(self, state):
        if state == 2:
            self.use_seed = True
            self.seed_value.setEnabled(True)
        else:
            self.use_seed = False
            self.seed_value.setEnabled(False)
            
    def lora_changed(self, state):
        if state == 2:
            self.use_lora = True
        else:
            self.use_lora = False
        
        self.model_changed = True
        
    def device_update(self):
        self.device_changed = True

    def generate_image(self):
        
        if self.device_changed and self.device.currentText() == "cpu":
            self.fesd = fesd.FastEasySD(device='cpu',use_fp16=False)
            print("cpu pipeline")
            self.device_changed = False
            
        elif self.device_changed and self.device.currentText() == "cuda":
            self.fesd = fesd.FastEasySD(device='cuda',use_fp16=True)
            print("cuda pipeline")
            self.device_changed = False
        
        prompt = self.prompt.toPlainText()
        n_prompt = self.n_prompt.toPlainText()
        guidance_scale = round(int(self.guidance.value()) / 10, 1)
        img_width = int(self.width.currentText())
        img_height = int(self.height.currentText())
        num_inference_steps = self.inference_steps.value()

        if self.use_seed:
            cur_seed = int(self.seed_value.text())
            torch.manual_seed(cur_seed)
        else :
            cur_seed = int(0)

        print(f"Prompt : {prompt}")
        print(f"Resolution : {img_width} x {img_height}")
        print(f"Guidance Scale : {guidance_scale}")
        print(f"Inference_steps  : {num_inference_steps}")
        if self.use_seed:
            print(f"Seed: {cur_seed}")
        else :
            print("Seed: Random")

        image_id = uuid4()
        
        if self.model_changed and self.fesd is not None:
            self.fesd.makeSampler()
            self.model_changed = False
        
        if self.use_lora:
        
            images = self.fesd.make(mode="txt2img",
                    model_type=self.type.currentText(),model_path=self.base_model.text(),
                    lora_path=os.path.dirname(self.lora_model.text()),lora_name=os.path.basename(self.lora_model.text()),
                    prompt=prompt,
                    n_prompt=n_prompt,
                    seed=cur_seed,steps=num_inference_steps,cfg=guidance_scale,height=img_height,width=img_width)
            
        else :
            images = self.fesd.make(mode="txt2img",
                    model_type=self.type.currentText(),model_path=self.base_model.text(),
                    #lora_path=os.path.dirname(self.lora_model.text()),lora_name=os.path.basename(self.lora_model.text()),
                    prompt=prompt,
                    n_prompt=n_prompt,
                    seed=cur_seed,steps=num_inference_steps,cfg=guidance_scale,height=img_height,width=img_width)
                
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
            
        images = self.fesd.return_PIL(images)

        images[0].save(os.path.join(self.output_path, f"{image_id}.png"))
        print(f"Image {image_id}.png saved")
        self.im = ImageQt(images[0]).copy()

    def text_to_image(self):
        self.img.setText("Please wait...")
        worker = Worker(self.generate_image)
        self.threadpool.start(worker)
        self.threadpool.waitForDone(-1)
        pixmap = QPixmap.fromImage(self.im)
        pixmap = pixmap.scaled(QSize(512,768),aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatioByExpanding,transformMode=Qt.TransformationMode.FastTransformation)
        self.img.setPixmap(pixmap)
        self.setFixedSize(self.tab_main.sizeHint())
        self.previous_model = "milkyWonderland_v20.safetensors"
        
    def find_model_src(self):
        model_src=QFileDialog.getOpenFileName(self)
        self.base_model.setText(model_src[0])
        self.model_changed = True
            
    def find_lora_src(self):
        model_src=QFileDialog.getOpenFileName(self)
        self.lora_model.setText(model_src[0])
        self.model_changed = True

    def latents_callback(self, i, t, latents):
        print(i)


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()