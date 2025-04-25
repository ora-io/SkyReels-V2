"""Script to generate text from a trained model using HuggingFace wrappers."""

import argparse
import json
import os
import socket
import base64
import builtins as __builtin__
from flask import Flask, request, Response
import waitress
import torch
from hashlib import sha256
import random
import time
import uuid
from diffusers.utils import load_image, export_to_video
from PIL import Image
import io
from threading import Lock
from typing import Dict, Optional, Any

from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import Text2VideoPipeline, Image2VideoPipeline
from skyreels_v2_infer.pipelines import resizecrop


# ========== Configuration ==========
class Config:
    PORT = 5001
    CACHE_DIR = "cache"
    DEFAULT_FPS = 24
    DEFAULT_FRAMES = 97
    DEFAULT_STEPS = 30
    DEFAULT_GUIDANCE_SCALE = 6.0
    DEFAULT_SHIFT = 8.0
    DEFAULT_HEIGHT = 544
    DEFAULT_WIDTH = 960


# ========== Task Management ==========
class TaskStatus:
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.lock = Lock()

    def create_task(self) -> str:
        task_id = str(uuid.uuid4())
        with self.lock:
            self.tasks[task_id] = {
                "status": TaskStatus.PENDING,
                "result": None,
                "error": None,
            }
        return task_id

    def update_task(
        self, task_id: str, status: str, result: Any = None, error: Any = None
    ) -> None:
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id]["status"] = status
                if result is not None:
                    self.tasks[task_id]["result"] = result
                if error is not None:
                    self.tasks[task_id]["error"] = error

    def get_task(self, task_id: str) -> Optional[Dict]:
        with self.lock:
            return self.tasks.get(task_id)


# ========== Utility Functions ==========
class Utils:
    @staticmethod
    def myhash(txt: str) -> str:
        return sha256(txt.encode("utf-8")).hexdigest()

    @staticmethod
    def parse_request(req_data) -> Dict:
        if request.is_json:
            return req_data.json
        return req_data.args

    @staticmethod
    def get_container_ip() -> str:
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)

    @staticmethod
    def encode_video_to_base64(video_file_path: str) -> str:
        with open(video_file_path, "rb") as video_file:
            video_data = video_file.read()
            return base64.b64encode(video_data).decode("utf-8")


# ========== Model Pipeline ==========
class ModelPipeline:
    def __init__(self):
        self.text_pipe = None
        self.image_pipe = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(self):
        # Setup text2video pipeline
        text_model_id = "Skywork/SkyReels-V2-T2V-14B-540P"
        text_model_id = download_model(text_model_id)
        print("text model_id:", text_model_id)

        self.text_pipe = Text2VideoPipeline(
            model_path=text_model_id,
            dit_path=text_model_id,
            use_usp=False,
            offload=True,
        )

        # Setup image2video pipeline
        image_model_id = "Skywork/SkyReels-V2-I2V-14B-540P"
        image_model_id = download_model(image_model_id)
        print("image model_id:", image_model_id)

        self.image_pipe = Image2VideoPipeline(
            model_path=image_model_id,
            dit_path=image_model_id,
            use_usp=False,
            offload=True,
        )

        if not os.path.exists(Config.CACHE_DIR):
            os.makedirs(Config.CACHE_DIR)

    @torch.inference_mode()
    def run_text2video(self, prompt: str) -> torch.Tensor:
        random.seed(time.time())
        seed = int(random.randrange(4294967294))

        kwargs = {
            "prompt": prompt,
            "negative_prompt": "",
            "num_frames": Config.DEFAULT_FRAMES,
            "num_inference_steps": Config.DEFAULT_STEPS,
            "guidance_scale": Config.DEFAULT_GUIDANCE_SCALE,
            "shift": Config.DEFAULT_SHIFT,
            "generator": torch.Generator(device="cuda").manual_seed(seed),
            "height": Config.DEFAULT_HEIGHT,
            "width": Config.DEFAULT_WIDTH,
        }

        return self.text_pipe(**kwargs)[0]

    @torch.inference_mode()
    def run_image2video(
        self, prompt: str, image: bytes, negative_prompt: Optional[str] = None
    ) -> torch.Tensor:
        if negative_prompt is None:
            negative_prompt = ""

        random.seed(time.time())
        seed = int(random.randrange(4294967294))

        # Process image
        image = Image.open(io.BytesIO(image)).convert("RGB")
        image_width, image_height = image.size
        height, width = Config.DEFAULT_HEIGHT, Config.DEFAULT_WIDTH
        if image_height > image_width:
            height, width = width, height
        image = resizecrop(image, height, width)

        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_frames": Config.DEFAULT_FRAMES,
            "num_inference_steps": Config.DEFAULT_STEPS,
            "guidance_scale": Config.DEFAULT_GUIDANCE_SCALE,
            "shift": Config.DEFAULT_SHIFT,
            "generator": torch.Generator(device="cuda").manual_seed(seed),
            "height": height,
            "width": width,
            "image": image,
        }

        return self.image_pipe(**kwargs)[0]


# ========== Flask Application ==========
app = Flask(__name__)
task_manager = TaskManager()
model_pipeline = ModelPipeline()
utils = Utils()


@app.route("/text2video", methods=["POST"])
def text2video():
    try:
        req = utils.parse_request(request)
        prompt = req["prompt"]
    except Exception:
        print("[-] failed to parse request")
        return {"content": "request params decode error", "status": "FAIL"}

    task_id = task_manager.create_task()

    def process_task():
        try:
            task_manager.update_task(task_id, TaskStatus.PROCESSING)
            h = utils.myhash(prompt)
            filepath = f"{Config.CACHE_DIR}/{h}.mp4"

            if os.path.isfile(filepath):
                print("[+] file exists, filepath: ", filepath)
                content = utils.encode_video_to_base64(filepath)
                task_manager.update_task(task_id, TaskStatus.COMPLETED, result=content)
            else:
                video = model_pipeline.run_text2video(prompt)
                export_to_video(video, filepath, fps=Config.DEFAULT_FPS)
                content = utils.encode_video_to_base64(filepath)
                task_manager.update_task(task_id, TaskStatus.COMPLETED, result=content)
        except Exception as e:
            print("[-] request inference failed" + str(e))
            task_manager.update_task(task_id, TaskStatus.FAILED, error=str(e))

    import threading

    thread = threading.Thread(target=process_task)
    thread.start()

    return {"task_id": task_id, "status": "PENDING"}


@app.route("/image2video", methods=["POST"])
def image2video():
    try:
        req = utils.parse_request(request)
        prompt = req["prompt"]
        image = base64.b64decode(req["image"])
        negative_prompt = req.get("negative_prompt")
    except Exception:
        print("[-] failed to parse request")
        return {"content": "request params decode error", "status": "FAIL"}

    print("[+] request params:", {k: v for k, v in req.items() if k != "image"})

    task_id = task_manager.create_task()

    def process_task():
        try:
            task_manager.update_task(task_id, TaskStatus.PROCESSING)
            h = utils.myhash(
                prompt + str(image) + (negative_prompt if negative_prompt else "")
            )
            filepath = f"{Config.CACHE_DIR}/{h}.mp4"

            if os.path.isfile(filepath):
                print("[+] file exists, filepath: ", filepath)
                content = utils.encode_video_to_base64(filepath)
                task_manager.update_task(task_id, TaskStatus.COMPLETED, result=content)
            else:
                video = model_pipeline.run_image2video(prompt, image, negative_prompt)
                export_to_video(video, filepath, fps=Config.DEFAULT_FPS)
                content = utils.encode_video_to_base64(filepath)
                task_manager.update_task(task_id, TaskStatus.COMPLETED, result=content)
        except Exception as e:
            print("[-] request inference failed" + str(e))
            task_manager.update_task(task_id, TaskStatus.FAILED, error=str(e))

    import threading

    thread = threading.Thread(target=process_task)
    thread.start()

    return {"task_id": task_id, "status": "PENDING"}


@app.route("/result/<hash_id>", methods=["GET"])
def get_task_result(hash_id):
    task = task_manager.get_task(hash_id)
    if not task:
        return {"error": "Task not found", "status": "FAIL"}

    if task["status"] == TaskStatus.COMPLETED:
        return {"content": task["result"], "status": "OK"}
    elif task["status"] == TaskStatus.FAILED:
        return Response({"content": "inference error", "status": "FAIL"}, status=500)
    else:
        return Response(
            {
                "content": "Your request is in progress, please wait...",
                "status": task["status"],
            },
            status=202,
        )


@app.route("/hash", methods=["GET"])
def model_program_hash():
    model_hash = "0xaf22a2daec6a97480c7bdab871d1c327417eea8f8ae975042afd7dc389af5ca2"
    program_hash = "0xc33e8fb8314f854c2418953750c7b2cfc27a9aed7c9eaa17bf6354a664e8b133"
    return {"modelHash": model_hash, "programHash": program_hash, "status": "OK"}


if __name__ == "__main__":
    model_pipeline.setup()
    waitress.serve(app, host=utils.get_container_ip(), port=Config.PORT)
