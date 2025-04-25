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

from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import Text2VideoPipeline, Image2VideoPipeline
from skyreels_v2_infer.pipelines import resizecrop

# The future may be packaged as a docker for tora users to use.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

builtin_print = __builtin__.print
app = Flask(__name__)
PORT = 5001

# Task management
tasks = {}
task_lock = Lock()

class TaskStatus:
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

def create_task():
    task_id = str(uuid.uuid4())
    with task_lock:
        tasks[task_id] = {
            "status": TaskStatus.PENDING,
            "result": None,
            "error": None
        }
    return task_id

def update_task_status(task_id, status, result=None, error=None):
    with task_lock:
        if task_id in tasks:
            tasks[task_id]["status"] = status
            if result is not None:
                tasks[task_id]["result"] = result
            if error is not None:
                tasks[task_id]["error"] = error

def get_task_status(task_id):
    with task_lock:
        return tasks.get(task_id)

def myhash(txt):
    return sha256(txt.encode('utf-8')).hexdigest()

def request_parse(req_data):
    if request.is_json:
        data = req_data.json
    else:
        data = req_data.args
    return data

def get_container_ip():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address

def base64EncodeVideo(video_file_path):
    with open(video_file_path, 'rb') as video_file:
        video_data = video_file.read()
        return base64.b64encode(video_data).decode('utf-8')

@torch.inference_mode()
def run_text2video(prompt):
    global text_pipe
    
    # Default parameters from generate_video.py
    negative_prompt = ""
    num_frames = 97
    num_inference_steps = 30
    guidance_scale = 6.0
    shift = 8.0
    height = 544
    width = 960
    
    # Generate random seed if not provided
    random.seed(time.time())
    seed = int(random.randrange(4294967294))
    
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "shift": shift,
        "generator": torch.Generator(device="cuda").manual_seed(seed),
        "height": height,
        "width": width,
    }
    
    video_frames = text_pipe(**kwargs)[0]
    return video_frames

@torch.inference_mode()
def run_image2video(prompt, image, negative_prompt=None):
    global image_pipe
    
    # Default parameters from generate_video.py
    if negative_prompt is None:
        negative_prompt = ""
    num_frames = 97
    num_inference_steps = 30
    guidance_scale = 6.0
    shift = 8.0
    height = 544
    width = 960
    
    # Generate random seed if not provided
    random.seed(time.time())
    seed = int(random.randrange(4294967294))
    
    # Process image
    image = Image.open(io.BytesIO(image)).convert("RGB")
    image_width, image_height = image.size
    if image_height > image_width:
        height, width = width, height
    image = resizecrop(image, height, width)
    
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "shift": shift,
        "generator": torch.Generator(device="cuda").manual_seed(seed),
        "height": height,
        "width": width,
        "image": image
    }
    
    video_frames = image_pipe(**kwargs)[0]
    return video_frames

def setup():
    global text_pipe
    global image_pipe
    
    # Setup text2video pipeline
    text_model_id = "Skywork/SkyReels-V2-T2V-14B-540P"
    text_model_id = download_model(text_model_id)
    print("text model_id:", text_model_id)
    
    text_pipe = Text2VideoPipeline(
        model_path=text_model_id,
        dit_path=text_model_id,
        use_usp=False,
        offload=True
    )
    
    # Setup image2video pipeline
    image_model_id = "Skywork/SkyReels-V2-I2V-14B-540P"
    image_model_id = download_model(image_model_id)
    print("image model_id:", image_model_id)
    
    image_pipe = Image2VideoPipeline(
        model_path=image_model_id,
        dit_path=image_model_id,
        use_usp=False,
        offload=True
    )
    
    folder_path = 'cache'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

@app.route("/text2video", methods=['POST'])
def text2video():
    try:
        req = request_parse(request)
        prompt = req['prompt']  # prompt is required
    except Exception:
        print("[-] failed to parse request")
        return {"content": "request params decode error", "status": "FAIL"}
    
    h = str(myhash(prompt))
    filepath = "cache/" + h + ".mp4"
    
    # Check cache
    if os.path.isfile(filepath): 
        print("[+] file exists, filepath: ", filepath)
        content = base64EncodeVideo(filepath)
        return {"content": content, "status": "OK"}
    
    # Create new task
    task_id = create_task()
    
    # Start async processing
    def process_task():
        try:
            update_task_status(task_id, TaskStatus.PROCESSING)
            video = run_text2video(prompt)
            export_to_video(video, filepath, fps=24)
            content = base64EncodeVideo(filepath)
            update_task_status(task_id, TaskStatus.COMPLETED, result=content)
        except Exception as e:
            print("[-] request inference failed" + str(e))
            update_task_status(task_id, TaskStatus.FAILED, error=str(e))
    
    # Start processing in background
    import threading
    thread = threading.Thread(target=process_task)
    thread.start()
    
    return {"task_id": task_id, "status": "PENDING"}

@app.route("/image2video", methods=['POST'])
def image2video():
    try:
        req = request_parse(request)
        prompt = req['prompt']  # prompt is required
        image = base64.b64decode(req['image'])  # image is required as base64 string
        negative_prompt = req.get('negative_prompt')  # negative_prompt is optional
    except Exception:
        print("[-] failed to parse request")
        return {"content": "request params decode error", "status": "FAIL"}
    
    print("[+] request params:", {k: v for k, v in req.items() if k != 'image'})
    h = str(myhash(prompt + str(image) + (negative_prompt if negative_prompt else "")))
    filepath = "cache/" + h + ".mp4"
    
    # Check cache
    if os.path.isfile(filepath): 
        print("[+] file exists, filepath: ", filepath)
        content = base64EncodeVideo(filepath)
        return {"content": content, "status": "OK"}
    
    # Create new task
    task_id = create_task()
    
    # Start async processing
    def process_task():
        try:
            update_task_status(task_id, TaskStatus.PROCESSING)
            video = run_image2video(prompt, image)
            export_to_video(video, filepath, fps=24)
            content = base64EncodeVideo(filepath)
            update_task_status(task_id, TaskStatus.COMPLETED, result=content)
        except Exception as e:
            print("[-] request inference failed" + str(e))
            update_task_status(task_id, TaskStatus.FAILED, error=str(e))
    
    # Start processing in background
    import threading
    thread = threading.Thread(target=process_task)
    thread.start()
    
    return {"task_id": task_id, "status": "PENDING"}

@app.route("/result/<hash_id>", methods=['GET'])
def get_task_result(hash_id):
    task = get_task_status(hash_id)
    if not task:
        return {"error": "Task not found", "status": "FAIL"}
    
    if task["status"] == TaskStatus.COMPLETED:
        return {"content": task["result"], "status": "OK"}
    elif task["status"] == TaskStatus.FAILED:
        return Response({"content": "inference error", "status": "FAIL"}, status=500)
    else:
        return Response({"content": "Your request is in progress, please wait...", "status": task["status"]}, status=202)

@app.route("/hash", methods=['GET'])
def model_program_hash():
    model_hash = "0xaf22a2daec6a97480c7bdab871d1c327417eea8f8ae975042afd7dc389af5ca2"
    program_hash = "0xc33e8fb8314f854c2418953750c7b2cfc27a9aed7c9eaa17bf6354a664e8b133"
    return {"modelHash": model_hash, "programHash": program_hash, "status": "OK"}

if __name__ == "__main__":
    setup()
    waitress.serve(app, host=get_container_ip(), port=PORT)