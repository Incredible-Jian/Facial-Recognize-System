import os
import cv2
import logging
import subprocess
import threading
import time
import re
import numpy as np
from datetime import datetime
from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from ultralytics import YOLO

# 初始化日志
logger = logging.getLogger(__name__)

# 加载YOLO模型
model = YOLO(os.path.join(settings.BASE_DIR, 'best.pt'))

# ========== 通用函数 ==========
def sanitize_filename(filename):
    """确保文件名是安全的ASCII名称"""
    name, ext = os.path.splitext(filename)
    name = re.sub(r'[^\w\-]', '', name)
    if not name:
        name = "video"
    return f"{name}{ext.lower()}"

def get_ffmpeg_path():
    """获取FFmpeg可执行路径"""
    if os.name == 'nt':  # Windows系统
        paths = [
            'ffmpeg',
            r'C:\ProgramData\chocolatey\bin\ffmpeg.exe',
            r'C:\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'
        ]
        for path in paths:
            try:
                subprocess.run([path, '-version'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               check=True)
                return path
            except:
                continue
        raise Exception("FFmpeg not found. Please ensure FFmpeg is installed")
    return 'ffmpeg'

# ========== 首页 ==========
def home(request):
    return render(request, "index.html")

# ========== 图片处理 ==========
def image_recognition(request):
    return render(request, 'image.html')

def process_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            img_file = request.FILES['image']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"img_{timestamp}_{img_file.name}"
            upload_path = os.path.join(settings.MEDIA_ROOT, 'uploads', filename)

            with open(upload_path, 'wb+') as dest:
                for chunk in img_file.chunks():
                    dest.write(chunk)

            img = cv2.imread(upload_path)
            results = model(img)
            annotated = results[0].plot()

            result_filename = f"result_{filename}"
            result_path = os.path.join(settings.MEDIA_ROOT, 'results', result_filename)
            cv2.imwrite(result_path, annotated)

            return JsonResponse({
                'status': 'success',
                'original': f'/media/uploads/{filename}',
                'result': f'/media/results/{result_filename}'
            })

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})

    return JsonResponse({'status': 'error', 'message': 'Invalid request'})

# ========== 视频处理 ==========
def video_recognition(request):
    return render(request, 'video.html')

def process_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        try:
            video_file = request.FILES['video']
            logger.info(f"Processing video with YOLO: {video_file.name}")

            # 生成安全的文件名和路径
            safe_filename = sanitize_filename(video_file.name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"result_{timestamp}_{safe_filename}"
            result_path = os.path.join(settings.MEDIA_ROOT, 'results', result_filename)
            os.makedirs(os.path.dirname(result_path), exist_ok=True)

            # 保存临时文件
            temp_path = os.path.join(settings.MEDIA_ROOT, 'temp', f"temp_{timestamp}.mp4")
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)

            with open(temp_path, 'wb+') as dest:
                for chunk in video_file.chunks():
                    dest.write(chunk)

            # 使用OpenCV处理视频
            cap = cv2.VideoCapture(temp_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(result_path, fourcc, fps, (frame_width, frame_height))

            # 进度跟踪
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            processed_frames = 0

            # 逐帧处理
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    # 使用YOLO处理帧
                    results = model(frame)
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)

                    # 更新进度
                    processed_frames += 1
                    if processed_frames % 10 == 0:
                        progress = int((processed_frames / total_frames) * 100)
                        logger.info(f"处理进度: {progress}%")

                except Exception as e:
                    logger.error(f"处理第{processed_frames}帧时出错: {str(e)}")
                    continue

            # 释放资源
            cap.release()
            out.release()

            # 验证生成的视频
            if not os.path.exists(result_path):
                raise ValueError("生成的视频文件无效")

            # 转换视频格式确保浏览器兼容性
            ffmpeg_path = get_ffmpeg_path()
            compatible_path = result_path.replace(".mp4", "_compatible.mp4")

            ffmpeg_cmd = [
                ffmpeg_path,
                '-y',
                '-i', result_path,
                '-c:v', 'libx264',
                '-profile:v', 'main',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '128k',
                compatible_path
            ]

            try:
                subprocess.run(
                    ffmpeg_cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    shell=True if os.name == 'nt' else False
                )
                # 使用转换后的视频
                final_path = compatible_path
                final_filename = os.path.basename(compatible_path)
            except subprocess.CalledProcessError as e:
                logger.error(f"视频转换失败，使用原始视频: {e.stderr}")
                final_path = result_path
                final_filename = os.path.basename(result_path)

            return JsonResponse({
                'status': 'success',
                'video_url': f'/media/results/{final_filename}'
            })

        except Exception as e:
            logger.error(f"Video processing error: {str(e)}", exc_info=True)
            # 清理临时文件
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try: os.remove(temp_path)
                except: pass
            if 'result_path' in locals() and os.path.exists(result_path):
                try: os.remove(result_path)
                except: pass
            if 'compatible_path' in locals() and os.path.exists(compatible_path):
                try: os.remove(compatible_path)
                except: pass

            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)

    return JsonResponse({
        'status': 'error',
        'message': '无效请求'
    }, status=400)

# ========== 摄像头处理 ==========
camera_state = {
    'active': False,
    'frame': None,
    'lock': threading.Lock(),
    'streaming': False,
    'cap': None
}

def camera_recognition(request):
    if not camera_state['active']:
        start_camera()
    return render(request, 'camera.html')

def start_camera():
    if not camera_state['active']:
        camera_state['active'] = True
        camera_state['cap'] = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        camera_state['cap'].set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera_state['cap'].set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        def camera_worker():
            while camera_state['active']:
                ret, frame = camera_state['cap'].read()
                if ret:
                    with camera_state['lock']:
                        camera_state['frame'] = frame
                time.sleep(0.05)

        threading.Thread(target=camera_worker, daemon=True).start()

def stop_camera():
    if camera_state['active']:
        camera_state['active'] = False
        if camera_state['cap'] and camera_state['cap'].isOpened():
            camera_state['cap'].release()
        camera_state['cap'] = None

def video_feed(request):
    def generate():
        camera_state['streaming'] = True
        try:
            while camera_state['streaming']:
                with camera_state['lock']:
                    if camera_state['frame'] is None:
                        time.sleep(0.1)
                        continue

                    results = model(camera_state['frame'])
                    annotated_frame = results[0].plot()

                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        finally:
            camera_state['streaming'] = False

    return StreamingHttpResponse(
        generate(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

@require_POST
@csrf_exempt
def capture_image(request):
    try:
        with camera_state['lock']:
            if camera_state['frame'] is None:
                return JsonResponse({'status': 'error', 'message': '未获取到摄像头画面'})

            results = model(camera_state['frame'])
            annotated_frame = results[0].plot()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            save_path = os.path.join(settings.MEDIA_ROOT, 'captures', filename)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, annotated_frame)

            return JsonResponse({
                'status': 'success',
                'image_url': f'/media/captures/{filename}'
            })
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})

@require_POST
@csrf_exempt
def stop_camera_request(request):
    stop_camera()
    return JsonResponse({'status': 'success'})