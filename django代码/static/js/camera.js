document.addEventListener('DOMContentLoaded', function() {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const captureBtn = document.getElementById('captureBtn');
    const liveVideo = document.getElementById('liveVideo');
    const currentEmotion = document.getElementById('currentEmotion');
    const confidence = document.getElementById('confidence');
    const processSpeed = document.getElementById('processSpeed');
    const galleryContainer = document.getElementById('galleryContainer');

    let stream = null;
    let processing = false;
    let lastProcessTime = 0;
    const processInterval = 500; // 处理间隔(毫秒)

    // 启动摄像头
    startBtn.addEventListener('click', function() {
        if (stream) return;

        startBtn.disabled = true;
        stopBtn.disabled = false;
        captureBtn.disabled = false;

        // 获取摄像头访问权限
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(mediaStream) {
                stream = mediaStream;
                liveVideo.srcObject = stream;
                liveVideo.play();

                // 开始处理帧
                processing = true;
                processFrame();
            })
            .catch(function(err) {
                console.error("摄像头访问错误:", err);
                alert("无法访问摄像头: " + err.message);
                startBtn.disabled = false;
            });
    });

    // 停止摄像头
    stopBtn.addEventListener('click', function() {
        if (!stream) return;

        processing = false;
        stream.getTracks().forEach(track => track.stop());
        stream = null;

        startBtn.disabled = false;
        stopBtn.disabled = true;
        captureBtn.disabled = true;
    });

    // 捕获图像
    captureBtn.addEventListener('click', function() {
        if (!stream) return;

        // 创建canvas捕获当前帧
        const canvas = document.createElement('canvas');
        canvas.width = liveVideo.videoWidth;
        canvas.height = liveVideo.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(liveVideo, 0, 0, canvas.width, canvas.height);

        // 发送到服务器保存
        canvas.toBlob(function(blob) {
            const formData = new FormData();
            formData.append('image', blob, `capture_${Date.now()}.jpg`);

            fetch('/capture_image/', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-CSRFToken': getCookie('csrftoken')
                }
            })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        // 添加到相册
                        const galleryItem = document.createElement('div');
                        galleryItem.className = 'gallery-item';
                        galleryItem.innerHTML = `
                        <img src="${data.image_url}" alt="捕获的表情">
                        <div class="emotion-label">${currentEmotion.textContent}</div>
                    `;
                        galleryContainer.prepend(galleryItem);
                    } else {
                        alert('保存失败: ' + data.message);
                    }
                });
        }, 'image/jpeg');
    });

    // 处理视频帧
    function processFrame() {
        if (!processing || !stream) return;

        const now = Date.now();
        if (now - lastProcessTime < processInterval) {
            requestAnimationFrame(processFrame);
            return;
        }
        lastProcessTime = now;

        // 创建canvas捕获当前帧
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(liveVideo, 0, 0, canvas.width, canvas.height);

        // 发送到服务器处理
        canvas.toBlob(function(blob) {
            const startTime = performance.now();

            fetch('/process_frame/', {
                method: 'POST',
                body: blob,
                headers: {
                    'Content-Type': 'image/jpeg',
                    'X-CSRFToken': getCookie('csrftoken')
                }
            })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        // 更新UI
                        currentEmotion.textContent = data.dominant_emotion || '未知';
                        confidence.textContent = data.confidence ? `${(data.confidence * 100).toFixed(1)}%` : '-';

                        // 计算处理速度
                        const processTime = performance.now() - startTime;
                        processSpeed.textContent = `${Math.round(processTime)}ms`;
                    }
                })
                .finally(() => {
                    requestAnimationFrame(processFrame);
                });
        }, 'image/jpeg');
    }

    // 获取CSRF token
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
});