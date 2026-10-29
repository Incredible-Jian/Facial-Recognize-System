document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const videoFile = document.getElementById('videoFile');
    const submitBtn = document.getElementById('submitBtn');
    const videoPreview = document.getElementById('videoPreview');
    const uploadedVideo = document.getElementById('uploadedVideo');
    const processingContainer = document.getElementById('processingContainer');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const resultsContainer = document.getElementById('resultsContainer');
    const originalVideo = document.getElementById('originalVideo');
    const resultVideo = document.getElementById('resultVideo');
    const statsSummary = document.getElementById('statsSummary');
    const downloadBtn = document.getElementById('downloadBtn');

    // 点击上传区域触发文件选择
    uploadArea.addEventListener('click', () => {
        videoFile.click();
    });

    // 文件选择变化时预览视频
    videoFile.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            const file = this.files[0];
            const validTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo'];

            // 检查文件类型
            if (!validTypes.includes(file.type)) {
                alert('请上传 MP4、MOV 或 AVI 格式的视频');
                return;
            }

            // 检查文件大小 (100MB)
            if (file.size > 100 * 1024 * 1024) {
                alert('视频大小不能超过 100MB');
                return;
            }

            const videoURL = URL.createObjectURL(file);
            uploadedVideo.src = videoURL;
            videoPreview.style.display = 'block';
        }
    });

    // 提交识别请求
    submitBtn.addEventListener('click', function() {
        if (!videoFile.files || videoFile.files.length === 0) {
            alert('请先选择要识别的视频');
            return;
        }

        // 显示处理中
        processingContainer.style.display = 'block';
        submitBtn.disabled = true;

        const formData = new FormData();
        formData.append('video', videoFile.files[0]);

        fetch('/process_video/', {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': getCookie('csrftoken')
            }
        })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    // 显示结果
                    originalVideo.src = data.uploaded_video;
                    resultVideo.src = data.result_video;

                    // 显示统计信息
                    let statsHTML = '<h4>表情统计</h4><ul>';
                    for (const [emotion, count] of Object.entries(data.emotion_counts)) {
                        statsHTML += `<li>${emotion}: ${count}次 (${Math.round(count/data.frame_count*100)}%)</li>`;
                    }
                    statsHTML += `</ul><p>总帧数: ${data.frame_count}</p>`;
                    statsHTML += `<p>视频时长: ${Math.round(data.duration)}秒</p>`;
                    statsSummary.innerHTML = statsHTML;

                    // 显示结果容器
                    resultsContainer.style.display = 'block';

                    // 设置下载按钮
                    downloadBtn.onclick = function() {
                        const link = document.createElement('a');
                        link.href = data.result_video;
                        link.download = '表情识别结果.mp4';
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                    };
                } else {
                    alert(`识别失败: ${data.message}`);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('处理过程中发生错误');
            })
            .finally(() => {
                processingContainer.style.display = 'none';
                submitBtn.disabled = false;
            });
    });

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