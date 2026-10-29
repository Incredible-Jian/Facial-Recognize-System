document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const imageFile = document.getElementById('imageFile');
    const submitBtn = document.getElementById('submitBtn');
    const processingContainer = document.getElementById('processingContainer');
    const resultsContainer = document.getElementById('resultsContainer');
    const originalImage = document.getElementById('originalImage');
    const resultImage = document.getElementById('resultImage');
    const statsSummary = document.getElementById('statsSummary');
    const downloadBtn = document.getElementById('downloadBtn');
    const inferenceTime = document.getElementById('inferenceTime');

    // 点击上传区域触发文件选择
    uploadArea.addEventListener('click', () => {
        imageFile.click();
    });

    // 文件选择变化时预览图片
    imageFile.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            const reader = new FileReader();
            reader.onload = function(e) {
                originalImage.src = e.target.result;
            };
            reader.readAsDataURL(this.files[0]);
        }
    });

    // 提交识别请求
    submitBtn.addEventListener('click', function() {
        if (!imageFile.files || imageFile.files.length === 0) {
            alert('请先选择要识别的图片');
            return;
        }

        // 显示处理中
        processingContainer.style.display = 'block';
        submitBtn.disabled = true;

        const formData = new FormData();
        formData.append('image', imageFile.files[0]);

        fetch('/process_image/', {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': getCookie('csrftoken')
            }
        })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    // 显示原始图片
                    if (!originalImage.src.startsWith('data:')) {
                        originalImage.src = data.uploaded_image;
                    }

                    // 显示结果图片
                    resultImage.src = data.result_image;

                    // 显示推理时间
                    inferenceTime.textContent = data.inference_time;

                    // 显示表情统计
                    let statsHTML = '<h4>检测到的表情</h4><ul>';
                    for (const [emotion, count] of Object.entries(data.emotions)) {
                        statsHTML += `<li>${emotion}: ${count}次</li>`;
                    }
                    statsHTML += '</ul>';
                    statsSummary.innerHTML = statsHTML;

                    // 显示结果容器
                    resultsContainer.style.display = 'block';

                    // 设置下载按钮
                    downloadBtn.onclick = function() {
                        const link = document.createElement('a');
                        link.href = data.result_image;
                        link.download = '表情识别结果.jpg';
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
    // 修复文件上传处理
    const handleUpload = async (file) => {
        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch('/process_image/', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-CSRFToken': getCookie('csrftoken')
                }
            });

            if (!response.ok) throw new Error('Network response was not ok');

            const data = await response.json();
            if (data.status === 'success') {
                // 强制刷新图片缓存
                document.getElementById('resultImage').src = data.result + `?t=${Date.now()}`;
                updateEmotionStats(data.emotions);
            } else {
                throw new Error(data.message || 'Processing failed');
            }
        } catch (error) {
            showError(error.message);
        }
    };

// 添加图片URL强制刷新
    const refreshImage = (imgElement, url) => {
        imgElement.src = url.includes('?')
            ? url.split('?')[0] + `?t=${Date.now()}`
            : url + `?t=${Date.now()}`;
    };
});