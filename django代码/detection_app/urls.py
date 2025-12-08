from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
                  path('', views.home, name='home'),
                  path('image/', views.image_recognition, name='image'),
                  path('process_image/', views.process_image, name='process_image'),
                  path('video/', views.video_recognition, name='video'),
                  path('process_video/', views.process_video, name='process_video'),
                  path('camera/', views.camera_recognition, name='camera'),
                  path('video_feed/', views.video_feed, name='video_feed'),
                  path('capture/', views.capture_image, name='capture'),
                  path('stop_camera/', views.stop_camera_request, name='stop_camera'),
              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)