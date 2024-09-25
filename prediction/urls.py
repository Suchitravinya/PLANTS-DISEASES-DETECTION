# prediction/urls.py
from django import views
from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from .import views

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('disease-recognition/', views.disease_recognition, name='disease_recognition'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
