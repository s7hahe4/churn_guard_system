from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    # This tells the main project to look at our predictions app for URLs
    path('', include('predictions.urls')), 
]