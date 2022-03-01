from django.urls import path

from Predict import views

urlpatterns = [
    path('index/',views.index,name='index'),
    path('txt_upDownload/',views.txt_upDownload,name='txt_upDownload'),
    path('submit/',views.submit,name='submit'),
]