
from django.urls import path
from . import views

app_name = 'report' # Namespace for URLs

urlpatterns = [
    path('', views.report_view, name='report'), # URL for your report page
]