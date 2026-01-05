from django.urls import path
from .views import home, learn

app_name = "core"

urlpatterns = [
    path("", home, name="home"),
    path("learn_more/", learn, name="learn"),
]
