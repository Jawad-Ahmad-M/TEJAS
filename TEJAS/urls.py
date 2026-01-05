from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("core.urls")),
    path("", include("accounts.urls")),
    path("tenders/", include("tenders.urls")),
<<<<<<< HEAD
=======
    path("chat/", include("chat.urls")),
>>>>>>> bd1274c (Added Chat and rafactored code)
]
