
from django.urls import path
from . import views

app_name = 'accounts'

urlpatterns = [
    # Authentication URLs
    path('register/', views.register, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
    # Profile URLs
    path('profile/', views.profile, name='profile'),
    path('settings/', views.settings_view, name='settings'),
    path('biometric/delete/<int:biometric_id>/', views.delete_biometric, name='delete_biometric'),
    
    # Biometric API URLs
<<<<<<< HEAD
    path('api/register-face/', views.register_face, name='register_face'),
    path('api/authenticate-face/', views.authenticate_face, name='authenticate_face'),
    path('api/verify-face/', views.verify_face, name='verify_face'),
=======
    path('api/current-user/', views.current_user, name='current_user'),
    path('api/validate-registration/', views.validate_registration_data, name='validate_registration'),
    path('api/register-face/', views.register_face, name='register_face'),
    path('api/register-voice/', views.register_voice, name='register_voice'),
    path('api/authenticate-voice/', views.authenticate_voice, name='authenticate_voice'),
    path('api/authenticate-face/', views.authenticate_face, name='authenticate_face'),
    path('api/verify-face/', views.verify_face, name='verify_face'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
>>>>>>> bd1274c (Added Chat and rafactored code)
]