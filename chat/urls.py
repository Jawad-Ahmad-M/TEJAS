from django.urls import path
from . import views

app_name = 'chat'

urlpatterns = [
    path('', views.chat_list, name='chat_list'),
    path('search/', views.search_users, name='search_users'),
    path('<int:conversation_id>/', views.chat_room, name='chat_room'),
    path('<int:conversation_id>/accept/', views.accept_chat, name='accept_chat'),
    path('<int:conversation_id>/reject/', views.reject_chat, name='reject_chat'),
    
    # Start chat with a user (with or without tender context)
    path('start/<int:participant_id>/', views.start_chat, name='start_chat_direct'),
    path('start/<int:participant_id>/<int:tender_id>/', views.start_chat, name='start_chat'),
]
