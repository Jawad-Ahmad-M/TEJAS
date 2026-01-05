from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from .models import Conversation, Message
from django.db.models import Q
from django.contrib.auth import get_user_model
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from tenders.models import Tender

User = get_user_model()

@login_required
def chat_list(request):
    # Active conversations
    active_conversations = request.user.conversations.filter(status='ACCEPTED').order_by('-updated_at')
    # Incoming pending requests
    pending_requests = request.user.conversations.filter(status='PENDING').exclude(initiator=request.user).order_by('-created_at')
    # Sent requests
    sent_requests = request.user.conversations.filter(status='PENDING', initiator=request.user).order_by('-created_at')
    
    return render(request, 'chat/chat_list.html', {
        'active_conversations': active_conversations,
        'pending_requests': pending_requests,
        'sent_requests': sent_requests
    })

@login_required
def chat_room(request, conversation_id):
    conversation = get_object_or_404(Conversation, id=conversation_id)
    if request.user not in conversation.participants.all():
        return redirect('chat:chat_list')
    
    # If pending and user is the recipient, they should see an acceptance UI
    is_pending = conversation.status == 'PENDING'
    is_initiator = conversation.initiator == request.user
    
    messages = conversation.messages.all()
    return render(request, 'chat/chat_room.html', {
        'conversation': conversation,
        'chat_messages': messages,
        'is_pending': is_pending,
        'is_initiator': is_initiator
    })

@login_required
def start_chat(request, participant_id, tender_id=None):
    recipient = get_object_or_404(User, id=participant_id)
    if recipient == request.user:
        return redirect('chat:chat_list')

    # Find ANY existing conversation between these two users (ignoring tender context to enforce singleton)
    conversation = Conversation.objects.filter(participants=request.user).filter(participants=recipient).first()
    
    if not conversation:
        tender = None
        if tender_id:
            tender = get_object_or_404(Tender, id=tender_id)
            
        startup_message = ""
        if request.method == 'POST':
            startup_message = request.POST.get('startup_message', '').strip()
        
        conversation = Conversation.objects.create(
            tender=tender,
            initiator=request.user,
            status='PENDING'
        )
        conversation.participants.add(request.user, recipient)
        
        if startup_message:
            Message.objects.create(
                conversation=conversation,
                sender=request.user,
                text=startup_message
            )
            
    return redirect('chat:chat_room', conversation_id=conversation.id)

@login_required
def search_users(request):
    query = request.GET.get('q', '')
    users = []
    if query:
        users = User.objects.filter(
            Q(username__icontains=query) | Q(email__icontains=query)
        ).exclude(id=request.user.id)[:10]
        
    return render(request, 'chat/search_users.html', {'users': users, 'query': query})

@login_required
def accept_chat(request, conversation_id):
    conversation = get_object_or_404(Conversation, id=conversation_id, participants=request.user)
    # Only the recipient (not initiator) can accept
    if conversation.status == 'PENDING' and conversation.initiator != request.user:
        conversation.status = 'ACCEPTED'
        conversation.save()
        
        # Notify the initiator via WebSocket that the chat has been accepted
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f'chat_{conversation.id}',
            {
                'type': 'chat_status_update',
                'status': 'ACCEPTED'
            }
        )
    return redirect('chat:chat_room', conversation_id=conversation.id)

@login_required
def reject_chat(request, conversation_id):
    conversation = get_object_or_404(Conversation, id=conversation_id, participants=request.user)
    # Only the recipient (not initiator) can reject
    if conversation.status == 'PENDING' and conversation.initiator != request.user:
        conversation.status = 'REJECTED'
        conversation.save()
    return redirect('chat:chat_list')
