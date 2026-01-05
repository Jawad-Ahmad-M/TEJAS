import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from .models import Conversation, Message

User = get_user_model()

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = f'chat_{self.room_name}'

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    # Receive message from WebSocket
    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message_text = text_data_json['message']
        sender_id = self.scope['user'].id

        # Check conversation status before saving/sending
        is_accepted = await self.check_conversation_status(self.room_name)
        if not is_accepted:
            return

        # Save message to database
        await self.save_message(sender_id, self.room_name, message_text)

        # Send message to room group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': message_text,
                'sender_id': sender_id,
                'sender_username': self.scope['user'].username
            }
        )

    @database_sync_to_async
    def check_conversation_status(self, conversation_id):
        try:
            conv = Conversation.objects.get(id=conversation_id)
            return conv.status == 'ACCEPTED'
        except Conversation.DoesNotExist:
            return False

    # Receive message from room group
    async def chat_message(self, event):
        message = event['message']
        sender_id = event['sender_id']
        sender_username = event['sender_username']

        # Send message to WebSocket
        await self.send(text_data=json.dumps({
            'type': 'chat_message',
            'message': message,
            'sender_id': sender_id,
            'sender_username': sender_username
        }))

    # Handle chat status update from room group
    async def chat_status_update(self, event):
        status = event['status']

        # Send status update to WebSocket
        await self.send(text_data=json.dumps({
            'type': 'status_update',
            'status': status
        }))

    @database_sync_to_async
    def save_message(self, sender_id, conversation_id, text):
        user = User.objects.get(id=sender_id)
        conversation = Conversation.objects.get(id=conversation_id)
        return Message.objects.create(
            conversation=conversation,
            sender=user,
            text=text
        )
