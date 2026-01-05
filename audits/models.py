from django.db import models

from accounts.models import User

class AuditLog(models.Model):
    actor = models.ForeignKey(User, on_delete=models.CASCADE, related_name='audit_logs')
    action = models.CharField(max_length=255)   # e.g., "Created Tender", "Submitted Bid"
    entity = models.CharField(max_length=50)    # e.g., "Tender", "Bid"
    entity_id = models.PositiveIntegerField()   # ID of the tender/bid affected
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.actor.full_name} performed {self.action} on {self.entity}({self.entity_id})"
