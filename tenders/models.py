from django.db import models
from accounts.models import User  # Your accounts app user model

# ---------- Tender Categories ----------
class TenderCategory(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.name

# ---------- Tender ----------
TENDER_STATUS = [
    ('OPEN', 'Open'),
    ('CLOSED', 'Closed'),
    ('EVALUATED', 'Evaluated'),
]

class Tender(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    extracted_layout = models.JSONField(blank=True, null=True)  # For structured data from doc
    category = models.ForeignKey(TenderCategory, on_delete=models.SET_NULL, null=True)
    organization_name = models.CharField(max_length=255)
    budget_min = models.DecimalField(max_digits=12, decimal_places=2)
    budget_max = models.DecimalField(max_digits=12, decimal_places=2)
    status = models.CharField(max_length=20, choices=TENDER_STATUS, default='OPEN')
    submission_deadline = models.DateField()
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='tenders_created')
    created_at = models.DateTimeField(auto_now_add=True)


    def __str__(self):
        return self.title

# ---------- Tender Files ----------
class TenderFile(models.Model):
    tender = models.ForeignKey(Tender, on_delete=models.CASCADE, related_name='files')
    file_type = models.CharField(max_length=50)
    local_path = models.TextField()  # Or use FileField
    checksum = models.CharField(max_length=64)  # MD5/SHA256 checksum
    uploaded_at = models.DateTimeField(auto_now_add=True)

# ---------- Bid ----------
BID_STATUS = [
    ('PENDING', 'Pending'),
    ('VALID', 'Valid'),
    ('INVALID', 'Invalid'),
    ('ACCEPTED', 'Accepted'),
    ('REJECTED', 'Rejected'),
]

class Bid(models.Model):
    tender = models.ForeignKey(Tender, on_delete=models.CASCADE, related_name='bids')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='bids_submitted')
    bid_amount = models.DecimalField(max_digits=12, decimal_places=2)
    bid_text = models.TextField(blank=True, null=True)
    extracted_bid_data = models.JSONField(blank=True, null=True)
    status = models.CharField(max_length=10, choices=BID_STATUS, default='PENDING')
    submitted_at = models.DateTimeField(auto_now_add=True)

# ---------- Bid Files ----------
class BidFile(models.Model):
    bid = models.ForeignKey(Bid, on_delete=models.CASCADE, related_name='files')
    file_type = models.CharField(max_length=50)
    local_path = models.TextField()  # Or FileField
    checksum = models.CharField(max_length=64)
    uploaded_at = models.DateTimeField(auto_now_add=True)

# ---------- Rule-Based Bid Evaluation ----------
class BidEvaluation(models.Model):
    bid = models.OneToOneField(Bid, on_delete=models.CASCADE, related_name='evaluation')
    tender = models.ForeignKey(Tender, on_delete=models.CASCADE, related_name='bid_evaluations')
    price_valid = models.BooleanField(default=False)
    documents_valid = models.BooleanField(default=False)
    compliance_valid = models.BooleanField(default=True)  # Optional rules
    status = models.CharField(max_length=10, choices=BID_STATUS, default='PENDING')
    remarks = models.TextField(blank=True, null=True)
    evaluated_at = models.DateTimeField(auto_now_add=True)

    def calculate_status(self):
        """Set status based on individual checks."""
        if self.price_valid and self.documents_valid and self.compliance_valid:
            self.status = 'VALID'
        else:
            self.status = 'INVALID'
        self.save()

# ---------- Bid Analysis for Dashboard ----------
class BidAnalysis(models.Model):
    tender = models.ForeignKey(Tender, on_delete=models.CASCADE, related_name='bid_analysis')
    avg_bid = models.FloatField(blank=True, null=True)
    std_dev = models.FloatField(blank=True, null=True)
    min_bid = models.FloatField(blank=True, null=True)
    max_bid = models.FloatField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

# ---------- Bookmarks ----------
class Bookmark(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='bookmarks')
    tender = models.ForeignKey(Tender, on_delete=models.CASCADE, related_name='bookmarked_by')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'tender')

    def __str__(self):
        return f"{self.user.username} - {self.tender.title}"

# ---------- Notifications ----------
NOTIFICATION_TYPES = [
    ('BID_ACCEPTED', 'Bid Accepted'),
    ('BID_REJECTED', 'Bid Rejected'),
    ('SYSTEM', 'System Message'),
]

class Notification(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    title = models.CharField(max_length=255)
    message = models.TextField()
    notification_type = models.CharField(max_length=20, choices=NOTIFICATION_TYPES, default='SYSTEM')
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Optional link to related object
    related_link = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return f"{self.user.username} - {self.title}"
