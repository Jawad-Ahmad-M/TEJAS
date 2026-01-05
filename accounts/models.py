from django.db import models
from django.contrib.auth.models import AbstractUser, Group, Permission

# -------------------------------
# Custom User Model
# -------------------------------
class User(AbstractUser):
    first_name = models.CharField(max_length=255)
    last_name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    company_name = models.CharField(max_length=255, blank=True, null=True)
    phone = models.CharField(max_length=20, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    groups = models.ManyToManyField(
        Group,
        related_name="accounts_user_set",
        blank=True,
        help_text='The groups this user belongs to.',
        verbose_name='groups',
    )
    user_permissions = models.ManyToManyField(
        Permission,
        related_name="accounts_user_permissions_set",
        blank=True,
        help_text='Specific permissions for this user.',
        verbose_name='user permissions',
    )

    # Optional: Use email as the username field
    # USERNAME_FIELD = 'email'
    # REQUIRED_FIELDS = ['username', 'first_name', 'last_name']

    def __str__(self):
        return self.username

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}".strip()


# -------------------------------
# Biometric Profiles (Face/Voice)
# -------------------------------
class BiometricProfile(models.Model):
    BIOMETRIC_CHOICES = (
        ('FACE', 'Face'),
        ('VOICE', 'Voice'),
    )

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='biometric_profiles')
    biometric_type = models.CharField(max_length=10, choices=BIOMETRIC_CHOICES)
    local_path = models.TextField()  # Path to the stored image/audio
    checksum = models.CharField(max_length=64, blank=True, null=True)  # For verifying file integrity
    model_version = models.CharField(max_length=50, blank=True, null=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.biometric_type}"


# -------------------------------
# Biometric Logs (optional, audit trail)
# -------------------------------
class BiometricLog(models.Model):
    BIOMETRIC_CHOICES = (
        ('FACE', 'Face'),
        ('VOICE', 'Voice'),
    )

    ACTION_CHOICES = (
        ('REGISTER', 'Register'),
        ('LOGIN', 'Login'),
        ('VERIFY', 'Verify'),
    )

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='biometric_logs')
    biometric_type = models.CharField(max_length=10, choices=BIOMETRIC_CHOICES)
    action = models.CharField(max_length=10, choices=ACTION_CHOICES)
    success = models.BooleanField(default=False)
    confidence = models.FloatField(blank=True, null=True)  # Face/voice recognition confidence
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.biometric_type} - {self.action} - {'Success' if self.success else 'Fail'}"


# -------------------------------
# User Statistics
# -------------------------------
class UserStats(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='stats')
    tenders_created = models.PositiveIntegerField(default=0)
    bids_submitted = models.PositiveIntegerField(default=0)
    bids_accepted = models.PositiveIntegerField(default=0)
    last_activity = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Stats for {self.user.full_name}"

    class Meta:
        verbose_name = "User Stats"
        verbose_name_plural = "User Stats"