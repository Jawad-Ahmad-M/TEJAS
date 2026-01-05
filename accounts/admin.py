from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, BiometricProfile, BiometricLog, UserStats

class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'first_name', 'last_name', 'company_name', 'is_staff')
    fieldsets = UserAdmin.fieldsets + (
        ('Additional Info', {'fields': ('company_name', 'phone')}),
    )
    add_fieldsets = UserAdmin.add_fieldsets + (
        ('Additional Info', {'fields': ('company_name', 'phone')}),
    )

@admin.register(BiometricProfile)
class BiometricProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'biometric_type', 'is_active', 'created_at')
    list_filter = ('biometric_type', 'is_active')
    search_fields = ('user__username', 'user__email')

@admin.register(BiometricLog)
class BiometricLogAdmin(admin.ModelAdmin):
    list_display = ('user', 'biometric_type', 'action', 'success', 'created_at')
    list_filter = ('biometric_type', 'action', 'success')
    readonly_fields = ('created_at',)

@admin.register(UserStats)
class UserStatsAdmin(admin.ModelAdmin):
    list_display = ('user', 'tenders_created', 'bids_submitted', 'bids_accepted')

admin.site.register(User, CustomUserAdmin)
