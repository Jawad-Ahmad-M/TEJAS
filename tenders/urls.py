from django.urls import path
from . import views

app_name = "tenders"

urlpatterns = [
    # Browse tenders page with filtering and search
    path('browse/', views.browse_tenders, name='browse'),
    
    # Tender details page
    path('<int:tender_id>/', views.tender_details, name='tender_details'),
    
    # Submit bid for a tender
    path('<int:tender_id>/bid/', views.submit_bid, name='submit_bid'),
    
    # View user's submitted bids
    path('my-bids/', views.my_bids, name='my_bids'),
    
    # View user's created tenders
    path('my-tenders/', views.my_tenders, name='my_tenders'),
    
    # View bids for a specific tender
    path('<int:tender_id>/bids/', views.tender_bids, name='tender_bids'),
    
    # Toggle bookmark (AJAX endpoint)
    path('<int:tender_id>/bookmark/', views.toggle_bookmark, name='toggle_bookmark'),
    
    # Create new tender with ML anomaly detection
    path('create/', views.create_tender, name='create_tender'),

<<<<<<< HEAD
=======
    # Notifications
    path('notifications/', views.notifications, name='notifications'),

>>>>>>> bd1274c (Added Chat and rafactored code)
    # Accept/Reject Bids
    path('bid/<int:bid_id>/accept/', views.accept_bid, name='accept_bid'),
    path('bid/<int:bid_id>/reject/', views.reject_bid, name='reject_bid'),
]