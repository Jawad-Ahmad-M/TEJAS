from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.db.models import Q, Count
from django.utils import timezone
from datetime import timedelta
<<<<<<< HEAD
from .models import Tender, TenderCategory, Bid, BidFile
=======
from .models import Tender, TenderCategory, Bid, BidFile, Notification
>>>>>>> bd1274c (Added Chat and rafactored code)
from django.contrib import messages
from django.contrib.auth import get_user_model

# Get the User model
User = get_user_model()


def browse(request):
    return render(request, "browse.html")


def browse_tenders(request):
    """
    Display tenders with filtering, sorting, and search functionality.
    Shows only the latest 8 tenders based on filters.
    """
    # Start with all open tenders
    tenders = Tender.objects.all().select_related('category', 'created_by')
    
    # Get filter parameters
    category = request.GET.get('category', '')
    sort_by = request.GET.get('sort', 'deadline')
    status_filter = request.GET.get('status', '')
    search_query = request.GET.get('search', '')
    
    # Apply category filter
    if category:
        tenders = tenders.filter(category__name__iexact=category)
    
    # Apply search filter
    if search_query:
        tenders = tenders.filter(
            Q(title__icontains=search_query) |
            Q(description__icontains=search_query) |
            Q(organization_name__icontains=search_query) |
            Q(id__icontains=search_query)
        )
    
    # Apply status filter (open or closing soon)
    today = timezone.now().date()
    if status_filter == 'open':
        # Tenders with deadline more than 7 days away
        closing_soon_date = today + timedelta(days=7)
        tenders = tenders.filter(submission_deadline__gt=closing_soon_date)
    elif status_filter == 'closing-soon':
        # Tenders with deadline within 7 days
        closing_soon_date = today + timedelta(days=7)
        tenders = tenders.filter(submission_deadline__lte=closing_soon_date, submission_deadline__gte=today)
    
    # Apply sorting
    if sort_by == 'deadline':
        tenders = tenders.order_by('submission_deadline')
    elif sort_by == 'budget-high':
        tenders = tenders.order_by('-budget_max')
    elif sort_by == 'budget-low':
        tenders = tenders.order_by('budget_min')
    elif sort_by == 'newest':
        tenders = tenders.order_by('-created_at')
    elif sort_by == 'oldest':
        tenders = tenders.order_by('created_at')
    else:
        tenders = tenders.order_by('submission_deadline')
    
    # Get only the latest 8 tenders
    tenders = tenders[:8]
    
    # Get user's bookmarks if authenticated
    bookmarked_tender_ids = []
    if request.user.is_authenticated:
        from .models import Bookmark
        bookmarked_tender_ids = Bookmark.objects.filter(user=request.user).values_list('tender_id', flat=True)

    # Add status calculation for each tender
    tender_list = []
    for tender in tenders:
        days_until_deadline = (tender.submission_deadline - today).days
        
        # Priority 1: Check Database Status
        if tender.status in ['CLOSED', 'EVALUATED']:
            tender_status = 'closed'
        # Priority 2: Check Date
        elif days_until_deadline < 0:
            tender_status = 'closed'
        elif days_until_deadline <= 7:
            tender_status = 'closing-soon'
        else:
            tender_status = 'open'
        
        tender_list.append({
            'tender': tender,
            'status': tender_status,
            'days_until_deadline': days_until_deadline,
            'is_bookmarked': tender.id in bookmarked_tender_ids
        })
    
    # Get all categories for filter dropdown
    categories = TenderCategory.objects.all()
    
    context = {
        'tenders': tender_list,
        'categories': categories,
        'selected_category': category,
        'selected_sort': sort_by,
        'selected_status': status_filter,
        'search_query': search_query,
        'total_count': len(tender_list)
    }
    
    return render(request, 'tenders/browse.html', context)


@login_required
def tender_details(request, tender_id):
    """
    Display detailed information about a specific tender.
    """
    tender = get_object_or_404(Tender, id=tender_id)
    
    # Calculate status
    today = timezone.now().date()
    days_until_deadline = (tender.submission_deadline - today).days

    if tender.status in ['CLOSED', 'EVALUATED']:
        tender_status = 'closed'
    elif days_until_deadline < 0:
        tender_status = 'closed'
    elif days_until_deadline <= 7:
        tender_status = 'closing-soon'
    else:
        tender_status = 'open'
    
    # Get bid count
    bid_count = tender.bids.count()
    
    # Check if user has already bid
    user_has_bid = False
    if request.user.is_authenticated:
        user_has_bid = tender.bids.filter(user=request.user).exists()
    
    # Get Top 5 Bids for public visibility (lowest amount first)
    top_bids = tender.bids.filter(status__in=['VALID', 'PENDING', 'ACCEPTED']).select_related('user').order_by('bid_amount')[:5]

    # Check bookmark status
    is_bookmarked = False
    if request.user.is_authenticated:
        from .models import Bookmark
        is_bookmarked = Bookmark.objects.filter(user=request.user, tender=tender).exists()

    context = {
        'tender': tender,
        'status': tender_status,
        'days_until_deadline': days_until_deadline,
        'bid_count': bid_count,
        'user_has_bid': user_has_bid,
        'top_bids': top_bids,
        'is_creator': request.user == tender.created_by,
        'is_bookmarked': is_bookmarked
    }
    
    return render(request, 'tenders/tender_details.html', context)


@login_required
def submit_bid(request, tender_id):
    """
    Handle bid submission for a tender.
    """
    tender = get_object_or_404(Tender, id=tender_id)
    
    # Check if tender is still open
    if tender.status in ['CLOSED', 'EVALUATED'] or tender.submission_deadline < timezone.now().date():
        messages.error(request, 'This tender is no longer accepting bids.')
        return redirect('tenders:tender_details', tender_id=tender_id)
    
    # Check if user has already submitted a bid
    if tender.bids.filter(user=request.user).exists():
        messages.warning(request, 'You have already submitted a bid for this tender.')
        return redirect('tenders:tender_details', tender_id=tender_id)
    
    # If creator tries to bid, block them
    if tender.created_by == request.user:
        messages.error(request, 'You cannot bid on your own tender.')
        return redirect('tenders:tender_details', tender_id=tender_id)

    if request.method == 'POST':
        bid_amount = request.POST.get('bid_amount')
        bid_text = request.POST.get('bid_text', '')
        
        # Validate bid amount
        try:
            bid_amount = float(bid_amount)
            if bid_amount < float(tender.budget_min) or bid_amount > float(tender.budget_max):
                messages.error(request, f'Bid amount must be between ${tender.budget_min} and ${tender.budget_max}')
                return render(request, 'tenders/place_bid.html', {'tender': tender})
        except (ValueError, TypeError):
            messages.error(request, 'Invalid bid amount')
            return render(request, 'tenders/place_bid.html', {'tender': tender})
<<<<<<< HEAD
=======

        # --- Word Count Logic ---
        full_text = bid_text
        
        # Extract text from files if provided
        if 'bid_files' in request.FILES:
            from .ml.text_extractor import extract_text_from_file
            extracted_parts = []
            for file in request.FILES.getlist('bid_files'):
                try:
                    extracted_parts.append(extract_text_from_file(file))
                except Exception as e:
                    print(f"Extraction error for {file.name}: {e}")
            
            if extracted_parts:
                full_text += "\n" + "\n".join(extracted_parts)
        
        # Count words using regex
        import re
        word_count = len(re.findall(r'\b\w+\b', full_text))
        
        # Apply the ONLY restriction: 150 word limit
        if word_count > 150:
            messages.error(
                request, 
                f'❌ Bid Rejected: Your proposal contains {word_count} words. '
                f'It should not contain more than 150 words.'
            )
            return render(request, 'tenders/place_bid.html', {
                'tender': tender,
                'bid_amount': bid_amount,
                'bid_text': bid_text
            })
>>>>>>> bd1274c (Added Chat and rafactored code)
        
        # Create bid
        bid = Bid.objects.create(
            tender=tender,
            user=request.user,
            bid_amount=bid_amount,
<<<<<<< HEAD
            bid_text=bid_text,
            status='PENDING'
        )
        
        # Handle file uploads if any
        
        # Handle file uploads if any
        if 'bid_files' in request.FILES:
            from .ml.text_extractor import extract_text_from_file
            
            extracted_text_content = []
            
            for file in request.FILES.getlist('bid_files'):
                try:
                    # Extract text from file directly (in-memory)
                    file_text = extract_text_from_file(file)
                    extracted_text_content.append(f"--- Content from {file.name} ---\n{file_text}\n")
                except Exception as e:
                    # Log error but continue (or could show message)
                    print(f"Error extracting text from {file.name}: {e}")
            
            # Append extracted text to bid_text
            if extracted_text_content:
                full_extracted_text = "\n".join(extracted_text_content)
                if bid.bid_text:
                    bid.bid_text = f"{bid.bid_text}\n\n{full_extracted_text}"
                else:
                    bid.bid_text = full_extracted_text
                
                bid.save()
                # Files are implicitly discarded as they are not saved to disk/model
        
=======
            bid_text=full_text,
            status='PENDING'
        )
        
>>>>>>> bd1274c (Added Chat and rafactored code)
        messages.success(request, 'Your bid has been submitted successfully!')
        return redirect('tenders:my_bids')
    
    return render(request, 'tenders/place_bid.html', {'tender': tender})


@login_required
def my_bids(request):
    """
    Display all bids submitted by the current user.
    """
    # Use the pk/id to ensure we're querying with the actual User instance
    bids = Bid.objects.filter(user_id=request.user.id).select_related('tender', 'user').order_by('-submitted_at')
    
    context = {
        'bids': bids
    }
    
    return render(request, 'tenders/my_bids.html', context)


@login_required
def toggle_bookmark(request, tender_id):
    """
    Toggle bookmark status for a tender. Supports AJAX.
    """
    tender = get_object_or_404(Tender, id=tender_id)
    
    # Check if bookmark exists
    from .models import Bookmark
    from django.http import JsonResponse
    
    bookmark, created = Bookmark.objects.get_or_create(user=request.user, tender=tender)
    is_bookmarked = True
    
    if not created:
        # If it existed, 'get_or_create' returns False for 'created', so we delete it
        bookmark.delete()
        is_bookmarked = False
        message = 'Bookmark removed.'
    else:
        message = 'Tender bookmarked successfully!'
    
    # Check for AJAX request
    if request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.accepts('application/json'):
        return JsonResponse({
            'success': True,
            'is_bookmarked': is_bookmarked,
            'message': message
        })
    
    messages.success(request, message)
    # Redirect back to where the user came from, or browse
    next_url = request.META.get('HTTP_REFERER', 'tenders:browse')
    return redirect(next_url)


@login_required
def my_tenders(request):
    """
    Display all tenders created by the current user with their bids.
    """
    # Use user_id to avoid the string conversion issue
    tenders = Tender.objects.filter(created_by_id=request.user.id).select_related('category', 'created_by').prefetch_related('bids__user').order_by('-created_at')
    
    # Prepare tender data with bid information
    tender_list = []
    today = timezone.now().date()
    
    for tender in tenders:
        # Calculate status
        days_until_deadline = (tender.submission_deadline - today).days
        
        if tender.status in ['CLOSED', 'EVALUATED']:
            tender_status = 'closed'
        elif days_until_deadline < 0:
            tender_status = 'closed'
        elif days_until_deadline <= 7:
            tender_status = 'closing-soon'
        else:
            tender_status = 'open'
        
        # Get all bids for this tender
        bids = tender.bids.all().order_by('-submitted_at')
        bid_count = bids.count()
        
        # Calculate bid statistics
        if bid_count > 0:
            bid_amounts = [float(bid.bid_amount) for bid in bids]
            avg_bid = sum(bid_amounts) / len(bid_amounts)
            min_bid = min(bid_amounts)
            max_bid = max(bid_amounts)
        else:
            avg_bid = None
            min_bid = None
            max_bid = None
        
        tender_list.append({
            'tender': tender,
            'status': tender_status,
            'days_until_deadline': days_until_deadline,
            'bids': bids,
            'bid_count': bid_count,
            'avg_bid': avg_bid,
            'min_bid': min_bid,
            'max_bid': max_bid,
        })
    
<<<<<<< HEAD
    context = {
        'tenders': tender_list,
        'total_count': len(tender_list)
=======
    closing_soon_count = sum(1 for t in tender_list if t['status'] == 'closing-soon')
    
    context = {
        'tenders': tender_list,
        'total_count': len(tender_list),
        'closing_soon_count': closing_soon_count
>>>>>>> bd1274c (Added Chat and rafactored code)
    }
    
    return render(request, 'tenders/my_tenders.html', context)


@login_required
def tender_bids(request, tender_id):
    """
    Display all bids for a specific tender (only accessible by tender creator).
    """
    tender = get_object_or_404(Tender, id=tender_id)
    
    # Ensure only the tender creator can view bids
    # Use id comparison to avoid type issues
    if tender.created_by_id != request.user.id:
        messages.error(request, 'You do not have permission to view bids for this tender.')
        return redirect('tenders:my_tenders')
    
    # Get all bids for this tender
    bids = tender.bids.all().select_related('user').order_by('-submitted_at')
    
    # Calculate status
    today = timezone.now().date()
    days_until_deadline = (tender.submission_deadline - today).days
    
    if tender.status in ['CLOSED', 'EVALUATED']:
        tender_status = 'closed'
    elif days_until_deadline < 0:
        tender_status = 'closed'
    elif days_until_deadline <= 7:
        tender_status = 'closing-soon'
    else:
        tender_status = 'open'
    
    # Calculate bid statistics for THIS tender
    bid_count = bids.count()
    if bid_count > 0:
        bid_amounts = [float(bid.bid_amount) for bid in bids]
        avg_bid = sum(bid_amounts) / len(bid_amounts)
        min_bid = min(bid_amounts)
        max_bid = max(bid_amounts)
    else:
        avg_bid = None
        min_bid = None
        max_bid = None
    
    # Calculate TOTAL bids received across ALL user's tenders
    total_bids_received = Bid.objects.filter(tender__created_by_id=request.user.id).count()
    
    context = {
        'tender': tender,
        'status': tender_status,
        'days_until_deadline': days_until_deadline,
        'bids': bids,
        'bid_count': bid_count,
        'avg_bid': avg_bid,
        'min_bid': min_bid,
        'max_bid': max_bid,
        'total_bids_received': total_bids_received,  # NEW: Total across all tenders
    }
    
    return render(request, 'tenders/tender_bids.html', context)


@login_required
def create_tender(request):
    """
    Create a new tender via file upload or manual form entry.
    Integrates ML anomaly detection.
    """
    categories = TenderCategory.objects.all()
    anomaly_result = None
    
    if request.method == 'POST':
        submission_type = request.POST.get('submission_type', 'form')
        
        # Prepare tender data dict for ML evaluation
        tender_data = {}
        
        if submission_type == 'upload':
            # Handle file upload
            tender_file = request.FILES.get('tender_file')
            estimated_value = request.POST.get('estimated_value_upload', '')
            
            if not tender_file:
                messages.error(request, 'Please upload a tender document.')
                return render(request, 'tenders/create_tender.html', {'categories': categories})
            
            try:
                # Extract text from file
                from .ml.text_extractor import extract_text_from_file, parse_text_to_columns
                
                raw_text = extract_text_from_file(tender_file)
                tender_data = parse_text_to_columns(raw_text)
                
                # Override estimated value from form (always required)
                if estimated_value:
                    tender_data['Estimated_Value'] = estimated_value
                
            except Exception as e:
                messages.error(request, f'Error processing file: {str(e)}')
                return render(request, 'tenders/create_tender.html', {'categories': categories})
        
        else:
            # Handle form submission
            tender_data = {
                'Title': request.POST.get('title', ''),
                'Authority': request.POST.get('authority', ''),
                'Object_Description': request.POST.get('object_description', ''),
                'CPV': request.POST.get('cpv', ''),
                'Estimated_Value': request.POST.get('estimated_value', ''),
                'Award_Criteria': request.POST.get('award_criteria', ''),
                'Conditions': request.POST.get('conditions', ''),
            }
<<<<<<< HEAD
        
        # Run ML evaluation
        try:
            from .ml.evaluator import TenderAnomalyEvaluator
            
            evaluator = TenderAnomalyEvaluator()
=======
            
        # --- NEW: Strict Input Validation Layer ---
        required_fields = {
            'Title': 'Title',
            'Authority': 'Authority',
            'Object_Description': 'Object Description',
            'CPV': 'CPV Code',
            'Estimated_Value': 'Estimated Value',
            'Award_Criteria': 'Award Criteria',
            'Conditions': 'Conditions'
        }
        
        missing_fields = [label for key, label in required_fields.items() if not tender_data.get(key)]
        
        if missing_fields:
            # Create a persistent notification ONLY for file uploads if parsing failed to find fields
            if submission_type == 'upload':
                create_notification(
                    user=request.user,
                    title="⚠️ Document Parsing Incomplete",
                    message=(
                        f"The uploaded file is missing or couldn't be parsed for: {', '.join(missing_fields)}. "
                        f"Please use manual entry or upload a more detailed document."
                    ),
                    type='SYSTEM'
                )

            messages.error(
                request, 
                f"❌ Evaluation Skipped: Please fill in all required fields: {', '.join(missing_fields)}."
            )
            return render(request, 'tenders/create_tender.html', {
                'categories': categories,
                'tender_data_raw': tender_data, # Pass back to pre-fill form
                'submission_type': submission_type
            })
        # ------------------------------------------
        
        # Run ML evaluation using pre-loaded global instance
        try:
            from django.apps import apps
            from .ml.evaluator import TenderAnomalyEvaluator
            
            # Use the instance pre-loaded at server startup
            evaluator = apps.get_app_config('tenders').evaluator
            
            # Fallback if not loaded
            if evaluator is None:
                evaluator = TenderAnomalyEvaluator(eager_load=True)
                
>>>>>>> bd1274c (Added Chat and rafactored code)
            anomaly_result = evaluator.evaluate(tender_data)
            
        except Exception as e:
            anomaly_result = {
                'anomaly_score': None,
                'is_anomaly': None,
                'category': 'ERROR',
                'explanation': f'ML evaluation error: {str(e)}'
            }
        
<<<<<<< HEAD
        # Check if tender should be blocked due to high anomaly
        anomaly_category = anomaly_result.get('category', 'NORMAL')
        is_blocked = anomaly_category in ['HIGH', 'EXTREME']
        
        if is_blocked:
=======
        # Check if tender should be blocked due to anomaly
        # Refined Blocking Logic: 
        # Pass NORMAL and LOW. 
        # Block MEDIUM, HIGH, and EXTREME if model classifies as Anomaly.
        anomaly_category = anomaly_result.get('category', 'NORMAL')
        is_anomaly = anomaly_result.get('is_anomaly', False)
        
        is_blocked = False
        if anomaly_category in ['HIGH', 'EXTREME', 'ERROR']:
            if is_anomaly or anomaly_category == 'ERROR':
                is_blocked = True
        
        if is_blocked:
            # Create a persistent notification ONLY for file uploads
            if submission_type == 'upload':
                create_notification(
                    user=request.user,
                    title="⚠️ Tender Document Flagged",
                    message=(
                        f"Your uploaded tender document was blocked. "
                        f"Reason: {anomaly_result.get('explanation')}. "
                        f"Please ensure the document contains realistic tender data."
                    ),
                    type='SYSTEM'
                )

>>>>>>> bd1274c (Added Chat and rafactored code)
            # Don't create the tender - it's too anomalous
            messages.error(
                request, 
                f'⚠️ Tender REJECTED: Your tender has been flagged as {anomaly_category} RISK. '
<<<<<<< HEAD
                f'This submission has suspicious patterns and cannot be created. '
                f'Please review and correct your tender details.'
=======
                f'This submission has suspicious patterns and cannot be created.'
>>>>>>> bd1274c (Added Chat and rafactored code)
            )
            context = {
                'categories': categories,
                'anomaly_result': anomaly_result,
                'tender_blocked': True,
            }
            return render(request, 'tenders/create_tender.html', context)
        
        # Create tender in database (only for NORMAL, LOW, MEDIUM risk)
        try:
            # Parse form values for database
            budget_min = request.POST.get('budget_min', '0') or '0'
            budget_max = request.POST.get('budget_max', '0') or '0'
            
            # Get category
            category_id = request.POST.get('category')
            category = TenderCategory.objects.filter(id=category_id).first() if category_id else None
            
            # Parse deadline
            deadline_str = request.POST.get('submission_deadline')
            if deadline_str:
                from datetime import datetime
                submission_deadline = datetime.strptime(deadline_str, '%Y-%m-%d').date()
            else:
                # Default to 30 days from now
                submission_deadline = timezone.now().date() + timedelta(days=30)
            
            # Create the tender
            tender = Tender.objects.create(
                title=tender_data.get('Title', 'Untitled Tender')[:200],
                description=tender_data.get('Object_Description', '')[:5000],
                organization_name=request.POST.get('organization_name', '') or tender_data.get('Authority', '')[:255],
                category=category,
                budget_min=float(budget_min) if budget_min else 0,
                budget_max=float(budget_max) if budget_max else 0,
                submission_deadline=submission_deadline,
                created_by=request.user,
                status='OPEN',
                # Store all extracted data in JSON field
                extracted_layout={
                    'Title': tender_data.get('Title', ''),
                    'Authority': tender_data.get('Authority', ''),
                    'CPV': tender_data.get('CPV', ''),
                    'Estimated_Value': tender_data.get('Estimated_Value', ''),
                    'Award_Criteria': tender_data.get('Award_Criteria', ''),
                    'Conditions': tender_data.get('Conditions', ''),
                    'anomaly_result': anomaly_result,
                }
            )
            
            messages.success(request, f'✅ Tender created successfully! Risk level: {anomaly_result.get("category", "N/A")}')
            
        except Exception as e:
            messages.error(request, f'Error creating tender: {str(e)}')
    
    context = {
        'categories': categories,
        'anomaly_result': anomaly_result,
    }
    
    return render(request, 'tenders/create_tender.html', context)


def create_notification(user, title, message, type='SYSTEM', link=None):
    """Helper to create a notification"""
    from .models import Notification
    Notification.objects.create(
        user=user,
        title=title,
        message=message,
        notification_type=type,
        related_link=link
    )

@login_required
def accept_bid(request, bid_id):
    """
    Accept a specific bid and close the tender.
    """
    bid = get_object_or_404(Bid, id=bid_id)
    tender = bid.tender
    
    # Check permission
    if tender.created_by != request.user:
        messages.error(request, "You do not have permission to manage this tender.")
        return redirect('tenders:my_tenders')
        
<<<<<<< HEAD
    if tender.status != 'OPEN':
=======
    if tender.status.upper() != 'OPEN':
>>>>>>> bd1274c (Added Chat and rafactored code)
        messages.error(request, "This tender is not open for bid acceptance.")
        return redirect('tenders:tender_bids', tender_id=tender.id)

    # 1. Mark this bid as ACCEPTED
    bid.status = 'ACCEPTED'
    bid.save()
    
    # Notify the winning bidder
    create_notification(
        user=bid.user,
        title="🎉 Bid Accepted!",
        message=f"Congratulations! Your bid for '{tender.title}' has been ACCEPTED.",
        type='BID_ACCEPTED',
        link=f"/tenders/{tender.id}/"
    )
    
    # 2. Reject all OTHER bids for this tender
    other_bids = tender.bids.exclude(id=bid.id)
    
    # Notify rejected bidders
    for other_bid in other_bids:
        if other_bid.status != 'REJECTED':  # Only notify if not already rejected
             create_notification(
                user=other_bid.user,
                title="Bid Update",
                message=f"The tender '{tender.title}' has been awarded to another bidder.",
                type='BID_REJECTED',
                link=f"/tenders/{tender.id}/"
            )

    # Don't update REJECTED ones if they're already rejected, just pending/valid/invalid
    other_bids.update(status='REJECTED')

    # 3. Close the tender
    tender.status = 'CLOSED'
    tender.save()
    
    messages.success(request, f"Bid from {bid.user.company_name or bid.user.username} accepted! Tender is now closed.")
    return redirect('tenders:tender_bids', tender_id=tender.id)


@login_required
def reject_bid(request, bid_id):
    """
    Reject a specific bid.
    """
    bid = get_object_or_404(Bid, id=bid_id)
    tender = bid.tender
    
    if tender.created_by != request.user:
        messages.error(request, "Permission denied.")
        return redirect('tenders:my_tenders')
        
    bid.status = 'REJECTED'
    bid.save()
    
    # Notify the bidder
    create_notification(
        user=bid.user,
        title="Bid Rejected",
        message=f"Your bid for '{tender.title}' has been rejected by the tender authority.",
        type='BID_REJECTED',
        link=f"/tenders/{tender.id}/"
    )
    
    messages.success(request, "Bid marked as rejected.")
    return redirect('tenders:tender_bids', tender_id=tender.id)
<<<<<<< HEAD
=======

@login_required
def notifications(request):
    """Display user notifications"""
    user_notifications = Notification.objects.filter(user=request.user).order_by('-created_at')
    
    # Optional: Mark all as read when visiting the page
    # user_notifications.filter(is_read=False).update(is_read=True)
    
    return render(request, 'tenders/notifications.html', {
        'notifications': user_notifications
    })
>>>>>>> bd1274c (Added Chat and rafactored code)
