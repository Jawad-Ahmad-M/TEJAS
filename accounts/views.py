"""
Django views for user authentication and biometric management
"""

import os
import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np

from django.conf import settings
<<<<<<< HEAD
=======
from django.db import transaction, IntegrityError
>>>>>>> bd1274c (Added Chat and rafactored code)
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import login as auth_login, authenticate, login, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib.auth.hashers import make_password, check_password
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import re

from .models import User, BiometricProfile, BiometricLog
from .ml.face_utils import FaceDetector, preprocess_face, calculate_checksum, save_face_image
<<<<<<< HEAD
=======
from .ml.voice_utils import generate_voice_embedding, compare_voice_embeddings
>>>>>>> bd1274c (Added Chat and rafactored code)

# Global model placeholders
_face_detector = None
_facenet_model = None

<<<<<<< HEAD
=======
@csrf_exempt
@require_http_methods(["POST"])
def validate_registration_data(request):
    """API endpoint for real-time username/email validation"""
    try:
        data = json.loads(request.body)
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        
        if username and User.objects.filter(username=username).exists():
            return JsonResponse({'success': False, 'error': 'Username already exists'}, status=400)
            
        if email and User.objects.filter(email=email).exists():
            return JsonResponse({'success': False, 'error': 'Email already registered'}, status=400)
            
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

>>>>>>> bd1274c (Added Chat and rafactored code)
def get_face_models():
    """Lazy load ML models (Deprecated for DeepFace)"""
    return None, None


# ============================================================================
# AUTHENTICATION VIEWS
# ============================================================================

<<<<<<< HEAD
def register(request):
    """Handle user registration"""
    if request.method == "POST":
        first_name = request.POST.get("first_name", "").strip()
        last_name = request.POST.get("last_name", "").strip()
        username = request.POST.get("username", "").strip()
        email = request.POST.get("email", "").strip()
        company = request.POST.get("company", "").strip()
        phone = request.POST.get("phone", "").strip()
        password = request.POST.get("password", "")
        confirm_password = request.POST.get("confirm_password", "")

        # Check if it's an AJAX request
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

        # Validate username format
        if not re.match(r'^[a-zA-Z0-9_]{3,20}$', username):
            if is_ajax:
                return JsonResponse({'success': False, 'error': 'Username must be 3-20 characters and contain only letters, numbers, and underscores'}, status=400)
            messages.error(request, "Username must be 3-20 characters and contain only letters, numbers, and underscores")
            return render(request, "register.html")

        # Validate password length
        if len(password) < 8:
            if is_ajax:
                return JsonResponse({'success': False, 'error': 'Password must be at least 8 characters long'}, status=400)
            messages.error(request, "Password must be at least 8 characters long")
            return render(request, "register.html")

        # Validate passwords match
        if password != confirm_password:
            if is_ajax:
                return JsonResponse({'success': False, 'error': 'Passwords do not match'}, status=400)
            messages.error(request, "Passwords do not match")
            return render(request, "register.html")

        # Check if email exists
        if User.objects.filter(email=email).exists():
            if is_ajax:
                return JsonResponse({'success': False, 'error': 'Email already registered'}, status=400)
            messages.error(request, "Email already registered")
            return render(request, "register.html")
        
        # Check if username exists
        if User.objects.filter(username=username).exists():
            if is_ajax:
                return JsonResponse({'success': False, 'error': 'Username already exists'}, status=400)
            messages.error(request, "Username already exists")
            return render(request, "register.html")

        # Create user
        try:
            user = User.objects.create(
                first_name=first_name,
                last_name=last_name,
                username=username,
                email=email,
                company_name=company if company else None,
                phone=phone if phone else None,
                password=make_password(password)
            )

            # Auto-login
            auth_login(request, user, backend='django.contrib.auth.backends.ModelBackend')

            # Return JSON for AJAX requests
            if is_ajax:
                return JsonResponse({
                    'success': True,
                    'user_id': user.id,
                    'message': 'Account created successfully!'
                })

            messages.success(request, "Account created successfully!")
            return redirect("tenders:browse")
            
        except Exception as e:
            if is_ajax:
                return JsonResponse({'success': False, 'error': str(e)}, status=500)
            messages.error(request, f"An error occurred: {str(e)}")
            return render(request, "register.html")
=======
@require_http_methods(["GET", "POST"])
def register(request):
    """Handle user registration with atomic transactions for biometrics"""
    if request.method == "POST":
        try:
            # Check for JSON request (Unified Payload)
            if request.headers.get('Content-Type') == 'application/json':
                data = json.loads(request.body)
                
                # Extract Personal Info
                first_name = data.get("first_name", "").strip()
                last_name = data.get("last_name", "").strip()
                username = data.get("username", "").strip()
                email = data.get("email", "").strip()
                company = data.get("company", "").strip()
                phone = data.get("phone", "").strip()
                password = data.get("password", "")
                confirm_password = data.get("confirm_password", "")
                
                # Extract Biometrics
                face_data = data.get("face_data")
                voice_data = data.get("voice_data")

            else:
                # Fallback for old form-data (Should not be hit by new frontend)
                first_name = request.POST.get("first_name", "").strip()
                last_name = request.POST.get("last_name", "").strip()
                username = request.POST.get("username", "").strip()
                email = request.POST.get("email", "").strip()
                company = request.POST.get("company", "").strip()
                phone = request.POST.get("phone", "").strip()
                password = request.POST.get("password", "")
                confirm_password = request.POST.get("confirm_password", "")
                face_data = None
                voice_data = None

            # --- Validation ---
            if not re.match(r'^[a-zA-Z0-9_]{3,20}$', username):
                return JsonResponse({'success': False, 'error': 'Invalid username format'}, status=400)

            if len(password) < 8:
                return JsonResponse({'success': False, 'error': 'Password too short'}, status=400)

            if password != confirm_password:
                return JsonResponse({'success': False, 'error': 'Passwords do not match'}, status=400)

            if User.objects.filter(email=email).exists():
                return JsonResponse({'success': False, 'error': 'Email already registered'}, status=400)
            
            if User.objects.filter(username=username).exists():
                return JsonResponse({'success': False, 'error': 'Username already exists'}, status=400)

            # --- Atomic Creation ---
            from django.db import transaction
            from .ml.face_utils import generate_embedding as gen_face_emb
            from .ml.voice_utils import generate_voice_embedding as gen_voice_emb
            import uuid

            with transaction.atomic():
                # 1. Create User
                user = User.objects.create(
                    first_name=first_name,
                    last_name=last_name,
                    username=username,
                    email=email,
                    company_name=company if company else None,
                    phone=phone if phone else None,
                    password=make_password(password),
                    is_active=True # Active only if everything succeeds
                )
                print(f"User {user.username} created (Pending commit)")

                # 2. Process Face
                if face_data:
                    image_array = base64_to_image(face_data)
                    face_emb = gen_face_emb(image_array)
                    
                    if face_emb is None:
                        raise Exception("Face verification failed. Could not generate embedding.")

                    # Save Face Profile
                    face_dir = os.path.join(settings.MEDIA_ROOT, 'biometric', f'user_{user.id}')
                    os.makedirs(face_dir, exist_ok=True)
                    
                    face_filename = f"face_{uuid.uuid4()}.jpg"
                    face_path = os.path.join(face_dir, face_filename)
                    Image.fromarray(image_array).save(face_path)
                    
                    emb_filename = face_filename.replace('.jpg', '_embedding.npy')
                    np.save(os.path.join(face_dir, emb_filename), face_emb)
                    
                    BiometricProfile.objects.create(
                        user=user,
                        biometric_type='FACE',
                        local_path=os.path.relpath(os.path.join(face_dir, emb_filename), settings.MEDIA_ROOT),
                        is_active=True
                    )
                else:
                    raise Exception("Missing Face Data")

                # 3. Process Voice
                if voice_data:
                    # Save audio
                    if 'base64,' in voice_data:
                        voice_data = voice_data.split('base64,')[1]
                    audio_bytes = base64.b64decode(voice_data)
                    
                    voice_dir = os.path.join(settings.MEDIA_ROOT, 'biometric', f'user_{user.id}')
                    os.makedirs(voice_dir, exist_ok=True)
                    
                    voice_filename = f"voice_{uuid.uuid4()}.wav"
                    voice_path = os.path.join(voice_dir, voice_filename)
                    with open(voice_path, 'wb') as f:
                        f.write(audio_bytes)
                        
                    voice_emb = gen_voice_emb(voice_path)
                    
                    if voice_emb is None:
                         raise Exception("Voice verification failed. Could not generate embedding.")

                    # Save Voice Profile
                    emb_filename = voice_filename.replace('.wav', '_embedding.npy')
                    np.save(os.path.join(voice_dir, emb_filename), voice_emb)
                    
                    BiometricProfile.objects.create(
                        user=user,
                        biometric_type='VOICE',
                        local_path=os.path.relpath(os.path.join(voice_dir, emb_filename), settings.MEDIA_ROOT),
                        is_active=True
                    )
                else:
                     raise Exception("Missing Voice Data")

            # 4. Success - Commit happens here
            # Auto-login
            auth_login(request, user, backend='django.contrib.auth.backends.ModelBackend')
            
            return JsonResponse({
                'success': True,
                'user_id': user.id,
                'message': 'Account created successfully with full biometrics!'
            })

        except IntegrityError as e:
            # Transaction auto-rolled back
            print(f"Registration Integrity Error: {str(e)}")
            error_msg = "A registration error occurred."
            if "username" in str(e).lower():
                error_msg = "Username already exists. Please choose another."
            elif "email" in str(e).lower():
                error_msg = "Email already registered."
            return JsonResponse({'success': False, 'error': error_msg}, status=400)

        except Exception as e:
            # Transaction auto-rolled back
            print(f"Registration Failed (Rolled Back): {str(e)}")
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
>>>>>>> bd1274c (Added Chat and rafactored code)

    return render(request, "register.html")

def login_view(request):
    """Handle user login"""
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")

        # Authenticate user
        user = authenticate(request, username=username, password=password)

        if user is not None:
            auth_login(request, user)
            messages.success(request, "Logged in successfully!")
            
            # Redirect to tenders browse page with namespace
            
            # Check for unread notifications
            from tenders.models import Notification
            unread_notifications = Notification.objects.filter(user=user, is_read=False)
            
            if unread_notifications.exists():
                count = unread_notifications.count()
                messages.info(request, f"You have {count} unread notifications.")
                
                # Specifically show high-priority ones like BID updates
                priority_notifs = unread_notifications.filter(notification_type__in=['BID_ACCEPTED', 'BID_REJECTED'])
                for notif in priority_notifs:
                    if notif.notification_type == 'BID_ACCEPTED':
                        messages.success(request, f"🎉 {notif.title}: {notif.message}")
                    else:
                        messages.warning(request, f"📢 {notif.title}: {notif.message}")
                    
                    # Mark as read
                    notif.is_read = True
                    notif.save()
            
            return redirect("tenders:browse")
        else:
            messages.error(request, "Invalid username or password")
            return render(request, "login.html")

    return render(request, "login.html")


from django.contrib.auth import logout as auth_logout

def logout_view(request):
    """Log out the user and redirect to the landing page"""
    auth_logout(request)
    messages.success(request, "You have been successfully logged out.")
    return redirect('core:home')

# ============================================================================
# PROFILE VIEWS
# ============================================================================

@login_required
def profile(request):
    """Display user profile with real-time statistics"""
    # Simple class to hold stats
    class Stats:
        tenders_created = 0
        bids_submitted = 0
        bids_accepted = 0
    
    stats = Stats()
    
    # Get real counts directly from database
    try:
        from tenders.models import Tender, Bid
        stats.tenders_created = Tender.objects.filter(created_by_id=request.user.id).count()
        stats.bids_submitted = Bid.objects.filter(user_id=request.user.id).count()
        stats.bids_accepted = Bid.objects.filter(user_id=request.user.id, status='ACCEPTED').count()
    except Exception:
        pass  # Use default 0 values if tenders app not available
    
    # Calculate success rate
    success_rate = round((stats.bids_accepted / stats.bids_submitted) * 100, 1) if stats.bids_submitted > 0 else 0
    
    return render(request, 'profile.html', {
        'user': request.user,
        'stats': stats,
        'success_rate': success_rate,
    })


@login_required
def settings_view(request):
    """User settings page for updating profile, password, and biometrics"""
    if request.method == "POST":
        action = request.POST.get("action")
        
        # Profile Update
        if action == "update_profile":
            first_name = request.POST.get("first_name", "").strip()
            last_name = request.POST.get("last_name", "").strip()
            username = request.POST.get("username", "").strip()
            email = request.POST.get("email", "").strip()
            company = request.POST.get("company", "").strip()
            phone = request.POST.get("phone", "").strip()
            
            # Validate username format
            if not re.match(r'^[a-zA-Z0-9_]{3,20}$', username):
                messages.error(request, "Username must be 3-20 characters and contain only letters, numbers, and underscores")
                return render(request, "settings.html")
            
            # Check if username exists (excluding current user)
            if User.objects.filter(username=username).exclude(id=request.user.id).exists():
                messages.error(request, "Username already taken")
                return render(request, "settings.html")
            
            # Check if email exists (excluding current user)
            if User.objects.filter(email=email).exclude(id=request.user.id).exists():
                messages.error(request, "Email already registered")
                return render(request, "settings.html")
            
            try:
                user = request.user
                user.first_name = first_name
                user.last_name = last_name
                user.username = username
                user.email = email
                user.company_name = company if company else None
                user.phone = phone if phone else None
                user.save()
                
                messages.success(request, "Profile updated successfully!")
                return redirect("accounts:settings")
            except Exception as e:
                messages.error(request, f"An error occurred: {str(e)}")
        
        # Password Update
        elif action == "update_password":
            current_password = request.POST.get("current_password", "")
            new_password = request.POST.get("new_password", "")
            confirm_password = request.POST.get("confirm_password", "")
            
            # Verify current password
            if not check_password(current_password, request.user.password):
                messages.error(request, "Current password is incorrect")
                return render(request, "settings.html")
            
            # Validate new password length
            if len(new_password) < 8:
                messages.error(request, "New password must be at least 8 characters long")
                return render(request, "settings.html")
            
            # Validate passwords match
            if new_password != confirm_password:
                messages.error(request, "New passwords do not match")
                return render(request, "settings.html")
            
            try:
                user = request.user
                user.password = make_password(new_password)
                user.save()
                
                # Keep user logged in after password change
                update_session_auth_hash(request, user)
                
                messages.success(request, "Password updated successfully!")
                return redirect("accounts:settings")
            except Exception as e:
                messages.error(request, f"An error occurred: {str(e)}")
<<<<<<< HEAD
=======
        
        # Biometric Registration (Handling the button in Settings)
        elif action == "register_biometric":
            biometric_type = request.POST.get("biometric_type")
            if biometric_type == 'VOICE':
                # Since voice needs audio capture, we redirect to a specific flow or 
                # handle via JS modal in the template. For now, let's warn that JS is needed.
                messages.error(request, "Please use the 'Register Voice' button which uses your microphone.")
                return redirect("accounts:settings")
>>>>>>> bd1274c (Added Chat and rafactored code)
    
    # Get user's biometric profiles
    face_biometric = BiometricProfile.objects.filter(
        user=request.user, 
        biometric_type='FACE', 
        is_active=True
    ).first()
    
    voice_biometric = BiometricProfile.objects.filter(
        user=request.user, 
        biometric_type='VOICE', 
        is_active=True
    ).first()
    
    context = {
        'user': request.user,
        'face_biometric': face_biometric,
        'voice_biometric': voice_biometric,
    }
    
    return render(request, "settings.html", context)


@login_required
def delete_biometric(request, biometric_id):
    """Delete a biometric profile and associated files"""
    try:
        biometric = BiometricProfile.objects.get(id=biometric_id, user=request.user)
        biometric_type = biometric.biometric_type
        
        # Delete associated files
        if biometric.local_path:
            try:
                embedding_path = os.path.join(settings.MEDIA_ROOT, biometric.local_path)
<<<<<<< HEAD
                face_image_path = embedding_path.replace('_embedding.npy', '.jpg')
=======
                
                # Biometric-specific file cleanup
                if biometric_type == 'FACE':
                    face_image_path = embedding_path.replace('_embedding.npy', '.jpg')
                    if os.path.exists(face_image_path):
                        os.remove(face_image_path)
                elif biometric_type == 'VOICE':
                    voice_audio_path = embedding_path.replace('_embedding.npy', '.wav')
                    if os.path.exists(voice_audio_path):
                        os.remove(voice_audio_path)
>>>>>>> bd1274c (Added Chat and rafactored code)
                
                # Delete embedding file
                if os.path.exists(embedding_path):
                    os.remove(embedding_path)
<<<<<<< HEAD
                
                # Delete face image
                if os.path.exists(face_image_path):
                    os.remove(face_image_path)
=======
>>>>>>> bd1274c (Added Chat and rafactored code)
                    
            except Exception as e:
                print(f"Error deleting biometric files: {str(e)}")
        
        # Delete database record
        biometric.delete()
        messages.success(request, f"{biometric_type.title()} biometric deleted successfully!")
        
    except BiometricProfile.DoesNotExist:
        messages.error(request, "Biometric profile not found")
    except Exception as e:
        messages.error(request, f"An error occurred: {str(e)}")
    
    return redirect("accounts:settings")


# ============================================================================
# BIOMETRIC AUTHENTICATION VIEWS
# ============================================================================

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image"""
    # Remove data URL prefix if present
    if 'base64,' in base64_string:
        base64_string = base64_string.split('base64,')[1]
    
    # Decode base64
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return np.array(image)


@login_required
@require_http_methods(["POST"])
def register_face(request):
    """
    Register a user's face for biometric authentication
    
    Expected JSON payload:
    {
        "image": "base64_encoded_image"
    }
    """
    try:
        data = json.loads(request.body)
<<<<<<< HEAD
=======
        # print("DEBUG: register_face called")
>>>>>>> bd1274c (Added Chat and rafactored code)
        image_base64 = data.get('image')
        
        if not image_base64:
            return JsonResponse({
                'success': False,
                'error': 'Missing image'
            }, status=400)
        
        # Use logged in user
        user = request.user
        
        # Convert base64 to image
        image_array = base64_to_image(image_base64)
        
        # Get face embedding using DeepFace
        from .ml.face_utils import generate_embedding, save_face_image
<<<<<<< HEAD
        embedding = generate_embedding(image_array)
=======
        print("DEBUG: Calling generate_embedding...")
        embedding = generate_embedding(image_array)
        print(f"DEBUG: generate_embedding result: {embedding is not None}")
>>>>>>> bd1274c (Added Chat and rafactored code)
        
        if embedding is None:
            BiometricLog.objects.create(
                user=user,
                biometric_type='FACE',
                action='REGISTER',
                success=False
            )
            return JsonResponse({
                'success': False,
                'error': 'No face detected in image'
            }, status=400)
        
        # Create directories
        user_dir = os.path.join(settings.MEDIA_ROOT, 'biometric', f'user_{user.id}')
        os.makedirs(user_dir, exist_ok=True)
        
        # Save face image (DeepFace handles its own detection, but we keep a reference)
        face_image_path = os.path.join(user_dir, 'face.jpg')
        checksum = save_face_image(image_array, face_image_path)
        
        # Save embedding
        embedding_path = os.path.join(user_dir, 'face_embedding.npy')
        np.save(embedding_path, embedding)
        
        # Get relative path for database
        relative_path = os.path.relpath(embedding_path, settings.MEDIA_ROOT)
        
        # Deactivate old profiles
        BiometricProfile.objects.filter(
            user=user,
            biometric_type='FACE',
            is_active=True
        ).update(is_active=False)
        
        # Create new biometric profile
        profile = BiometricProfile.objects.create(
            user=user,
            biometric_type='FACE',
            local_path=relative_path,
            checksum=checksum,
            model_version='facenet512',
            is_active=True
        )
        
        # Log success
        BiometricLog.objects.create(
            user=user,
            biometric_type='FACE',
            action='REGISTER',
            success=True,
            confidence=1.0
        )
        
        return JsonResponse({
            'success': True,
            'message': 'Face registered successfully',
            'profile_id': profile.id
        })
        
    except Exception as e:
        import traceback
        print(f"ERROR in register_face: {str(e)}")
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def authenticate_face(request):
    """
    Authenticate a user using face recognition
    """
    try:
        import time
        start_t = time.time()
        
        data = json.loads(request.body)
        image_base64 = data.get('image')
        
        if not image_base64:
            return JsonResponse({'success': False, 'error': 'No image data'}, status=400)
            
        image_array = base64_to_image(image_base64)
        
        from .ml.face_utils import generate_embedding
        print("🔍 [Biometric] Generating fingerprint from photo...")
        new_embedding = generate_embedding(image_array)
        
        if new_embedding is None:
            return JsonResponse({'success': False, 'error': 'Face not detected in photo'}, status=400)
        
        # Search for matching user
        profiles = BiometricProfile.objects.filter(biometric_type='FACE', is_active=True).select_related('user')
        match_found = None
        
        print(f"🔍 [Biometric] Searching {profiles.count()} profiles...")
        for profile in profiles:
            try:
                emb_path = os.path.join(settings.MEDIA_ROOT, profile.local_path)
                if not os.path.exists(emb_path): continue
                
                stored_emb = np.load(emb_path)
                
                # FORCE NORMALIZATION for reliable comparison (Rescue old profiles)
                def force_norm(v):
                    n = np.linalg.norm(v)
                    return v / n if n > 0 else v
                
                vec1 = force_norm(new_embedding)
                vec2 = force_norm(stored_emb)
                
                distance = np.linalg.norm(vec1 - vec2)
                
                # DIAGNOSTICS
                print(f"📊 [Biometric] Comparison with {profile.user.username}:")
                # print(f"   - Distance: {distance:.4f} (Threshold: 0.75)")
                
                if distance <= 0.80: # Relaxed threshold for better UX
                    print(f"✅ [Biometric] Perfect Match! distance={distance:.4f}")
                    match_found = profile
                    print(f"✅ [Biometric] Match found: {profile.user.username}")
                    break
            except Exception: continue
        
        if match_found:
            login(request, match_found.user, backend='django.contrib.auth.backends.ModelBackend')
            print(f"✅ [Biometric] Authentication successful in {time.time() - start_t:.2f}s")
            return JsonResponse({
                'success': True,
                'user': {'username': match_found.user.username, 'id': match_found.user.id}
            })
        
        print(f"❌ [Biometric] No match found in {time.time() - start_t:.2f}s")
        return JsonResponse({'success': False, 'error': 'Identity not recognized'}, status=401)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def verify_face(request):
    """
    Verify if a face matches a specific user
    
    Expected JSON payload:
    {
        "user_id": 123,
        "image": "base64_encoded_image"
    }
    """
    try:
        data = json.loads(request.body)
        user_id = data.get('user_id')
        image_base64 = data.get('image')
        
        if not user_id or not image_base64:
            return JsonResponse({
                'success': False,
                'error': 'Missing user_id or image'
            }, status=400)
        
        # Get user and profile
        try:
            user = User.objects.get(id=user_id)
            profile = BiometricProfile.objects.get(
                user=user,
                biometric_type='FACE',
                is_active=True
            )
        except User.DoesNotExist:
            return JsonResponse({
                'success': False,
                'error': 'User not found'
            }, status=404)
        except BiometricProfile.DoesNotExist:
            return JsonResponse({
                'success': False,
                'error': 'No face profile found for this user'
            }, status=404)
        
        # Get models
        face_detector_instance, facenet_model_instance = get_face_models()
        
        # Convert base64 to image
        image_array = base64_to_image(image_base64)
        
        # Detect and extract face
        face_pixels = face_detector_instance.extract_face(image_array)
        
        if face_pixels is None:
            BiometricLog.objects.create(
                user=user,
                biometric_type='FACE',
                action='VERIFY',
                success=False
            )
            return JsonResponse({
                'success': False,
                'error': 'No face detected in image'
            }, status=400)
        
        # Preprocess face
        face_preprocessed = preprocess_face(face_pixels)
        
        # Get face embedding
        new_embedding = facenet_model_instance.get_embedding(face_preprocessed)
        
        # Load stored embedding
        embedding_path = os.path.join(settings.MEDIA_ROOT, profile.local_path)
        
        # Verify checksum
        if profile.checksum:
            face_image_path = embedding_path.replace('_embedding.npy', '.jpg')
            if os.path.exists(face_image_path):
                current_checksum = calculate_checksum(face_image_path)
                if current_checksum != profile.checksum:
                    return JsonResponse({
                        'success': False,
                        'error': 'Biometric data integrity check failed'
                    }, status=400)
        
        stored_embedding = facenet_model_instance.load_embedding(embedding_path)
        
        # Compare embeddings
        is_match, distance, confidence = facenet_model_instance.compare_embeddings(
            new_embedding,
            stored_embedding,
            threshold=0.6
        )
        
        # Log the verification
        BiometricLog.objects.create(
            user=user,
            biometric_type='FACE',
            action='VERIFY',
            success=is_match,
            confidence=confidence
        )
        
        return JsonResponse({
            'success': True,
            'is_match': is_match,
            'confidence': float(confidence),
            'distance': float(distance)
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
<<<<<<< HEAD
        }, status=500)
=======
        }, status=500)


@login_required
@require_http_methods(["GET"])
def current_user(request):
    """
    Get current logged in user details
    """
    return JsonResponse({
        'user_id': request.user.id,
        'username': request.user.username,
        'email': request.user.email
    })


@csrf_exempt
@require_http_methods(["POST"])
def register_voice(request):
    """
    Register a user's voice for biometric authentication
    """
    try:
        data = json.loads(request.body)
        user_id = data.get('user_id')
        audio_base64 = data.get('audio')
        
        if not user_id or not audio_base64:
            return JsonResponse({'success': False, 'error': 'Missing user_id or audio'}, status=400)
            
        # Get user
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'User not found'}, status=404)
            
        # Save audio to temp file
        import uuid
        if 'base64,' in audio_base64:
            audio_base64 = audio_base64.split('base64,')[1]
            
        audio_data = base64.b64decode(audio_base64)
        
        # Create user directory for biometrics
        user_dir = os.path.join(settings.MEDIA_ROOT, 'biometric', f'user_{user.id}')
        os.makedirs(user_dir, exist_ok=True)
        
        # Save original audio file acting as the "template"
        filename = f"voice_template_{uuid.uuid4()}.wav"
        audio_path = os.path.join(user_dir, filename)
        
        with open(audio_path, 'wb') as f:
            f.write(audio_data)
            
        # Generate embedding
        print(f"🎙️ Generating voice embedding for {user.username}...")
        embedding = generate_voice_embedding(audio_path)
        
        if embedding is None:
            # Clean up
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
            BiometricLog.objects.create(
                user=user,
                biometric_type='VOICE',
                action='REGISTER',
                success=False
            )
            return JsonResponse({'success': False, 'error': 'Could not analyze voice. Please try again.'}, status=400)
            
        # Save embedding
        emb_filename = filename.replace('.wav', '_embedding.npy')
        emb_path = os.path.join(user_dir, emb_filename)
        np.save(emb_path, embedding)
        
        # Deactivate old voice profiles
        BiometricProfile.objects.filter(
            user=user,
            biometric_type='VOICE',
            is_active=True
        ).update(is_active=False)
        
        # Create new profile
        relative_path = os.path.relpath(emb_path, settings.MEDIA_ROOT)
        
        profile = BiometricProfile.objects.create(
            user=user,
            biometric_type='VOICE',
            local_path=relative_path,
            model_version='speechbrain/ecapa-voxceleb',
            is_active=True
        )
        
        BiometricLog.objects.create(
            user=user,
            biometric_type='VOICE',
            action='REGISTER',
            success=True,
            confidence=1.0
        )
        
        return JsonResponse({
            'success': True,
            'message': 'Voice registered successfully',
            'profile_id': profile.id
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def authenticate_voice(request):
    """
    Authenticate a user using voice recognition
    """
    try:
        import time
        
        data = json.loads(request.body)
        audio_base64 = data.get('audio')
        
        if not audio_base64:
            return JsonResponse({'success': False, 'error': 'No audio data'}, status=400)
            
        # Save audio to temp file
        import uuid
        if 'base64,' in audio_base64:
            audio_base64 = audio_base64.split('base64,')[1]
            
        audio_data = base64.b64decode(audio_base64)
        
        filename = f"voice_auth_{uuid.uuid4()}.wav"
        audio_path = os.path.join(settings.MEDIA_ROOT, 'temp', filename)
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        
        with open(audio_path, 'wb') as f:
            f.write(audio_data)
        
        print(f"🔍 [Biometric] Generating voice embedding...")
        new_embedding = generate_voice_embedding(audio_path)
        
        # Cleanup temp file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        if new_embedding is None:
            return JsonResponse({'success': False, 'error': 'Voice not detected/analyzed'}, status=400)
        
        # Search for matching user
        profiles = BiometricProfile.objects.filter(biometric_type='VOICE', is_active=True).select_related('user')
        match_found = None
        
        print(f"🔍 [Biometric] Searching {profiles.count()} voice profiles...")
        for profile in profiles:
            try:
                emb_path = os.path.join(settings.MEDIA_ROOT, profile.local_path)
                if not os.path.exists(emb_path): continue
                
                stored_emb = np.load(emb_path)
                
                is_match, score = compare_voice_embeddings(new_embedding, stored_emb)
                
                print(f"📊 [Biometric] Voice compare with {profile.user.username}: Score={score:.4f}")
                
                if is_match:
                    match_found = profile
                    print(f"✅ [Biometric] Voice match found: {profile.user.username}")
                    break
            except Exception: continue
        
        if match_found:
            login(request, match_found.user, backend='django.contrib.auth.backends.ModelBackend')
            return JsonResponse({
                'success': True,
                'user': {'username': match_found.user.username, 'id': match_found.user.id}
            })
        
        return JsonResponse({'success': False, 'error': 'Voice identity not recognized'}, status=401)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@login_required
def dashboard_view(request):
    """
    Data Analysis Dashboard View.
    Uses pandas/matplotlib/plotly to generate insights.
    """
    try:
        from .analytics import generate_analytics
        context = generate_analytics()
        context['user'] = request.user
        return render(request, 'dashboard.html', context)
    except Exception as e:
        import traceback
        traceback.print_exc()
        messages.error(request, f"Error generating analytics: {str(e)}")
        return redirect('accounts:profile')
>>>>>>> bd1274c (Added Chat and rafactored code)
