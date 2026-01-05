from django.apps import AppConfig


class AccountsConfig(AppConfig):
    name = 'accounts'

    def ready(self):
        """Initialize biometrics on startup in a background thread"""
        import os
        import threading
        # Only run in the main process (skip the reloader)
        if os.environ.get('RUN_MAIN') == 'true':
            try:
                from .ml.face_utils import FaceDetector
<<<<<<< HEAD
                # Move to background thread to prevent hanging the server start
                init_thread = threading.Thread(target=FaceDetector().warm_up, daemon=True)
=======
                from .ml.voice_utils import VoiceRecognizer
                
                def warm_up_biometrics():
                    print("🚀 Warming up biometrics...")
                    FaceDetector().warm_up()
                    # Also warm up Speaker Recognition
                    VoiceRecognizer.get_instance()
                    print("✅ Biometrics warmed up and ready!")
                
                # Move to background thread to prevent hanging the server start
                init_thread = threading.Thread(target=warm_up_biometrics, daemon=True)
>>>>>>> bd1274c (Added Chat and rafactored code)
                init_thread.start()
            except Exception as e:
                print(f"Warning: Failed to start biometric pre-loader: {str(e)}")
