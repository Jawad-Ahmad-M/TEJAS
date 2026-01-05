from django.apps import AppConfig


class TendersConfig(AppConfig):
    name = 'tenders'
<<<<<<< HEAD
=======
    evaluator = None

    def ready(self):
        # Pre-load the anomaly evaluator to speed up tender creation
        import os
        if os.environ.get('RUN_MAIN') == 'true':
            try:
                from .ml.evaluator import TenderAnomalyEvaluator
                print("🚀 Loading Tender Anomaly Models...")
                self.__class__.evaluator = TenderAnomalyEvaluator(eager_load=True)
                print("✅ Tender Anomaly Models Ready")
            except Exception as e:
                print(f"⚠️ Warning: Could not pre-load Tender Evaluator: {e}")
>>>>>>> bd1274c (Added Chat and rafactored code)
