from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
import random
from tenders.models import Tender, TenderCategory, Bid, BidEvaluation
from accounts.models import User

class Command(BaseCommand):
    help = 'Populates the database with comprehensive sample data for full system testing'

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.WARNING('Start Full Database Population...'))

        # 1. Clear existing data (Order matters for foreign keys)
        self.stdout.write('Clearing existing data...')
        BidEvaluation.objects.all().delete()
        Bid.objects.all().delete()
        Tender.objects.all().delete()
        # Only delete non-superuser test accounts to keep admin accessible
        User.objects.filter(is_superuser=False).delete()
        
        self.stdout.write('Data cleared.')

        # 2. Create Users
        self.stdout.write('Creating users...')
        
        # Authority Users
        auth_city = User.objects.create_user('auth_city', 'city@example.com', 'password123', first_name='City', last_name='Planner', company_name='City Infrastructure Dept')
        auth_health = User.objects.create_user('auth_health', 'health@example.com', 'password123', first_name='Dr. Sarah', last_name='Connor', company_name='National Health Board')
        auth_tech = User.objects.create_user('auth_tech', 'tech@example.com', 'password123', first_name='Tech', last_name='Director', company_name='Ministry of Technology')
        
        # Bidder Users
        bidder_acme = User.objects.create_user('bidder_acme', 'acme@example.com', 'password123', first_name='John', last_name='Doe', company_name='Acme Construction Ltd')
        bidder_global = User.objects.create_user('bidder_global', 'global@example.com', 'password123', first_name='Jane', last_name='Smith', company_name='Global Supplies Inc')
        bidder_local = User.objects.create_user('bidder_local', 'local@example.com', 'password123', first_name='Bob', last_name='Builder', company_name='Local Services Co')
        bidder_tech_sol = User.objects.create_user('bidder_tech', 'techsol@example.com', 'password123', first_name='Alice', last_name='Coder', company_name='Tech Solutions LLC')

        bidders = [bidder_acme, bidder_global, bidder_local, bidder_tech_sol]
        authorities = [auth_city, auth_health, auth_tech]

        self.stdout.write(f'Created {len(authorities)} Authorities and {len(bidders)} Bidders.')

        # 3. Create Categories
        categories_data = [
            ('Construction', 'Infrastructure and building projects'),
            ('IT Services', 'Software development and IT consulting'),
            ('Medical Supply', 'Medical equipment and pharmaceuticals'),
            ('Transport', 'Logistics and transportation services'),
            ('Energy', 'Renewable and non-renewable energy projects'),
            ('Consulting', 'Professional advisory services'),
        ]
        
        categories = {}
        for name, desc in categories_data:
            cat, _ = TenderCategory.objects.get_or_create(name=name, defaults={'description': desc})
            categories[name] = cat
        
        self.stdout.write('Categories created.')

        # 4. Create Tenders
        tenders_data = [
            {
                'title': 'Construction of New City Library',
                'description': 'Proposals are invited for the construction of a new public library. Requires strict adherence to safety standards.',
                'budget_min': 5000000, 'budget_max': 8500000,
                'status': 'OPEN', 'days_offset': 45,
                'user': auth_city, 'cat': 'Construction'
            },
            {
                'title': 'Supply of High-Performance Laptops',
                'description': 'Supply of 500 laptops. i7, 32GB RAM.',
                'budget_min': 800000, 'budget_max': 1200000,
                'status': 'OPEN', 'days_offset': 20,
                'user': auth_tech, 'cat': 'IT Services'
            },
            {
                'title': 'Hospital Cleaning Services',
                'description': 'Deep cleaning services for General Hospital.',
                'budget_min': 150000, 'budget_max': 300000,
                'status': 'OPEN', 'days_offset': 15,
                'user': auth_health, 'cat': 'Medical Supply' # technically service but okay
            },
            {
                'title': 'Solar Panel Installation',
                'description': 'Solar power for 10 schools.',
                'budget_min': 2000000, 'budget_max': 3500000,
                'status': 'OPEN', 'days_offset': 60,
                'user': auth_city, 'cat': 'Energy'
            },
             {
                'title': 'Urban Road Resurfacing',
                'description': 'Resurfacing 20km of roads.',
                'budget_min': 3000000, 'budget_max': 4500000,
                'status': 'OPEN', 'days_offset': 30,
                'user': auth_city, 'cat': 'Construction'
            },
            {
                'title': 'Cloud Migration Consultancy',
                'description': 'Migrate legacy systems to AWS.',
                'budget_min': 400000, 'budget_max': 750000,
                'status': 'OPEN', 'days_offset': 25,
                'user': auth_tech, 'cat': 'IT Services'
            },
             {
                'title': 'Supply of Ventilators',
                'description': 'Urgent procurement of 50 ICU ventilators.',
                'budget_min': 1000000, 'budget_max': 1500000,
                'status': 'closing-soon', 'days_offset': 3,
                'user': auth_health, 'cat': 'Medical Supply'
            },
             {
                'title': 'Waste Management Zone A',
                'description': 'Collection of solid waste.',
                'budget_min': 600000, 'budget_max': 900000,
                'status': 'CLOSED', 'days_offset': -5,
                'user': auth_city, 'cat': 'Transport' 
            }
        ]

        created_tenders = []
        for data in tenders_data:
            deadline = timezone.now().date() + timedelta(days=data['days_offset'])
            
            # Simple heuristic for anomaly result based on budget
            anomaly_category = 'NORMAL'
            if data['budget_max'] > 5000000:
                anomaly_category = 'LOW'

            layout = {
                'Title': data['title'],
                'Authority': data['user'].company_name,
                'Estimated_Value': f"{data['budget_min']} - {data['budget_max']}",
                'anomaly_result': {'category': anomaly_category, 'score': random.uniform(0.1, 0.4)},
                'extracted_text': data['description']
            }
            
            tender = Tender.objects.create(
                title=data['title'],
                description=data['description'],
                organization_name=data['user'].company_name,
                budget_min=data['budget_min'],
                budget_max=data['budget_max'],
                status=data['status'],
                submission_deadline=deadline,
                created_by=data['user'],
                category=categories.get(data['cat']),
                extracted_layout=layout
            )
            created_tenders.append(tender)
        
        self.stdout.write(f'Created {len(created_tenders)} Tenders.')

        # 5. Create Bids
        self.stdout.write('Creating Bids...')
        
        bid_texts = [
            "We are pleased to submit our proposal. We have extensive experience in this field.",
            "Please find our competitive bid attached. We guarantee timely delivery.",
            "Our company is the best fit for this project due to our localized presence.",
            "High quality service at an affordable price. References available upon request."
        ]

        # distribute bids
        # For open tenders, add some bids
        open_tenders = [t for t in created_tenders if t.status in ['OPEN', 'closing-soon']]
        
        for tender in open_tenders:
            # 1 to 3 bids per tender
            num_bids = random.randint(1, 3)
            selected_bidders = random.sample(bidders, num_bids)
            
            for bidder in selected_bidders:
                # Random bid amount within or slightly outside budget
                variation = random.uniform(0.9, 1.1)
                bid_amount = float(tender.budget_min) * variation
                
                # Ensure bid amount is somewhat realistic relative to range
                if bid_amount < float(tender.budget_min) * 0.8:
                    bid_amount = float(tender.budget_min)
                
                bid = Bid.objects.create(
                    tender=tender,
                    user=bidder,
                    bid_amount=round(bid_amount, 2),
                    bid_text=random.choice(bid_texts),
                    status='PENDING'
                )
                
                # 6. Create Bid Evaluations for some
                if random.choice([True, False]):
                    price_valid = bid_amount <= float(tender.budget_max)
                    BidEvaluation.objects.create(
                        bid=bid,
                        tender=tender,
                        price_valid=price_valid,
                        documents_valid=True,
                        compliance_valid=True,
                        status='VALID' if price_valid else 'INVALID',
                        remarks='Automated evaluation completed.'
                    )
                    bid.status = 'VALID' if price_valid else 'INVALID'
                    bid.save()

        self.stdout.write(self.style.SUCCESS(f'Successfully populated database! Created {Bid.objects.count()} Bids with evaluations.'))
        self.stdout.write(self.style.SUCCESS('Users created: auth_city, auth_health, auth_tech, bidder_acme, bidder_global, etc. (Password: "password123")'))
