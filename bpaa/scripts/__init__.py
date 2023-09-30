import django
import os
import sys

# If the script is inside the bpaa directory, we adjust the path accordingly.
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('/bestproperties')

# Set up the Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

django.setup()
# import files by calling api.scripts.cleanup.purge_odds
