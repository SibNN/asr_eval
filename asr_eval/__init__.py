from pathlib import Path
from dotenv import load_dotenv

# load env variables from .env file in same directory
# as python script or searches for it incrementally higher up.
load_dotenv()

ROOT_DIR = Path(__file__).parent