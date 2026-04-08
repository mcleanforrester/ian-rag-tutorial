from dotenv import load_dotenv

load_dotenv(override=True)

from phoenix.otel import register

register(project_name="tutorial2", auto_instrument=True)
