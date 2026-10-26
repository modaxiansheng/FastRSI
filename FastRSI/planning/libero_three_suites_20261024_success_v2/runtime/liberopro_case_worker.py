"""Run the exact frozen P/R worker with explicit suite-budget compatibility."""
import runpy,os
from pathlib import Path
from liberopro_binding import budget_compatibility
from session_io_retry_v3 import install
install()
budget_compatibility()
runpy.run_path(str(Path(os.environ['LIBEROPRO_CAMPAIGN_CODE'])/'scripts/system_case_worker.py'),run_name='__main__')
