"""Frozen model service with suite-specific OpenVLA normalization binding."""
import ast,os,sys
from pathlib import Path
source=Path(os.environ['LIBEROPRO_CAMPAIGN_CODE'])/'scripts/multi_vla_server.py'
sys.path.insert(0,str(source.parent))
text=source.read_text();old="unnorm_key='libero_10'"
if old in text:text=text.replace(old,"unnorm_key=os.environ['LIBEROPRO_BASE_SUITE']")
exec(compile(text,str(source),'exec'),{'__name__':'__main__','__file__':str(source),'os':os})
