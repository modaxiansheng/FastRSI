"""One live schema repair check using the saved real request, no new robot run."""
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from PhyAgentOS.providers.custom_provider import CustomProvider
from PhyAgentOS.verification.service import _normalize
from fastrsi.verification_engine import VerificationEngine

async def main():
    source=Path('/project/runs/native_baseline_v3_20261002/stream/service_calls/call_42214109939a423cbf273501b4c08574/REQUEST.json')
    body=json.loads(source.read_text());out=Path(os.environ['FASTRSI_NATIVE_RUN'])/'VERIFIER_REPLAY_RESULT.json'
    provider=CustomProvider(api_base='http://127.0.0.1:49510/v1',default_model=os.environ['FASTRSI_AGENT_MODEL'])
    result={'status':'RUNNING','scope':'schema repair replay, not another robot episode or learning event',
            'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    start=time.time()
    try:
        value=await VerificationEngine(provider=provider,model=os.environ['FASTRSI_AGENT_MODEL']).complete(
            system_prompt=body['messages'][0]['content'],content=body['messages'][1]['content'])
        normalized=_normalize(value);assert normalized['verifier_status']=='completed'
        result.update(status='PASSED',verdict=normalized)
    except Exception as exc:
        result.update(status='FAILED',error=repr(exc));raise
    finally:
        await provider._client.close();result['seconds']=time.time()-start;out.write_text(json.dumps(result,indent=2))
if __name__=='__main__':asyncio.run(main())
