"""Four predeclared saved-request calls; labels are metadata, never task answers."""
import asyncio,copy,hashlib,json,os,time
from pathlib import Path
from PhyAgentOS.providers.custom_provider import CustomProvider
from PhyAgentOS.verification.service import _normalize
from fastrsi.contracts import json_object_text
from fastrsi.session_image_labels import label_session_images

async def main():
    root=Path(os.environ['FASTRSI_NATIVE_RUN']);rows=[];started=time.time()
    provider=CustomProvider(api_base='http://127.0.0.1:49510/v1',default_model=os.environ['FASTRSI_AGENT_MODEL'])
    try:
        for arm in ['A','B']:
            paths=list((Path('/project/runs/session_flow_v1_20261002')/arm/'stream/service_calls').glob('*/REQUEST.json'))
            paths=[p for p in paths if 'tools' not in json.loads(p.read_text())]
            assert len(paths)==1
            source=paths[0];original=json.loads(source.read_text())
            for variant in ['original','labelled']:
                body=copy.deepcopy(original)
                if variant=='labelled':body['messages'][1]['content']=label_session_images(body['messages'][1]['content'])
                assert [p for p in body['messages'][1]['content'] if p['type']=='image_url']==[p for p in original['messages'][1]['content'] if p['type']=='image_url']
                assert body['messages'][0]==original['messages'][0]
                assert body['messages'][1]['content'][0]==original['messages'][1]['content'][0]
                folder=root/(arm+'_'+variant);folder.mkdir()
                (folder/'REQUEST.json').write_text(json.dumps(body,ensure_ascii=False))
                start=time.time();response=await provider.chat(**body)
                (folder/'RESPONSE.json').write_text(json.dumps(response.__dict__,default=str))
                normalized=_normalize(json.loads(json_object_text(response.content)))
                assert normalized['verifier_status']=='completed'
                row={'source_arm':arm,'variant':variant,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                     'seconds':time.time()-start,'verdict':normalized,'images_unchanged':True,'native_text_unchanged':True}
                rows.append(row);(folder/'RESULT.json').write_text(json.dumps(row,indent=2))
    finally:
        await provider._client.close()
        (root/'RESULT.json').write_text(json.dumps({'status':'COMPLETED' if len(rows)==4 else 'PARTIAL','cases':rows,'seconds':time.time()-started,
          'scope':'Saved evidence diagnostic only; no physical episodes, experience updates or learned benefit'},indent=2))

if __name__=='__main__':asyncio.run(main())
