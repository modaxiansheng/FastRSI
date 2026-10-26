"""Preserve original Agent/Session lifecycle; choose fixed or memory-selected flow."""
import json
import run_native_value_worker_v1 as base
original=base.rewrite_launch

def rewrite_launch(role,argv,folder,root,project,campaign,arm,ports):
    argv=original(role,argv,folder,root,project,campaign,arm,ports)
    if role=='pi05':
        argv[-1]=argv[-1].replace('/serve_native_value_v1.py','/serve_native_complete_flow_v1.py')
        model='/project/'+str((campaign/('VALUE_MODEL_'+arm+'.npz')).relative_to(project))
        argv=['FASTRSI_NATIVE_VALUE_MODEL='+model if x.startswith('FASTRSI_NATIVE_VALUE_MODEL=') else x for x in argv]
        spec=json.loads((campaign/'FLOW_REGISTRY_SPEC.json').read_text());registry='/project/'+str((campaign/'FLOW_REGISTRY.json').relative_to(project))
        index=argv.index(base.worker.life.IMAGE)
        argv[index:index]=['-e','FASTRSI_COMPLETE_FLOW_REGISTRY='+registry,'-e','FASTRSI_COMPLETE_FLOW_REGISTRY_SHA256='+spec['sha256'],'-e','FASTRSI_COMPLETE_FLOW_MODE='+{'N':'N','F':'F','R':'C'}[arm]]
    return argv
base.rewrite_launch=rewrite_launch
if __name__=='__main__':base.main()
