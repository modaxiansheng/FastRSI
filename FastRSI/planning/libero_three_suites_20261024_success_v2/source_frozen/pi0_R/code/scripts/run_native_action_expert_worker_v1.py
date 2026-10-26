"""Original Agent lifecycle, per-arm expert head, unchanged robot/verification."""
import run_native_value_worker_v1 as base
original=base.rewrite_launch

def rewrite_launch(role,argv,folder,root,project,campaign,arm,ports):
    argv=original(role,argv,folder,root,project,campaign,arm,ports)
    if role=='pi05':
        argv[-1]=argv[-1].replace('/serve_native_value_v1.py','/serve_native_action_expert_v1.py')
        filename='/project/'+str((campaign/('VALUE_MODEL_'+arm+'.npz')).relative_to(project))
        argv=['FASTRSI_NATIVE_VALUE_MODEL='+filename if x.startswith('FASTRSI_NATIVE_VALUE_MODEL=') else x for x in argv]
    return argv

base.rewrite_launch=rewrite_launch
if __name__=='__main__':base.main()
