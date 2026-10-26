"""Configure the existing native worker's policy and runner entrypoints."""
import json
import sys
from pathlib import Path
import run_placement_worker as worker
from service_endpoints import shifted_ports,docker_port_env

def rewrite_launch(role, argv, folder, base, project, campaign, arm, ports):
    argv=list(argv)
    # The simulator has a separate CPU image. It requires no rewrite here.
    if role=='target':return argv
    index=argv.index(worker.life.IMAGE)
    if role in ('qwen','runner'):
        # Bind complete endpoint data even when the preserved historical
        # lifecycle template predates explicit port environment forwarding.
        argv[index:index]=docker_port_env(ports)
        index=argv.index(worker.life.IMAGE)
    if role=='pi05':
        assert 'python -m PhyAgentOS.runtime.policy.openpi.native_openpi_server' in argv[-1]
        argv[-1]=argv[-1].replace('python -m PhyAgentOS.runtime.policy.openpi.native_openpi_server',
                                 'python '+base+'/code/scripts/serve_native_value_v1.py')
        argv=[value.replace('PYTHONPATH=', 'PYTHONPATH='+base+'/code/scripts:'+base+'/code/src:')
              if value.startswith('PYTHONPATH=') else value for value in argv]
        (folder/'value_service').mkdir()
        model='/project/'+str((campaign/'VALUE_MODEL.npz').relative_to(project))
        argv[index:index]=['-v',str(project)+':/project:ro',
            '-e','FASTRSI_NATIVE_VALUE_SERVICE=/pilot/value_service',
            '-e','FASTRSI_NATIVE_VALUE_MODEL='+model]
    elif role=='runner':
        assert argv[-1].endswith('/run_native_baseline.py')
        argv[-1]=argv[-1].removesuffix('/run_native_baseline.py')+'/run_native_value_v1.py'
        config='/project/'+str((campaign/('VALUE_CONFIG_'+arm+'.json')).relative_to(project))
        argv[index:index]=['-e','FASTRSI_NATIVE_VALUE_CONFIG='+config]
    return argv


def main():
    name,arm=sys.argv[1:];project=worker.life.PROJECT;campaign=project/'runs'/name
    protocol=json.loads((campaign/'PROTOCOL.json').read_text())
    ports=shifted_ports(protocol['port_offset']+protocol['slot']*1000+('N','F','R','S').index(arm)*100)
    worker.life.OLD=campaign/'templates'
    original=worker.life.launch
    def launch(role,argv,folder):
        return original(role,rewrite_launch(role,argv,folder,worker.life.BASE,project,campaign,arm,ports),folder)
    worker.life.launch=launch
    worker.main()

if __name__=='__main__':main()
