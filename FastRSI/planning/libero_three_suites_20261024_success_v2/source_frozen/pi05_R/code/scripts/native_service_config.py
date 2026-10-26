"""Explicit shared model configuration for the controlled native Agent baseline."""
import json

def configure_qwen(argv, cfg, life):
    argv=list(argv)
    idx=argv.index(life.IMAGE)
    argv[idx:idx]=['-v',str(life.PROJECT)+':/project:ro','-e','PYTHONPATH='+life.BASE+'/code/src:/project/envs/structured-decode-v1',
        '-e','FASTRSI_SERVICE_LOG=/pilot/service_calls',
        '-e','FASTRSI_AGENT_MODEL='+cfg['model_id'],'-e','FASTRSI_QWEN_ARCH='+cfg['architecture'],
        '-e','FASTRSI_QWEN_WIRE='+cfg['wire_format'],
        '-e','FASTRSI_DECODER_VARIANT='+cfg.get('decoder_variant','reference'),
        '-e','FASTRSI_MEMORY_REPETITION_PENALTY='+str(cfg.get('common_generation',{}).get('memory',{}).get('generated_repetition_penalty',1.0))]
    argv=[cfg['weights']+':/model:ro' if a.endswith(':/model:ro') else a for a in argv]
    argv[-1]='source /opt/dtk/env.sh && source /opt/hyhal/env.sh && exec '+cfg['service_python']+' '+life.BASE+'/code/scripts/qwen_agent_server.py'
    return argv

def model_env(cfg):
    return ['-e','FASTRSI_AGENT_MODEL='+cfg['model_id']]
