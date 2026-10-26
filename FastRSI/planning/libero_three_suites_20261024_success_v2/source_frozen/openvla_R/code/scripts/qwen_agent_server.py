"""Qwen2.5-VL service with full history and validated structured tool calls."""
import socket
from service_endpoints import service_ports
PORTS=service_ports((49500,49502,49510,49520))
# Reserve the listening port before model loading to avoid startup port races.
LISTENER = socket.socket()
LISTENER.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
LISTENER.bind(("127.0.0.1", PORTS['qwen']))
LISTENER.listen(128)
import asyncio
import base64
import io
import hashlib
import transformers
import json
import os
import time
import uuid
from pathlib import Path
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, StoppingCriteria, StoppingCriteriaList
from fastrsi.tool_protocol import prepare_messages, parse_qwen_completion, prepare_native_messages, parse_native_completion, file_edit_wire_guidance, ProtocolError

OUT = Path(os.environ['FASTRSI_SERVICE_LOG']); OUT.mkdir(parents=True, exist_ok=True)
MODEL_NAME=os.environ.get('FASTRSI_AGENT_MODEL','qwen25vl7b')
NATIVE_TEMPLATE=os.environ.get('FASTRSI_QWEN_WIRE')=='checkpoint_native'
model_class=Qwen2_5_VLForConditionalGeneration
if os.environ.get('FASTRSI_QWEN_ARCH')=='qwen3_vl':
    from transformers import Qwen3VLForConditionalGeneration
    model_class=Qwen3VLForConditionalGeneration
processor = AutoProcessor.from_pretrained('/model', trust_remote_code=False)
model = model_class.from_pretrained('/model', torch_dtype=torch.bfloat16,
    device_map='cuda', attn_implementation='sdpa', trust_remote_code=True).eval()
(OUT/'SERVICE_IDENTITY.json').write_text(json.dumps({
    'model_name':MODEL_NAME,'model_class':type(model).__name__,'model_type':model.config.model_type,
    'transformers':transformers.__version__,'torch':torch.__version__,'native_template':NATIVE_TEMPLATE,
    'config_sha256':hashlib.sha256(Path('/model/config.json').read_bytes()).hexdigest(),
    'chat_template_sha256':hashlib.sha256(str(processor.chat_template).encode()).hexdigest(),
    'hip_visible_devices':os.environ.get('HIP_VISIBLE_DEVICES'),'loaded_at':time.time()},indent=2))
from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
from system_tool_adapter import parse_native_completion, parse_qwen_completion, recent_handoff
from fastrsi.constrained_tools import ToolJSONLogitsProcessor
from fastrsi.decoding_policy import generation_options,memory_repetition_processor
from transformers import LogitsProcessorList
tokenizer_data = build_token_enforcer_tokenizer_data(processor.tokenizer)
tokenizer_data.eos_token_id = model.generation_config.eos_token_id
app = FastAPI()
lock = asyncio.Lock()


@app.get('/healthz')
def health():
    return {'ok': True, 'protocol': 'fastrsi_qwen_tools_v1','model':MODEL_NAME,'native_template':NATIVE_TEMPLATE}


@app.post('/v1/chat/completions')
async def chat(request: Request):
    body = await request.json()
    call = OUT / ('call_' + uuid.uuid4().hex); call.mkdir()
    (call/'REQUEST.json').write_text(json.dumps(body, ensure_ascii=False))
    receipt = {'started_at': time.time(), 'stage': 'input_validation'}
    try:
        convo = prepare_native_messages(body) if NATIVE_TEMPLATE else prepare_messages(body)
        images = []
        for m in convo:
            if isinstance(m['content'], list):
                for part in m['content']:
                    if part.get('type') == 'image_url':
                        url = part['image_url']['url']
                        if not url.startswith('data:image/'):
                            raise ProtocolError('Only inline image data supported')
                        images.append(Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))).convert('RGB'))
                        part.clear(); part.update(type='image')
                    elif part.get('type') != 'text':
                        raise ProtocolError('Unsupported content part')
        continuing = not NATIVE_TEMPLATE and convo[-1]['role'] == 'assistant'
        wire_tools = file_edit_wire_guidance(body.get('tools'))
        (call/'WIRE_TOOLS.json').write_text(json.dumps(wire_tools, ensure_ascii=False))
        prompt = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=not continuing, **({'tools':wire_tools} if NATIVE_TEMPLATE else {}))
        if continuing:
            # Continue the last assistant turn, as in QwenFnCallPrompt.
            tail='<|im_end|>\n'
            if not prompt.endswith(tail):
                raise ProtocolError('Unexpected chat-template continuation suffix')
            prompt=prompt[:-len(tail)]
        (call/'MODEL_PROMPT.txt').write_text(prompt)
        limit = int(body.get('max_tokens') or 1024)
        if not 1 <= limit <= 4096:
            raise ProtocolError('Token limit outside bounded service contract')
        async with lock:
            receipt['queue_seconds'] = time.time() - receipt['started_at']
            started = time.perf_counter()
            inputs = processor(text=[prompt], images=images or None, return_tensors='pt').to(model.device)
            size = inputs['input_ids'].shape[1]
            if size > 65536:
                raise ProtocolError('Context limit exceeded; no silent truncation')
            receipt['stage'] = 'generation'
            class ToolBoundary(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    tail=processor.tokenizer.decode(input_ids[0,size:],skip_special_tokens=True)
                    return any(marker in tail for marker in ('✿RESULT✿','✿RETURN✿'))
            variant=body.get('fastrsi_decoder',os.environ.get('FASTRSI_DECODER_VARIANT','reference'))
            constraint = ToolJSONLogitsProcessor(processor.tokenizer, tokenizer_data, size, [t['function']['name'] for t in body.get('tools',[])],variant=variant) if NATIVE_TEMPLATE and body.get('tools') else None
            decoding, seed = generation_options(body)
            memory_penalty=float(os.environ.get('FASTRSI_MEMORY_REPETITION_PENALTY','1.0'))
            repeat=memory_repetition_processor(body,size,memory_penalty)
            receipt['memory_generated_repetition_penalty']=memory_penalty if repeat else 1.0
            receipt['memory_prompt_penalized']=False
            receipt['decoding'] = dict(decoding,seed=seed,
                checkpoint_top_p=model.generation_config.top_p,checkpoint_top_k=model.generation_config.top_k,
                checkpoint_repetition_penalty=model.generation_config.repetition_penalty)
            if seed is not None:torch.manual_seed(seed)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=limit, **decoding,
                    logits_processor=LogitsProcessorList(([repeat] if repeat else [])+([constraint] if constraint else [])),
                    stopping_criteria=StoppingCriteriaList([] if NATIVE_TEMPLATE else [ToolBoundary()]))
            output = generated[:, size:]
            raw = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            receipt.update(generation_seconds=time.perf_counter()-started,
                prompt_tokens=int(size), completion_tokens=int(output.shape[1]))
        receipt['structured_decode'] = dict(constraint.stats) if constraint else None
        token_ids = output[0].tolist()
        eos_ids = model.generation_config.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
        ended_with_eos = bool(token_ids and token_ids[-1] in eos_ids)
        reached_limit = len(token_ids) >= limit
        generation_finish = 'eos' if ended_with_eos else ('length' if reached_limit else 'other_stop')
        receipt.update(requested_max_tokens=limit, generation_finish_reason=generation_finish,
            ended_with_eos=ended_with_eos, reached_token_limit=reached_limit,
            eos_token_ids=eos_ids, last_token_id=token_ids[-1] if token_ids else None)
        (call/'OUTPUT_TOKEN_IDS.json').write_text(json.dumps(token_ids))
        (call/'RAW.txt').write_text(raw)
        receipt['stage'] = 'output_validation'
        receipt['wire_format'] = 'checkpoint_native' if NATIVE_TEMPLATE else 'qwen_function_markers_v1'
        message, finish = (parse_native_completion if NATIVE_TEMPLATE else parse_qwen_completion)(raw, body.get('tools'), body.get('tool_choice', 'auto'),
                                          truncated=reached_limit and not ended_with_eos)
        receipt['native_tool_validation_handoff'] = recent_handoff()
        receipt['api_finish_reason'] = finish
        reply = {'id':call.name,'object':'chat.completion','created':int(time.time()),'model':MODEL_NAME,
            'choices':[{'index':0,'message':message,'finish_reason':finish}],
            'usage':{'prompt_tokens':int(size),'completion_tokens':int(output.shape[1]),'total_tokens':int(generated.shape[1])}}
        (call/'RESPONSE.json').write_text(json.dumps(reply, ensure_ascii=False))
        receipt['status'] = 'OK'
        return JSONResponse(reply)
    except (ValueError, KeyError, TypeError) as exc:
        receipt.update(status='REJECTED',error=repr(exc))
        error_body={'error':{'message':str(exc),'type':'tool_protocol_error'}}
        (call/'ERROR_RESPONSE.json').write_text(json.dumps(error_body))
        return JSONResponse(error_body,status_code=422)
    except Exception as exc:
        receipt.update(status='FAILED',error=repr(exc)); raise
    finally:
        receipt.update(finished_at=time.time(),seconds=time.time()-receipt['started_at'])
        (call/'RECEIPT.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    uvicorn.run(app, fd=LISTENER.fileno(), log_level='info')
