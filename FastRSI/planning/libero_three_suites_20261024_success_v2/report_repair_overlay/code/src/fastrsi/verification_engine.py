"""Drop-in engine overlay: strict fence parsing, budget and complete call receipts."""
import asyncio
import json
import os
import time
from .contracts import json_object_text
from .session_image_labels import label_session_images
from .libero_verification_semantics import adapt_libero_verification_content
from cost_trace import emit, span


SCHEMA_CONTRACT = '\nOutput schema requirement: lesson must be a non-empty string for every verdict, including success. Use one brief statement supported by the supplied evidence; do not invent a generalized skill. Use null for failure_reason or replan_task_description when not applicable.'


class VerificationEngine:
    def __init__(self, *, provider, model, timeout_s=180.0):
        self.provider, self.model, self.timeout_s = provider, model, float(timeout_s)

    async def complete(self, *, system_prompt, content):
        semantics_mode = os.environ.get('FASTRSI_VERIFICATION_SEMANTICS', 'off')
        if semantics_mode != 'off':
            semantics_started = time.perf_counter()
            content, semantics_receipt = adapt_libero_verification_content(content, semantics_mode)
            emit(dict(event='verification_semantics', **semantics_receipt,
                      seconds=time.perf_counter()-semantics_started))
        content = label_session_images(content)
        budget = int(os.environ.get('FASTRSI_TOKEN_BUDGET', '384'))
        for attempt, limit in enumerate([budget] + ([384] if budget < 384 else [])):
            started = time.perf_counter()
            response = None
            try:
                with span('verification_model', token_budget=limit, attempt=attempt):
                    response = await asyncio.wait_for(self.provider.chat_with_retry(
                        messages=[{'role': 'system', 'content': system_prompt + SCHEMA_CONTRACT},
                                  {'role': 'user', 'content': content}],
                        model=self.model, tools=None, temperature=0.0, max_tokens=limit),
                        timeout=self.timeout_s)
                if response.finish_reason in ('error', 'length') or not response.content:
                    raise ValueError('Unusable completion: ' + str(response.finish_reason))
                value = json.loads(json_object_text(response.content))
                # Use the original service schema before declaring a budget trial valid.
                from PhyAgentOS.verification.service import _normalize
                normalized=_normalize(value)
                if normalized['verifier_status'] != 'completed':
                    raise ValueError('Invalid semantic-verdict schema: '+str(normalized.get('failure_reason')))
                emit({'event': 'engine_completion', 'ok': True, 'attempt': attempt,
                      'budget': limit, 'usage': response.usage, 'seconds': time.perf_counter()-started})
                return value
            except Exception as exc:
                emit({'event': 'engine_completion', 'ok': False, 'attempt': attempt,
                      'budget': limit, 'usage': getattr(response, 'usage', {}),
                      'error': str(exc), 'seconds': time.perf_counter()-started})
                if limit == 384:
                    raise
