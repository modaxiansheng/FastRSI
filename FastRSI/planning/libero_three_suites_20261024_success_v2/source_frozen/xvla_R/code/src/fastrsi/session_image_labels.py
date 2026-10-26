"""Public evidence mapping only: no answer, reward or semantic label is added."""
import copy
import json

MARKER = 'VERIFICATION_BUNDLE:\n'


def label_session_images(content):
    if not isinstance(content, list):
        raise ValueError('Verification content must be a list')
    bundles = [p['text'].split(MARKER, 1)[1] for p in content
               if p.get('type') == 'text' and MARKER in p.get('text', '')]
    if not bundles:
        return content  # unrelated episode protocol remains native
    if len(bundles) != 1:
        raise ValueError('Ambiguous session verification bundle')
    bundle = json.loads(bundles[0])
    mapping = []
    for phase in ('initial', 'final'):
        paths = bundle.get(phase + '_rgb_paths')
        if not isinstance(paths, list) or not paths:
            raise ValueError('Missing paired RGB paths')
        if not all(isinstance(p, str) and p for p in paths):
            raise ValueError('Invalid RGB path')
        mapping.extend((phase, p) for p in paths)
    if len({p for _, p in mapping}) != len(mapping):
        raise ValueError('Ambiguous duplicate RGB path')
    if sum(p.get('type') == 'image_url' for p in content) != len(mapping):
        raise ValueError('RGB path/image cardinality mismatch')
    if any(p.get('type') not in ('text', 'image_url') for p in content):
        raise ValueError('Unsupported session evidence part')
    if any(p.get('type') == 'text' and p.get('text', '').startswith('RGB_EVIDENCE_MAPPING:') for p in content):
        raise ValueError('Evidence already labelled; do not duplicate metadata')
    result = []; index = 0
    for part in content:
        if part.get('type') == 'image_url':
            phase, path = mapping[index]; index += 1
            result.append({'type': 'text', 'text': 'RGB_EVIDENCE_MAPPING: ' + json.dumps(
                {'image_index': index, 'phase': phase, 'path': path}, ensure_ascii=False)})
        result.append(copy.deepcopy(part))
    return result
