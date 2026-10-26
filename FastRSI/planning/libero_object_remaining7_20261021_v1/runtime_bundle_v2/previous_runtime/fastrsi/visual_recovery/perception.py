"""Compose a fixed initial locator with the accepted tracking/basket implementation."""
from pathlib import Path
import hashlib
import json
import time
import cv2
import numpy as np
from .proposals import distinct_rgb_proposals
from .refinement import refine_seeds


def vision_class(frozen_vision, source_path):
    """Explicit adapter; no global source mutation or runtime text replacement."""
    class MultiProposalVision(frozen_vision):
        def subject(self, observation, meta, world_prior=None):
            if world_prior is not None or meta['subject'] != 'chocolate_pudding_1':
                return super().subject(observation, meta, world_prior)
            from PIL import Image
            self.requests += 1
            assert self.requests <= 8
            rgb = np.asarray(observation['agentview_image'])
            assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
            begin = time.time()
            folder = self.out / ('%02d' % self.requests)
            folder.mkdir()
            Image.fromarray(rgb).save(folder/'front.png')
            source = json.loads(Path(source_path).read_text())
            cal = self.camera(meta['variant'], meta['task_id'])
            template = next(t for t in self.manifest['templates']['chocolate_pudding_1']
                            if t['path'].endswith('step0.png'))
            path = self.root/template['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == template['sha256']
            proposals = distinct_rgb_proposals(
                rgb[:, :, ::-1].copy(), cv2.imread(str(path)), template,
                source['skill']['source_object_pose'],
                source['calibration']['environment']['cameras']['agentview'], cal,
                -.445 if meta['variant'] == 'position' else 0.)
            world, hypotheses, forwards = refine_seeds(self, rgb, cal, meta['subject'], proposals)
            for proposal in proposals:
                proposal['seed'] = proposal['seed'].tolist()
            value = dict(event='current_rgb_subject_pose', request=self.requests,
                         step=meta['step'], phase=meta['phase'], subject=meta['subject'],
                         seconds=time.time()-begin,
                         input_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                         png_sha256=hashlib.sha256((folder/'front.png').read_bytes()).hexdigest(),
                         world_pose=world.tolist(), prior=None, forward_calls=forwards,
                         proposals=proposals, hypotheses=hypotheses,
                         selection='FROZEN_NEURAL_POSE_LOGIT_ARGMAX', GT_read=False)
            (folder/'PREDICTION.json').write_text(json.dumps(value, indent=2))
            self.emit(value)
            return world
    return MultiProposalVision
