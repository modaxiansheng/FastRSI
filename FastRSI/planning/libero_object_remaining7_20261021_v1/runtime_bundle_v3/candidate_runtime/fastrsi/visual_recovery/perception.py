"""Compose a fixed initial locator with the accepted tracking/basket implementation."""
from pathlib import Path
import hashlib
import json
import time
import cv2
import numpy as np
from .proposals import distinct_rgb_proposals
from .refinement import refine_seeds


def vision_class(frozen_vision, source_path, locator_registry_path=None):
    """Explicit adapter; no global source mutation or runtime text replacement."""
    registry = (json.loads(Path(locator_registry_path).read_text())
                if locator_registry_path is not None else {'subjects': {}})
    registered = registry['subjects']
    class MultiProposalVision(frozen_vision):
        def subject(self, observation, meta, world_prior=None):
            label = meta['subject']
            if world_prior is not None or (label != 'chocolate_pudding_1' and label not in registered):
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
            cal = self.camera(meta['variant'], meta['task_id'])
            registration = None
            if label in registered:
                registration = registered[label]
                if 'variants' in registration:
                    registration = registration['variants'][meta['variant']]
                target = registration['current_static_registration'][meta['variant']+':'+str(meta['task_id'])]
                current_cal_path = self.root/target['calibration_path']
                assert hashlib.sha256(current_cal_path.read_bytes()).hexdigest() == target['calibration_sha256']
                source_cal_path = self.root/registration['source_calibration_path']
                assert hashlib.sha256(source_cal_path.read_bytes()).hexdigest() == registration['source_calibration_sha256']
                template = registration['template']
                assert template in self.manifest['templates'][label]
                source_pose = registration['source_object_pose']
                source_camera = registration['source_camera']
                plane_delta = target['plane_delta_m']
            else:
                # Preserve the accepted pudding implementation byte-for-byte numerics.
                source = json.loads(Path(source_path).read_text())
                template = next(t for t in self.manifest['templates']['chocolate_pudding_1']
                                if t['path'].endswith('step0.png'))
                source_pose = source['skill']['source_object_pose']
                source_camera = source['calibration']['environment']['cameras']['agentview']
                plane_delta = -.445 if meta['variant'] == 'position' else 0.
            path = self.root/template['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == template['sha256']
            proposals = distinct_rgb_proposals(
                rgb[:, :, ::-1].copy(), cv2.imread(str(path)), template,
                source_pose, source_camera, cal, plane_delta)
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
            if registration is not None:
                value.update(locator_registration_sha256=hashlib.sha256(Path(locator_registry_path).read_bytes()).hexdigest(),
                             template_path=template['path'], source_identity=registration['source_identity'],
                             source_step=0, source_pose_supervision=registration['source_pose_supervision'],
                             static_support_plane_delta_m=plane_delta,
                             current_truth_read=False, current_static_registration=target)
            (folder/'PREDICTION.json').write_text(json.dumps(value, indent=2))
            self.emit(value)
            return world
    return MultiProposalVision
