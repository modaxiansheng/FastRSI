"""Frozen neural refinement and explicit learned-score ranking, no outcome access."""
import numpy as np


def refine_seeds(vision, rgb, camera, label, proposals):
    import pandas as pd
    from megapose.utils.tensor_collection import PandasTensorCollection
    from megapose.inference.types import ObservationTensor
    assert 1 <= len(proposals) <= 6
    coarse = PandasTensorCollection(pd.DataFrame([
        dict(label=label, batch_im_id=0, instance_id=0, hypothesis_id=i)
        for i in range(len(proposals))]), poses=vision.torch.tensor(
            np.stack([p['seed'] for p in proposals]), dtype=vision.torch.float32))
    obs = ObservationTensor.from_numpy(rgb[:, ::-1].copy(), None,
                                       np.asarray(camera['K'], np.float32))
    before = vision.forward_count
    with vision.torch.no_grad():
        pred, extra = vision.est.run_inference_pipeline(
            obs, coarse_estimates=coarse, n_pose_hypotheses=len(proposals),
            n_refiner_iterations=5, run_depth_refiner=False)
    assert extra['coarse']['data'] is None
    world_camera = np.asarray(camera['T_world_camera_cv'])
    scoring = extra['scoring']['preds']
    poses = scoring.poses.cpu().numpy()
    infos = scoring.infos.to_dict('records')
    rows = [dict(info=info, world_pose=(world_camera @ pose).tolist())
            for info, pose in zip(infos, poses)]
    chosen = pred.poses.cpu().numpy()[0]
    top_score = max(float(x['pose_logit']) for x in infos)
    assert any(float(info['pose_logit']) == top_score and np.array_equal(chosen, p)
               for info, p in zip(infos, poses))
    return world_camera @ chosen, rows, vision.forward_count-before
