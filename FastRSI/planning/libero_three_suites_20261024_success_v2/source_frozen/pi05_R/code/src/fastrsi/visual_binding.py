"""RGB-only template correspondence. Pixel offsets are never robot actions.

The caller supplies a source annotation from a recorded demonstration. This
module estimates appearance correspondence, not semantic identity or 3D pose.
"""
import numpy as np


def match_template(source, target, roi, count=2):
    import cv2
    for image in (source, target):
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('Expected HWC uint8 RGB image')
    if len(roi) != 4 or any(type(x) is not int for x in roi):
        raise ValueError('ROI must contain four integer coordinates')
    x0,y0,x1,y1=roi
    if not (0 <= x0 < x1 <= source.shape[1] and 0 <= y0 < y1 <= source.shape[0]):
        raise ValueError('ROI outside source')
    if type(count) is not int or count < 1:
        raise ValueError('Positive peak count required')
    template=cv2.cvtColor(source[y0:y1,x0:x1],cv2.COLOR_RGB2GRAY)
    if template.std() < 1:
        raise ValueError('Template has insufficient texture')
    h,w=template.shape
    if h > target.shape[0] or w > target.shape[1]:
        raise ValueError('Target is smaller than template')
    scores=cv2.matchTemplate(cv2.cvtColor(target,cv2.COLOR_RGB2GRAY),template,cv2.TM_CCOEFF_NORMED)
    work=scores.copy();peaks=[]
    for _ in range(count):
        _,score,_,(x,y)=cv2.minMaxLoc(work)
        if not np.isfinite(score):break
        peaks.append(dict(box_xyxy=[x,y,x+w,y+h],score=float(score),pixel_delta_xy=[x-x0,y-y0]))
        # Suppress placements overlapping the same object center.
        work[max(0,y-h//2):min(work.shape[0],y+h//2+1),max(0,x-w//2):min(work.shape[1],x+w//2+1)]=-np.inf
    return dict(peaks=peaks,peak_gap=None if len(peaks)<2 else peaks[0]['score']-peaks[1]['score'],
                metric_pose_available=False,execution_authorized=False)
