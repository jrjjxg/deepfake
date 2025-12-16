from __future__ import division
import collections
import numpy as np
import glob
import os
import os.path as osp
import cv2

# Use custom SCRFD detector instead of insightface model_zoo
from .scrfd_onnx import SCRFD

# face_align from insightface.utils (this import should still work)
try:
    from insightface.utils import face_align
except ImportError:
    # Fallback: minimal face_align implementation
    import skimage.transform as trans

    class face_align:
        @staticmethod
        def estimate_norm(lmk, image_size=112, mode='arcface'):
            arcface_dst = np.array([
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041]
            ], dtype=np.float32)
            if image_size != 112:
                arcface_dst = arcface_dst * image_size / 112.0
            tform = trans.SimilarityTransform()
            tform.estimate(lmk, arcface_dst)
            M = tform.params[0:2, :]
            return M, None


__all__ = ['Face_detect_crop', 'Face']

Face = collections.namedtuple('Face', [
    'bbox', 'kps', 'det_score', 'embedding', 'gender', 'age',
    'embedding_norm', 'normed_embedding',
    'landmark'
])

Face.__new__.__defaults__ = (None, ) * len(Face._fields)


class Face_detect_crop:
    def __init__(self, name, root='~/.insightface_func/models'):
        self.models = {}
        root = os.path.expanduser(root)
        onnx_files = glob.glob(osp.join(root, name, '*.onnx'))
        onnx_files = sorted(onnx_files)
        
        # Find detection model (prefer scrfd, then det_10g)
        det_model_file = None
        # First pass: look for scrfd model
        for onnx_file in onnx_files:
            basename = osp.basename(onnx_file).lower()
            if 'scrfd' in basename:
                det_model_file = onnx_file
                break
        
        # Second pass: look for det_10g model
        if det_model_file is None:
            for onnx_file in onnx_files:
                basename = osp.basename(onnx_file).lower()
                if 'det_10g' in basename:
                    det_model_file = onnx_file
                    break
        
        # Fallback: use first non-landmark onnx file
        if det_model_file is None:
            for onnx_file in onnx_files:
                basename = osp.basename(onnx_file).lower()
                if onnx_file.find('_selfgen_') > 0:
                    continue
                # Skip landmark models
                if '2d106' in basename or '1k3d68' in basename:
                    continue
                det_model_file = onnx_file
                break
        
        if det_model_file is None:
            raise RuntimeError(f'No detection model found in {root}/{name}')
        
        print('Loading detection model:', det_model_file)
        self.det_model = SCRFD(det_model_file)
        self.models['detection'] = self.det_model

    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname == 'detection':
                model.prepare(ctx_id, input_size=det_size)
            else:
                model.prepare(ctx_id)

    def get(self, img, crop_size, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             threshold=self.det_thresh,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:
            return None
        kps_list = []
        for i in range(bboxes.shape[0]):
            kps = None
            if kpss is not None:
                kps = kpss[i]
            kps_list.append(kps)
        return kps_list
