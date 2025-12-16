"""
SCRFD Face Detector - Direct ONNX Runtime Implementation
Bypasses insightface model_zoo routing issues
Auto-adapts to different SCRFD model configurations
"""
from __future__ import division
import numpy as np
import onnxruntime
import cv2


def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clip(min=0, max=max_shape[1])
        y1 = y1.clip(min=0, max=max_shape[0])
        x2 = x2.clip(min=0, max=max_shape[1])
        y2 = y2.clip(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clip(min=0, max=max_shape[1])
            py = py.clip(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


class SCRFD:
    def __init__(self, model_file, session=None):
        self.model_file = model_file
        self.session = session
        self.taskname = 'detection'
        self.center_cache = {}
        self.nms_thresh = 0.4
        self._init_vars()

    def _init_vars(self):
        self.input_size = None
        self.input_mean = 127.5
        self.input_std = 128.0
        self.use_kps = False
        self._num_anchors = 1
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]

    def prepare(self, ctx_id, input_size=None, **kwargs):
        if input_size is not None:
            self.input_size = input_size
        if self.session is None:
            if ctx_id >= 0:
                available = onnxruntime.get_available_providers()
                if 'CUDAExecutionProvider' in available:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                else:
                    providers = ['CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            self.session = onnxruntime.InferenceSession(
                self.model_file, providers=providers
            )
        
        # Parse model info from outputs
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
            self.input_size = (input_shape[2], input_shape[3])
        self.input_name = input_cfg.name
        
        outputs = self.session.get_outputs()
        self.output_names = [o.name for o in outputs]
        
        # Detect model configuration from output count
        output_count = len(outputs)
        if output_count == 6:
            # 3 score outputs + 3 bbox outputs (no keypoints)
            self.fmc = 3
            self.use_kps = False
        elif output_count == 9:
            # 3 score + 3 bbox + 3 kps outputs
            self.fmc = 3
            self.use_kps = True
        elif output_count == 10:
            # 5 score + 5 bbox (no keypoints, 5 FPN levels)
            self.fmc = 5
            self.use_kps = False
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
        elif output_count == 15:
            # 5 score + 5 bbox + 5 kps (5 FPN levels)
            self.fmc = 5
            self.use_kps = True
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
        else:
            # Fallback
            self.fmc = 3
            self.use_kps = output_count > 6
        
        # Detect number of anchors from output shape
        self._num_anchors = 2  # default for most SCRFD models

    def forward(self, img, threshold):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            img, 1.0 / self.input_std, input_size,
            (self.input_mean, self.input_mean, self.input_mean), swapRB=True
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        
        for idx, stride in enumerate(self._feat_stride_fpn):
            # Get outputs for this FPN level
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + fmc] * stride
            
            if self.use_kps:
                kps_preds = net_outs[idx + fmc * 2] * stride
            else:
                kps_preds = None
            
            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                # Generate anchor centers
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                
                # Determine num_anchors from score shape vs expected grid size
                expected_size = height * width
                actual_size = scores.size
                num_anchors = actual_size // expected_size
                if num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * num_anchors, axis=1
                    ).reshape((-1, 2))
                
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers
            
            # Reshape predictions to match anchor centers
            scores = scores.reshape(-1)
            bbox_preds = bbox_preds.reshape(-1, 4)
            
            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            
            if kps_preds is not None:
                kps_preds = kps_preds.reshape(-1, 10)
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, threshold=0.5, max_num=0, metric='default'):
        input_size = self.input_size
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        
        scores_list, bboxes_list, kpss_list = self.forward(det_img, threshold)
        
        # Filter out empty arrays
        scores_list = [s for s in scores_list if s.size > 0]
        bboxes_list = [b for b in bboxes_list if b.size > 0]
        if self.use_kps:
            kpss_list = [k for k in kpss_list if k.size > 0]
        
        if len(scores_list) == 0 or len(bboxes_list) == 0:
            return np.zeros((0, 5)), None
        
        scores = np.concatenate(scores_list)
        if scores.size == 0:
            return np.zeros((0, 5)), None
        
        bboxes = np.vstack(bboxes_list) / det_scale
        
        if self.use_kps and len(kpss_list) > 0:
            kpss = np.vstack(kpss_list) / det_scale
        else:
            kpss = None
        
        # Reshape scores to (N, 1) for hstack
        scores = scores.reshape(-1, 1)
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        
        order = pre_det[:, 4].argsort()[::-1]
        pre_det = pre_det[order, :]
        keep = nms(pre_det, self.nms_thresh)
        det = pre_det[keep, :]
        
        if kpss is not None:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        
        return det, kpss
