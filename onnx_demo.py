import cv2
import copy
import torch
import argparse
import onnxruntime
import numpy as np

from utils.utils import \
    letterbox,scale_coords,non_max_suppression,split_for_trace_model,\
    driving_area_mask,lane_line_mask,plot_one_box,show_seg_result

class Video():
    def __init__(self, args) -> None:
        self.args = args
        if isinstance(int(args.source), int): self._stream = cv2.VideoCapture(int(args.source), cv2.CAP_DSHOW)
        elif isinstance(args.source, str): self._stream = cv2.VideoCapture(args.source)
        self._fps = self._stream.get(cv2.CAP_PROP_FPS)
        self.stride = 32
        self._resized_width = 640
        self._resized_height = 384
        self._fourcc = cv2.resize(self._stream.get(cv2.CAP_PROP_FOURCC), (self._resized_width, self._resized_height), cv2.INTER_LINEAR)
        self.sess = onnxruntime.InferenceSession(args.onnx_model, providers = ['CUDAExecutionProvider'])

    def run(self):
        while self._stream.isOpened():
            ret, self.img = self._stream.read()
            if ret:
                self.img = cv2.resize(self.img, (1280, 720), interpolation=cv2.INTER_LINEAR)
                self.img_copy = copy.copy(self.img)
                self.img = letterbox(self.img, self._resized_width, stride=self.stride)[0]
                self.img = self.img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                self.img = np.ascontiguousarray(self.img)
                self.img = torch.from_numpy(self.img).float()
                self.img /= 255.0
                if self.img.ndimension() == 3: self.img = self.img.unsqueeze(0)
                onnx_pred, onnx_anchor_grid_0, onnx_anchor_grid_1, onnx_anchor_grid_2, onnx_seg, onnx_ll = \
                    self.sess.run(None, {self.sess.get_inputs()[0].name: self.img.numpy()})
                self.post_process(onnx_pred, onnx_anchor_grid_0, onnx_anchor_grid_1, onnx_anchor_grid_2, onnx_seg, onnx_ll)
                cv2.imshow('YOLOPv2_demo', self.img_copy)
                if cv2.waitKey(1) & 0xFF == 27: break
            else: break
        self.destory()

    def post_process(self, onnx_pred, onnx_anchor_grid_0, onnx_anchor_grid_1, onnx_anchor_grid_2, onnx_seg, onnx_ll):
        onnx_pred_0, onnx_pred_1, onnx_pred_2 = onnx_pred
        onnx_anchor_grid_0, onnx_anchor_grid_1, onnx_anchor_grid_2 = onnx_anchor_grid_0[None], onnx_anchor_grid_1[None], onnx_anchor_grid_2[None]
        onnx_pred_0 = torch.from_numpy(onnx_pred_0)
        onnx_pred_1 = torch.from_numpy(onnx_pred_1)
        onnx_pred_2 = torch.from_numpy(onnx_pred_2)
        onnx_anchor_grid_0 = torch.from_numpy(onnx_anchor_grid_0)
        onnx_anchor_grid_1 = torch.from_numpy(onnx_anchor_grid_1)
        onnx_anchor_grid_2 = torch.from_numpy(onnx_anchor_grid_2)
        pred = [onnx_pred_0, onnx_pred_1, onnx_pred_2]
        anchor_grid = [onnx_anchor_grid_0, onnx_anchor_grid_1, onnx_anchor_grid_2]
        seg = torch.from_numpy(onnx_seg)
        ll = torch.from_numpy(onnx_ll)
        pred = split_for_trace_model(pred,anchor_grid)
        pred = non_max_suppression(pred, self.args.conf_thres, self.args.iou_thres, classes=self.args.classes, agnostic=self.args.agnostic_nms)
        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        for i, det in enumerate(pred):
            gn = torch.tensor(self.img_copy.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(self.img.shape[2:], det[:, :4], self.img_copy.shape).round()
                for c in det[:, -1].unique(): n = (det[:, -1] == c).sum()
            for *xyxy, conf, cls in reversed(det): plot_one_box(xyxy, self.img_copy, line_thickness=3)
            show_seg_result(self.img_copy, (da_seg_mask, ll_seg_mask), is_demo=True)

    def destory(self):
        self._stream.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--onnx-model', type=str, default='YOLOPv2.onnx', help='ONNX model')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    args = parser.parse_args()

    print(args)
    video = Video(args)
    video.run()
