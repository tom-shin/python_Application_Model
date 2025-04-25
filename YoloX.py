import os
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random


class YoloXONNXDetector:
    def __init__(self, model_path, use_cuda=False,
                 conf_threshold=0.3, nms_threshold=0.45,
                 strides=[8, 16, 32]):
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider' if use_cuda else 'CPUExecutionProvider']
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        _, _, self.input_height, self.input_width = inp.shape

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.strides = strides

        self._init_preprocessing()
        self.class_labels, self.colors = None, None

    def _init_preprocessing(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def load_class_labels(self, path=None):
        if path and os.path.isfile(path):
            with open(path) as f:
                self.class_labels = [l.strip() for l in f if l.strip()]
            self.colors = [
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                for _ in self.class_labels
            ]
        else:
            print(f"[WARN] class labels file not found at {path}, proceeding without labels.")
            self.class_labels, self.colors = None, None

    def resize_with_letterbox(self, image):
        iw, ih = image.size
        w, h = self.input_width, self.input_height
        print(w, h)
        scale = min(w / iw, h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        image_resized = image.resize((nw, nh), Image.BILINEAR)
        new_img = Image.new('RGB', (w, h), (114, 114, 114))
        new_img.paste(image_resized, ((w - nw) // 2, (h - nh) // 2))
        return new_img, scale, (w - nw) // 2, (h - nh) // 2

    def preprocess(self, img_path):
        img = Image.open(img_path).convert('RGB')
        resized_img, scale, pad_x, pad_y = self.resize_with_letterbox(img)
        tensor = transforms.ToTensor()(resized_img)
        tensor = self.normalize(tensor).unsqueeze(0).numpy()
        return tensor, img, scale, pad_x, pad_y

    def infer(self, tensor):
        return self.session.run(None, {self.input_name: tensor})[0][0]

    def _make_grid(self, nx, ny):
        xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
        return np.stack((xv, yv), axis=-1).reshape(-1, 2)

    def _generate_grids_and_strides(self):
        grids, strides = [], []
        for s in self.strides:
            nx, ny = self.input_width // s, self.input_height // s
            g = self._make_grid(nx, ny)
            grids.append(g)
            strides.append(np.full((g.shape[0], 1), s))
        return np.concatenate(grids, axis=0), np.concatenate(strides, axis=0)

    def _xywh2xyxy(self, xywh):
        c, wh = xywh[:, :2], xywh[:, 2:4] / 2
        tl = c - wh
        br = c + wh
        return np.concatenate((tl, br), axis=1)

    def _iou(self, box, boxes):
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return inter / (area1 + area2 - inter + 1e-6)

    def _nms(self, boxes, scores):
        idxs = np.argsort(scores)[::-1]
        keep = []
        while idxs.size:
            i = idxs[0]
            keep.append(i)
            if idxs.size == 1:
                break
            rest = idxs[1:]
            ious = self._iou(boxes[i], boxes[rest])
            idxs = rest[ious < self.nms_threshold]
        return keep

    def postprocess(self, preds, scale, pad_x, pad_y, orig_size):
        preds_sig = 1 / (1 + np.exp(-preds))
        grids, strides = self._generate_grids_and_strides()

        raw_xy, raw_wh = preds_sig[:, :2], preds_sig[:, 2:4]
        xy = (raw_xy * 2 - 0.5 + grids) * strides
        wh = (raw_wh * 2) ** 2 * strides
        xywh = np.concatenate((xy, wh), axis=1)

        obj_conf = preds_sig[:, 4:5]
        cls_conf = preds_sig[:, 5:]
        scores = (obj_conf * cls_conf).max(axis=1)
        class_ids = cls_conf.argmax(axis=1)

        mask = scores > self.conf_threshold
        xywh, scores, class_ids = xywh[mask], scores[mask], class_ids[mask]

        boxes = self._xywh2xyxy(xywh)

        # 원본 비율에 맞게 복원
        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes /= scale

        ow, oh = orig_size
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, ow)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, oh)

        keep = self._nms(boxes, scores)
        boxes = boxes[keep].astype(np.int32)
        scores = scores[keep]
        class_ids = class_ids[keep]

        return boxes.tolist(), scores.tolist(), class_ids.tolist()

    def draw_boxes(self, image, boxes, scores, class_ids):
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()

        for box, score, cid in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            color = self.colors[cid] if self.colors else "red"
            label = self.class_labels[cid] if self.class_labels else str(cid)

            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
            text = f"{label},  {score:.2f}"
            text_size = font.getbbox(text)[2:]
            text_bg = [x1, y1 - text_size[1], x1 + text_size[0], y1]
            draw.rectangle(text_bg, fill=color)
            draw.text((x1, y1 - text_size[1]), text, fill="black", font=font)
        return image

    def detect(self, image_path):
        tensor, ori_img, scale, pad_x, pad_y = self.preprocess(image_path)
        # print(tensor)
        preds = self.infer(tensor)
        boxes, scores, cids = self.postprocess(preds, scale, pad_x, pad_y, ori_img.size)
        vis = self.draw_boxes(ori_img.copy(), boxes, scores, cids)
        return ori_img, vis


if __name__ == '__main__':
    BASE = os.path.dirname(os.path.abspath(__file__))
    model = os.path.join(BASE, 'models/ObjectDetect/YoloX/yolox_m.onnx')
    img = os.path.join(BASE, 'models/test_img/car4.jpg')
    cls = os.path.join(BASE, 'models/ObjectDetect/YoloX/yolox.txt')

    det = YoloXONNXDetector(model,
                            use_cuda=False,
                            conf_threshold=0.3,
                            nms_threshold=0.2)
    det.load_class_labels(cls)

    ori, vis = det.detect(img)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(ori); ax1.axis('off'); ax1.set_title("Original")
    ax2.imshow(vis); ax2.axis('off'); ax2.set_title("Detected")
    plt.tight_layout(); plt.show()
