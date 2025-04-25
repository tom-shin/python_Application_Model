import os
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import onnxruntime as ort
import torchvision.transforms as transforms


class DDRNetONNXSegmenter:
    def __init__(self, model_path, use_cuda=False):
        self.model_path = model_path
        self.providers = ['CUDAExecutionProvider' if use_cuda else 'CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=self.providers)

        # 입력 텐서 정보
        input_shape = self.session.get_inputs()[0].shape  # (N, C, H, W)
        self.input_name = self.session.get_inputs()[0].name
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        self._init_preprocessing()

    def _init_preprocessing(self):
        self.transform = transforms.Compose([
            transforms.Resize((self.input_height, self.input_width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # 일반적인 ImageNet 기준
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess(self, image_path):
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        transformed = self.transform(image).unsqueeze(0)  # (1, C, H, W)
        return transformed.numpy(), original_size, image

    def infer(self, input_tensor):
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return outputs

    def postprocess(self, output, original_size):
        seg_map = np.squeeze(output[0])  # (1, H, W) or (num_classes, H, W)

        if seg_map.ndim == 3:
            seg_map = np.argmax(seg_map, axis=0)  # 다중 클래스 처리

        # 동적으로 color_map 생성
        num_classes = len(np.unique(seg_map))  # 고유한 클래스 수 확인
        color_map = self.generate_color_map(num_classes)

        # 색상 맵을 사용하여 세그멘테이션 맵에 색상 적용
        colored_seg_map = self.apply_color_map(seg_map, color_map)

        # Resize to original image size
        seg_image = Image.fromarray(colored_seg_map)
        seg_image = seg_image.resize(original_size, resample=Image.NEAREST)
        return seg_image

    def generate_color_map(self, num_classes):
        # 고유한 색상 생성 (랜덤 색상 생성)
        color_map = {}
        for i in range(num_classes):
            # RGB 색상을 랜덤으로 생성
            color_map[i] = [random.randint(0, 255) for _ in range(3)]
        return color_map

    def apply_color_map(self, seg_map, color_map):
        # 각 픽셀에 색상 적용
        colored_map = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
        for i in range(seg_map.shape[0]):
            for j in range(seg_map.shape[1]):
                class_id = seg_map[i, j]
                colored_map[i, j] = color_map.get(class_id, [0, 0, 0])  # 기본적으로 검정색 (배경)
        return colored_map

    def predict_and_save(self, image_path, save_path):
        input_tensor, original_size, original_image = self.preprocess(image_path)
        output = self.infer(input_tensor)
        seg_image = self.postprocess(output, original_size)

        # 원본 이미지와 세그멘테이션 결과 함께 표시
        self.plot_and_save(original_image, seg_image, save_path)

    def plot_and_save(self, original_image, seg_image, save_path):
        # matplotlib로 원본 이미지와 세그멘테이션 결과를 나란히 표시
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.imshow(original_image)
        ax1.set_title("Original Image")
        ax1.axis('off')

        ax2.imshow(seg_image)
        ax2.set_title("Segmentation Result")
        ax2.axis('off')

        # 결과를 이미지로 저장
        # plt.savefig(save_path)
        print(f"Segmentation result saved to: {save_path}")

        # 화면에 표시 (선택 사항)
        plt.show()


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, 'models', 'ImageSeg', 'DDRNet23_slim', 'ddrnet23_slim.onnx')
    image_path = os.path.join(BASE_DIR, 'models', 'test_img', 'test4.jpg')
    save_path = os.path.join(BASE_DIR, 'models', 'ImageSeg', 'DDRNet23_slim', 'result_segmentation.png')

    segmenter = DDRNetONNXSegmenter(model_path, use_cuda=False)
    segmenter.predict_and_save(image_path, save_path)
