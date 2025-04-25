import os
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class ResNetONNXClassifier:
    def __init__(self, model_path, class_labels_path, use_cuda=False):
        self.model_path = model_path
        self.class_labels_path = class_labels_path

        # 클래스 라벨 로드
        self.class_labels = self.load_class_labels(self.class_labels_path)

        self.providers = ['CUDAExecutionProvider' if use_cuda else 'CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name

        # 입력 텐서의 크기 추출 (N, C, H, W)
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        self._init_preprocessing()

    def _init_preprocessing(self):
        self.transform = transforms.Compose([
            transforms.Resize((self.input_height, self.input_width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_class_labels(self, class_labels_path):
        with open(class_labels_path, 'r') as f:
            class_labels = f.read().splitlines()
        return class_labels

    def preprocess(self, image_path):
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0)  # 배치 차원 추가
        return tensor.numpy(), image

    def infer(self, input_tensor):
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return outputs

    def postprocess(self, output):
        probabilities = np.squeeze(output[0])
        top_class = np.argmax(probabilities)
        confidence = probabilities[top_class]
        return top_class, confidence

    def predict(self, image_path):
        input_tensor, original_image = self.preprocess(image_path)
        output = self.infer(input_tensor)
        top_class, confidence = self.postprocess(output)
        try:
            class_name = self.class_labels[top_class]
        except:
            class_name = str(top_class)
        return class_name, confidence, original_image

    def visualize_prediction(self, image, class_name, confidence):
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Prediction: {class_name} ({confidence:.2%})", fontsize=14)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, 'models', 'ObjectDetect', 'DETR_ResNet50', 'resnet50.onnx')
    class_labels_path = os.path.join(BASE_DIR, 'models', 'ObjectDetect', 'DETR_ResNet50', 'DETR_ResNet50_dc5.txt')
    image_path = os.path.join(BASE_DIR, 'models', 'test_img', 'test4.jpg')

    classifier = ResNetONNXClassifier(model_path, class_labels_path, use_cuda=False)
    class_name, confidence, image = classifier.predict(image_path)

    print(f'예측 클래스: {class_name}, 확신도: {confidence:.4f}')
    classifier.visualize_prediction(image, class_name, confidence)
