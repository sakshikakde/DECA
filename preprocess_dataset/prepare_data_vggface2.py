import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from image_landmark_detector import LandmarkDetectorFA
from face_segmentation.face_semantic_seg import FaceSemanticSegmentation
from PIL import Image

class VggFace2Processor:
    def __init__(self, input_folder, output_folder, n_images=5, face_seg_model_path=None, device="cuda:0"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.n_images = n_images
        self.face_seg_model_path = face_seg_model_path
        self.device = device
        
        self.lmk_detector = LandmarkDetectorFA(device=self.device)
        self.face_segmentation = FaceSemanticSegmentation(self.face_seg_model_path)

        os.makedirs(output_folder, exist_ok=True)
        self.save_fld = output_folder
        # self.rgb_save_fld = os.path.join(output_folder, "rgb_image")
        # self.seg_save_fld = os.path.join(output_folder, "segmentation_mask")
        # self.kpt_save_fld = os.path.join(output_folder, "keypoints")
        # os.makedirs(self.rgb_save_fld, exist_ok=True)
        # os.makedirs(self.seg_save_fld, exist_ok=True)
        # os.makedirs(self.kpt_save_fld, exist_ok=True)

    def detect_faces(self, image):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    def calculate_sharpness(self, image):
        return cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

    def calculate_face_size(self, face):
        return face.shape[0] * face.shape[1]

    def evaluate_faces(self, image, faces):
        face_data = [{"face": image[y:y+h, x:x+w], "sharpness": self.calculate_sharpness(image[y:y+h, x:x+w]), 
                      "size": self.calculate_face_size(image[y:y+h, x:x+w]), "bounding_box": (x, y, w, h)} for (x, y, w, h) in faces]
        return max(face_data, key=lambda x: (x["size"]))

    def select_n_images(self, image_folder):
        images = [i for i in os.listdir(image_folder) if i.lower().endswith('.jpg')]
        all_faces_dict = []
        
        for i in images:
            image = cv2.imread(os.path.join(image_folder, i))
            faces = self.detect_faces(image)
            if len(faces) == 1:
                face_eval = self.evaluate_faces(image, faces)
                face_eval["image_filename"] = i.split(".")[0]
                all_faces_dict.append(face_eval)
        
        return sorted(all_faces_dict, key=lambda x: (x["size"]), reverse=True)[:self.n_images]

    def process_dataset(self, n=-1):
        persons = os.listdir(self.input_folder)
        print(f"Found {len(persons)} individuals.")

        all_data_npy = []
        selected_faces_dict = {}
        if n < 0:
            n = len(persons)
        for person in tqdm(persons[:n], desc="Processing individuals", unit="person"):
            image_folder = os.path.join(self.input_folder, person)
            selected_faces = self.select_n_images(image_folder)
            
            data_line = [f"{person}/{sf['image_filename']}" for sf in selected_faces]
            all_data_npy.append(data_line)
            selected_faces_dict[person] = selected_faces

        print(len(all_data_npy))
        np.save(os.path.join(self.save_fld, "data_list.npy"), np.array(all_data_npy))

        # Generate plots for the selected faces
        fig, axes = plt.subplots(len(selected_faces_dict), self.n_images, figsize=(self.n_images * 2, len(selected_faces_dict) * 2))

        for i, (person, selected_faces) in enumerate(selected_faces_dict.items()):
            rgb_save_fld = os.path.join(output_folder, "rgb_image", person)
            seg_save_fld = os.path.join(output_folder, "segmentation_mask", person)
            kpt_save_fld = os.path.join(output_folder, "keypoints", person)
            os.makedirs(rgb_save_fld, exist_ok=True)
            os.makedirs(seg_save_fld, exist_ok=True)
            os.makedirs(kpt_save_fld, exist_ok=True)
            for j, sf in enumerate(selected_faces):
                image_path = os.path.join(self.input_folder, person, sf["image_filename"] + ".jpg")
                if os.path.isfile(image_path):
                    image = cv2.imread(image_path)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Detect landmarks and segment the face
                    bbox, lmk = self.lmk_detector.detect_landmarks(image_rgb)
                    mask = self.face_segmentation.get_segmented_skin(image_path)

                    # Save the segmentation mask
                    output_file_path = os.path.join(seg_save_fld, sf["image_filename"] + ".npy")
                    np.save(output_file_path, mask)

                    # Save mask as PNG
                    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)  # Ensure it's in uint8 format for PNG
                    output_file_path_png = os.path.join(seg_save_fld, sf["image_filename"] + ".png")
                    mask_image.save(output_file_path_png)

                    # Save the RGB image
                    rgb_image_filename = os.path.join(rgb_save_fld, sf["image_filename"] + ".jpg")
                    cv2.imwrite(rgb_image_filename, image_rgb)

                    # Save the landmarks
                    lmk_filename = os.path.join(kpt_save_fld, sf["image_filename"] + ".npy")
                    np.save(lmk_filename, lmk)


if __name__ == "__main__":
    input_folder = "/home/sakshi/projects/datasets/vggface2/train"
    output_folder = "data/vggface2"  
    face_seg_model_path = 'preprocess_dataset/face_segmentation/checkpoints/79999_iter.pth'

    processor = VggFace2Processor(input_folder=input_folder, output_folder=output_folder, n_images=5, face_seg_model_path=face_seg_model_path)
    processor.process_dataset(n=2)
