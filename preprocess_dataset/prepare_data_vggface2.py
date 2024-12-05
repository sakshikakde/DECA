import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from image_landmark_detector import LandmarkDetectorFA
from face_segmentation.face_semantic_seg import FaceSemanticSegmentation
from PIL import Image
n = 5

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

def calculate_sharpness(image):
    return cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

def calculate_face_size(face):
    return face.shape[0] * face.shape[1]

def evaluate_faces(image, faces):
    face_data = [{"face": image[y:y+h, x:x+w], "sharpness": calculate_sharpness(image[y:y+h, x:x+w]), 
                  "size": calculate_face_size(image[y:y+h, x:x+w]), "bounding_box": (x, y, w, h)} for (x, y, w, h) in faces]
    return max(face_data, key=lambda x: (x["size"]))

def select_n_images(image_folder, n=5):
    images = [i for i in os.listdir(image_folder) if i.lower().endswith('.jpg')]
    all_faces_dict = []
    
    for i in images:
        image = cv2.imread(os.path.join(image_folder, i))
        faces = detect_faces(image)
        if len(faces) == 1:
            face_eval = evaluate_faces(image, faces)
            face_eval["image_filename"] = i.split(".")[0]
            all_faces_dict.append(face_eval)
    
    return sorted(all_faces_dict, key=lambda x: (x["size"]), reverse=True)[:n]

data_type = "vggface2"
root_data_fld = f"/home/sakshi/projects/datasets/{data_type}/train"
save_fld = f"data/{data_type}"
os.makedirs(save_fld, exist_ok=True)

persons = os.listdir(root_data_fld)
print(f"Found {len(persons)} individuals.")

all_data_npy = []
selected_faces_dict = {}

for person in tqdm(persons[:2], desc="Processing individuals", unit="person"):
    image_folder = os.path.join(root_data_fld, person)
    selected_faces = select_n_images(image_folder)
    
    data_line = [f"{person}/{sf['image_filename']}" for sf in selected_faces]
    all_data_npy.append(data_line)
    selected_faces_dict[person] = selected_faces

print(len(all_data_npy))
np.save(os.path.join(save_fld, "data_list.npy"), np.array(all_data_npy))


fig, axes = plt.subplots(len(selected_faces_dict), n, figsize=(n * 2, len(selected_faces_dict) * 2))

lmk_detector = LandmarkDetectorFA(device="cuda:0")
model_path = os.path.join('/home/sakshi/projects/DECA/preprocess_dataset/face_segmentation/checkpoints/79999_iter.pth')  # Update with your model path
face_segmentation = FaceSemanticSegmentation(model_path)

for i, (person, selected_faces) in enumerate(selected_faces_dict.items()):
    rgb_save_fld = os.path.join(save_fld, f"rgb_image/{person}")
    seg_save_fld = os.path.join(save_fld, f"segmentation_mask/{person}")
    kpt_save_fld = os.path.join(save_fld, f"keypoints/{person}")

    os.makedirs(rgb_save_fld, exist_ok=True)
    os.makedirs(seg_save_fld, exist_ok=True)
    os.makedirs(kpt_save_fld, exist_ok=True)
    
    for j, sf in enumerate(selected_faces):
        image_path = os.path.join(root_data_fld, person, sf["image_filename"] + ".jpg")

        if os.path.isfile(image_path):
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            bbox, lmk = lmk_detector.detect_landmarks(image_rgb)

            mask = face_segmentation.get_segmented_skin(image_path)

            # save mask as npy
            output_file_path = os.path.join(seg_save_fld, sf["image_filename"] + ".npy")
            np.save(output_file_path, mask)

            # save mask as png
            mask_image = Image.fromarray(mask.astype(np.uint8) * 255)  # Ensure it's in uint8 format for PNG
            output_file_path_png = os.path.join(seg_save_fld, sf["image_filename"] + ".png")
            mask_image.save(output_file_path_png)

            # Save the RGB image as .jpg
            rgb_image_filename = os.path.join(rgb_save_fld, sf["image_filename"] + ".jpg")
            cv2.imwrite(rgb_image_filename, image_rgb)
            
            # Save the landmarks as .npy
            lmk_filename = os.path.join(kpt_save_fld, sf["image_filename"] + ".npy")
            np.save(lmk_filename, lmk)

        # # save rgb
        # axes[i, j].imshow(image_rgb)
        # axes[i, j].axis('off')
        # if j == 0:
        #     axes[i, j].set_ylabel(person, fontsize=10)
            
# plt.tight_layout()
# plt.savefig(os.path.join(save_fld, "sample.png"))
