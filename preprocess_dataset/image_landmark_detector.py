import face_alignment
import numpy as np
from typing import Literal
import sys
sys.path.append('.')
from preprocess_dataset.utils.plot_utils import plot_landmarks_on_image

class LandmarkDetectorFA:
    def __init__(
        self,
        face_detector:Literal["sfd", "blazeface"]="sfd",
        device="cpu"
    ):
        print("Initialize FaceAlignment module...")
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, 
            face_detector=face_detector,
            flip_input=True, 
            device=device
        )


    def detect_landmarks(self, image):
        bbox = self.fa.face_detector.detect_from_image(image)

        if len(bbox) == 0:
            lmks = np.zeros([68, 3]) - 1  # set to -1 when landmarks is inavailable

        else:
            if len(bbox) > 1:
                # if multiple boxes detected, use the one with highest confidence
                bbox = [bbox[np.argmax(np.array(bbox)[:, -1])]]

            lmks = self.fa.get_landmarks_from_image(image, detected_faces=bbox)[0]
            lmks = np.concatenate([lmks, np.ones_like(lmks[:, :1])], axis=1)

            if (lmks[:, :2] == -1).sum() > 0:
                lmks[:, 2:] = 0.0
            else:
                lmks[:, 2:] = 1.0

        return bbox, lmks
    
if __name__ == "__main__":
    from skimage import io

    device="cuda:0"
    image_path = '/home/sakshi/projects/DollGPT/data/images/deepika/deepika_resized.jpg'
    image = io.imread(image_path)
    image_landmark_detector = LandmarkDetectorFA(device=device)
    bbox, image_landmarks = image_landmark_detector.detect_landmarks(image)
    fig = plot_landmarks_on_image(image, image_landmarks)
