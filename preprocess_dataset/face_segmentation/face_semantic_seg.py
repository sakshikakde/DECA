import os
import sys
sys.path.append('/home/sakshi/projects/DollGPT')
import os.path as osp
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import cv2
from face_segmentation.model.bisenet import BiSeNet

class FaceSemanticSegmentation:
    def __init__(self, model_path, n_classes=19):
        # Initialize the model
        self.n_classes = n_classes
        self.model = BiSeNet(n_classes=n_classes)
        self.model.cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Preprocessing transformations
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def _open_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((512, 512), Image.BILINEAR)
        return img

    def _segment(self, image):
        img_tensor = self.to_tensor(image).unsqueeze(0).cuda()
        with torch.no_grad():
            out = self.model(img_tensor)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
        return parsing

    def get_segmented_skin(self, image_path):
        img = Image.open(image_path)
        original_size = img.size
        img = img.resize((512, 512), Image.BILINEAR)

        parsing = self._segment(img)

        # Initialize a mask for combined parts
        combined_mask = np.zeros_like(parsing, dtype=np.uint8)
        for part_id in range(self.n_classes):
            mask = (parsing == part_id).astype(np.uint8)
            
            # Combine masks for all parts except 0, 9, and 17
            if part_id not in [0, 9, 16, 17]:
                combined_mask = np.maximum(combined_mask, mask)

        combined_mask_resized = np.array(Image.fromarray(combined_mask.astype(np.uint8)).resize(original_size, Image.NEAREST))
        return combined_mask_resized


if __name__ == "__main__":
    model_path = osp.join('./face_segmentation/checkpoints/79999_iter.pth')  # Update with your model path
    face_segmentation = FaceSemanticSegmentation(model_path)

    # Example usage: Save segmented parts
    # face_segmentation.save_segmented_parts('/home/sakshi/projects/DollGPT/data/images/disney/thumb003.jpg')

    data_fld = "/home/sakshi/projects/DECA/data/vggface2"

    rgb_fld = os.path.join(data_fld, "rgb_image")
    seg_fld = os.path.join(data_fld, "segmentation_mask")

    data_list_file = os.path.join(data_fld, "data_list.npy")
    data_list = np.load(data_list_file)

    for data_line in data_list:
        for data in data_line:
            rgb_image_path = os.path.join(rgb_fld, data + ".jpg")
            if os.path.isfile(rgb_image_path):
                mask = face_segmentation.get_segmented_skin(rgb_image_path)

                output_file_path = os.path.join(seg_fld,  data + ".npy")
                np.save(output_file_path, mask)

                mask_image = Image.fromarray(mask.astype(np.uint8) * 255)  # Ensure it's in uint8 format for PNG
                output_file_path_png = os.path.join(seg_fld,  data + ".png")
                mask_image.save(output_file_path_png)