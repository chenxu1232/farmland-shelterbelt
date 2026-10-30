
# Complete Eight-Channel Data Processing Pipeline
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(SegmentationDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        # 1. Load eight-channel data
        image_path = os.path.join(self.dataset_path, "VOC2007/JPEGImages", name + ".npy")
        try:
            jpg = np.load(image_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {image_path}")
        
        # The pixel values are approximately in the range of 0-5800, with uint16 data type
        jpg = jpg.astype(np.float32)

        # ---------------------------------------------------
        # Critical Fix: Normalization
        # ---------------------------------------------------
        # Since the maximum value is approximately 5800, this is typical reflectance data (scale factor = 10000)
        # Divide by 10000 to map the values to approximately 0.0-0.6, providing a suitable input range for neural networks
        jpg = jpg / 10000.0
        
        # Apply clipping to prevent a small number of noisy pixels from exceeding 1.0
        # Although the observed maximum is only about 5800, this operation provides an additional safeguard
        jpg = np.clip(jpg, 0, 1.0)

        # ---------------------------------------------------
        # 2. Load segmentation labels
        # ---------------------------------------------------
        png_path = os.path.join(self.dataset_path, "VOC2007/SegmentationClass", name + ".png")
        png = Image.open(png_path)
        
        # ... (Keep the subsequent data augmentation and transpose operations unchanged) ...
        
        jpg, png = self.get_random_data(jpg, png, self.input_shape, random=self.train)
        
        jpg = np.transpose(jpg, [2, 0, 1])
        png = np.array(png)
        
        # For binary segmentation, ensure that label values are valid
        if self.num_classes == 2:
             png[png >= self.num_classes] = 0
        else:
             png[png >= self.num_classes] = self.num_classes

        seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, random=True):
        """
        Data augmentation for eight-channel images, with HSV transformations removed.
        """
        if isinstance(label, Image.Image):
            label = np.array(label)

        iw, ih  = image.shape[1], image.shape[0]
        h, w    = input_shape

        if not random:
            # Validation set: Simple resizing and padding while preserving the aspect ratio
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            
            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
            label = cv2.resize(label, (nw, nh), interpolation=cv2.INTER_NEAREST)

            # Background padding: Either 0 or 128 can be used, but 0 is chosen for consistency with normalized data
            # For unnormalized uint8 images, a padding value of 128 may be appropriate
            # For floating-point data, zero-padding is generally recommended
            # Here, zero-padding is used as a precaution
            new_image = np.full((h, w, image.shape[2]), 0, dtype=np.float32)
            new_label = np.full((h, w), 0, dtype=np.uint8) # The background label is typically 0

            dx = (w-nw)//2
            dy = (h-nh)//2
            
            new_image[dy:dy+nh, dx:dx+nw, :] = image
            new_label[dy:dy+nh, dx:dx+nw] = label

            return new_image, new_label

        # ==========================================================
        # Training Data Augmentation
        # ==========================================================
        
        # 1. Resize and jitter (aspect-ratio jittering and scaling)
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.5, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        
        # Recent versions of OpenCV generally support multichannel resizing without splitting channels
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, (nw, nh), interpolation=cv2.INTER_NEAREST)

        # 2. Flip (horizontal flipping)
        flip = self.rand()<.5
        if flip: 
            image = cv2.flip(image, 1) # 1: Horizontal flip
            label = cv2.flip(label, 1)

        # 3. Place on canvas (padding/cropping)
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        
        new_image = np.full((h, w, image.shape[2]), 0, dtype=np.float32)
        new_label = np.full((h, w), 0, dtype=np.uint8)
        
        # Safe assignment to handle cases where cropping extends beyond the canvas boundaries
        if nw > w or nh > h:
             # If the resized image exceeds the target dimensions, resize it directly to the target size
             image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
             label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
             new_image = image
             new_label = label
        else:
            # Otherwise, place the resized image on the canvas
            new_image[dy:dy+nh, dx:dx+nw, :] = image
            new_label[dy:dy+nh, dx:dx+nw] = label

        image = new_image
        label = new_label

        # 4. Blur (Gaussian blur)
        # Fix: cv2.GaussianBlur may not support more than four channels in some cases
        # Process each channel separately for improved robustness
        blur = self.rand() < 0.25
        if blur: 
            if image.shape[2] > 4:
                channels = cv2.split(image)
                blurred_channels = []
                for ch in channels:
                    ch_blur = cv2.GaussianBlur(ch, (5, 5), 0)
                    blurred_channels.append(ch_blur)
                image = cv2.merge(blurred_channels)
            else:
                image = cv2.GaussianBlur(image, (5, 5), 0)

        # 5. Rotate (rotation)
        rotate = self.rand() < 0.25
        if rotate: 
            center      = (w // 2, h // 2)
            rotation    = np.random.randint(-10, 11)
            # Obtain the rotation matrix
            M           = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            
            # --- Core modification: Perform rotation separately for each channel ---
            # The borderValue parameter of warpAffine does not support tuples containing more than four values
            if image.shape[2] > 4:
                channels = cv2.split(image)
                warped_channels = []
                for ch in channels:
                    # Apply single-channel warping with borderValue=0 (assuming normalized float32 data)
                    warped_ch = cv2.warpAffine(ch, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=0)
                    warped_channels.append(warped_ch)
                image = cv2.merge(warped_channels)
            else:
                # Images with four or fewer channels can be processed directly
                image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(0,)*image.shape[2])
            
            # Process the segmentation labels
            label = cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

        return image, label


def seg_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels
