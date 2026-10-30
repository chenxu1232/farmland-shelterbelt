
# segformer.py
import colorsys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.segformer import SegFormer
from utils.utils import show_config


class SegFormer_Segmentation(object):
    _defaults = {
        "model_path": "logs/最强ep090-loss0.052-val_loss0.152.pth",
        "num_classes": 2,

        # Eight-channel input data
        "input_channels": 8,

        "phi": "b5",
        "input_shape": [512, 512],

        "mix_type": 1,
        "cuda": True,

        # Normalization modes: auto / div / none
        # auto: If max > 1.5, assume data are scaled (e.g., 0-10000)
        # and normalize by dividing by norm_div
        "norm_mode": "auto",
        "norm_div": 10000.0,

        # Visualization settings (stretch the first three channels into RGB)
        "vis_percentile_low": 2,
        "vis_percentile_high": 98,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # Color palette
        if self.num_classes <= 21:
            self.colors = [
                (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
                (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0),
                (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0),
                (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                (128, 64, 12)
            ]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            rgb = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), rgb))

        self.generate()
        show_config(**{k: getattr(self, k) for k in self._defaults.keys()})

    # -------------------------- Utility: Convert input to HWC format --------------------------
    def _to_hwc(self, x):
        # Support direct input of .npy file paths
        if isinstance(x, str) and x.lower().endswith(".npy"):
            x = np.load(x)

        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if x.ndim != 3:
            raise ValueError(f"Input must be a 3D array, but received shape={x.shape}")

        # (H, W, C)
        if x.shape[-1] == self.input_channels:
            return x

        # (C, H, W)
        if x.shape[0] == self.input_channels:
            return np.transpose(x, (1, 2, 0))

        raise ValueError(
            f"Channel dimension mismatch: expected (H,W,{self.input_channels}) "
            f"or ({self.input_channels},H,W), but received {x.shape}"
        )

    # -------------------------- Utility: Normalization --------------------------
    def _normalize(self, x_hwc: np.ndarray) -> np.ndarray:
        x = x_hwc.astype(np.float32)

        if self.norm_mode == "none":
            pass
        elif self.norm_mode == "div":
            x = x / float(self.norm_div)
        elif self.norm_mode == "auto":
            mx = float(np.max(x))
            if mx > 1.5:  # Assume scaled data such as 0-10000 or 0-5000
                x = x / float(self.norm_div)
        else:
            raise ValueError(f"Unknown norm_mode: {self.norm_mode}")

        return np.clip(x, 0.0, 1.0)

    # -------------------------- Utility: RGB Visualization (First Three Channels) --------------------------
    def _make_vis_rgb(self, x_hwc: np.ndarray) -> Image.Image:
        if x_hwc.shape[2] < 3:
            arr = x_hwc.astype(np.float32)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255.0
            return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        rgb = x_hwc[:, :, :3].astype(np.float32)
        lo = np.percentile(rgb, self.vis_percentile_low, axis=(0, 1))
        hi = np.percentile(rgb, self.vis_percentile_high, axis=(0, 1))
        rgb = (rgb - lo) / (hi - lo + 1e-6) * 255.0
        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))

    # -------------------------- Utility: Letterbox Resizing (Multichannel Support) --------------------------
    def _letterbox(self, x_hwc: np.ndarray):
        oh, ow = x_hwc.shape[:2]
        ih, iw = self.input_shape

        scale = min(iw / ow, ih / oh)
        nw, nh = int(ow * scale), int(oh * scale)

        x_rs = cv2.resize(x_hwc, (nw, nh), interpolation=cv2.INTER_CUBIC)
        new_img = np.zeros((ih, iw, self.input_channels), dtype=np.float32)

        dx = (iw - nw) // 2
        dy = (ih - nh) // 2
        new_img[dy:dy + nh, dx:dx + nw, :] = x_rs
        return new_img, dx, dy, nw, nh, ow, oh

    # -------------------------- Model Loading --------------------------
    def generate(self, onnx=False):
        self.net = SegFormer(
            num_classes=self.num_classes,
            phi=self.phi,
            pretrained=False,
            in_chans=self.input_channels,   # Eight-channel input configuration
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(self.model_path, map_location=device)

        # Support checkpoints with nested state dictionaries
        if isinstance(ckpt, dict):
            for k in ["state_dict", "model", "model_state_dict", "net", "network", "weights"]:
                if k in ckpt and isinstance(ckpt[k], dict):
                    ckpt = ckpt[k]
                    break

        incompatible = self.net.load_state_dict(ckpt, strict=False)
        self.net = self.net.eval()

        print(f'{self.model_path} model, and classes loaded.')
        print("Missing keys:", len(incompatible.missing_keys))
        print("Unexpected keys:", len(incompatible.unexpected_keys))

        if not onnx and self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    # -------------------------- Inference (Supports Eight-Channel NumPy Data) --------------------------
    def detect_image(self, image, count=False, name_classes=None):
        # 1) Load input and convert to HWC format
        x = self._to_hwc(image)                # (H, W, 8)
        old_img = self._make_vis_rgb(x)        # Visualization using the first three channels

        # 2) Normalize the input
        x = self._normalize(x)

        # 3) Apply letterbox resizing
        new_img, dx, dy, nw, nh, ow, oh = self._letterbox(x)

        # 4) Convert to tensor format (BCHW)
        image_data = np.transpose(new_img, (2, 0, 1))[None, ...].astype(np.float32)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]  # (C, H, W)
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()  # (H, W, C)

            # Remove padding
            pr = pr[dy:dy + nh, dx:dx + nw]

            # Resize predictions to the original image dimensions
            pr = cv2.resize(pr, (ow, oh), interpolation=cv2.INTER_LINEAR)

            # Obtain predicted class labels
            pr = pr.argmax(axis=-1).astype(np.uint8)

            # Debugging statistics
            unique, counts = np.unique(pr, return_counts=True)
            print(f"Prediction statistics: {dict(zip(unique.tolist(), counts.tolist()))}")

        # Pixel counting (optional)
        if count:
            classes_nums = np.zeros([self.num_classes], dtype=np.int64)
            total_points_num = oh * ow
            print('-' * 63)
            print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num = int(np.sum(pr == i))
                ratio = num / total_points_num * 100
                if num > 0:
                    cname = str(name_classes[i]) if name_classes is not None else str(i)
                    print("|%25s | %15s | %14.2f%%|" % (cname, str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        # Generate visualization results
        if self.mix_type == 0:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [oh, ow, -1])
            mask_img = Image.fromarray(seg_img.astype(np.uint8))
            image_out = Image.blend(old_img.convert("RGB"), mask_img.convert("RGB"), 0.7)

        elif self.mix_type == 1:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [oh, ow, -1])
            image_out = Image.fromarray(seg_img.astype(np.uint8))

        elif self.mix_type == 2:
            old_np = np.array(old_img.convert("RGB"), np.float32)
            seg_np = (np.expand_dims(pr != 0, -1) * old_np).astype('uint8')
            image_out = Image.fromarray(seg_np)

        else:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [oh, ow, -1])
            image_out = Image.fromarray(seg_img.astype(np.uint8))

        return image_out

    # -------------------------- FPS Measurement --------------------------
    def get_FPS(self, image, test_interval=10):
        x = self._to_hwc(image)
        x = self._normalize(x)
        new_img, dx, dy, nw, nh, ow, oh = self._letterbox(x)
        image_data = np.transpose(new_img, (2, 0, 1))[None, ...].astype(np.float32)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            _ = self.net(images)[0]

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                _ = self.net(images)[0]
        t2 = time.time()
        return (t2 - t1) / test_interval

    # -------------------------- ONNX Export (Eight-Channel Input) --------------------------
    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im = torch.zeros(1, self.input_channels, *self.input_shape).to('cpu')
        input_layer_names = ["images"]
        output_layer_names = ["output"]

        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(
            self.net,
            im,
            f=model_path,
            verbose=False,
            opset_version=12,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=input_layer_names,
            output_names=output_layer_names,
            dynamic_axes=None
        )

        model_onnx = onnx.load(model_path)
        onnx.checker.check_model(model_onnx)

        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(model_onnx, dynamic_input_shape=False, input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    # -------------------------- mIoU Evaluation Output (Label Mask) --------------------------
    def get_miou_png(self, image):
        x = self._to_hwc(image)
        x = self._normalize(x)
        new_img, dx, dy, nw, nh, ow, oh = self._letterbox(x)
        image_data = np.transpose(new_img, (2, 0, 1))[None, ...].astype(np.float32)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[dy:dy + nh, dx:dx + nw]
            pr = cv2.resize(pr, (ow, oh), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1).astype(np.uint8)

        return Image.fromarray(pr)
