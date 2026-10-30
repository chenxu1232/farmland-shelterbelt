
import os
import matplotlib
import torch
import torch.nn.functional as F
import scipy.signal
import cv2
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.utils_metrics import compute_mIoU

# ==========================================
# Newly added imports for topological morphology and connected-component analysis
# Note: Ensure scikit-image is installed before running this code
# pip install scikit-image
# ==========================================
from skimage.morphology import skeletonize
from skimage.measure import label

matplotlib.use('Agg')
from matplotlib import pyplot as plt

# ---------------------------------------------------#
#   Core addition: Metrics for evaluating the linear structure of shelterbelts
# ---------------------------------------------------#
def calc_cldice(v_pred, v_true):
    """Calculate Centerline Dice (clDice)."""
    v_pred = v_pred.astype(bool)
    v_true = v_true.astype(bool)

    if np.sum(v_pred) == 0 and np.sum(v_true) == 0:
        return 1.0
    if np.sum(v_pred) == 0 or np.sum(v_true) == 0:
        return 0.0

    # Extract skeletons
    s_pred = skeletonize(v_pred)
    s_true = skeletonize(v_true)

    # Calculate topological overlap
    t_prec = np.sum(s_pred & v_true) / (np.sum(s_pred) + 1e-8)
    t_sens = np.sum(s_true & v_pred) / (np.sum(s_true) + 1e-8)

    cldice = 2 * t_prec * t_sens / (t_prec + t_sens + 1e-8)
    return cldice

def calc_cc_f1(v_pred, v_true, iou_threshold=0.1):
    """Calculate Connected Component F1 (CC-F1)."""
    v_pred = v_pred.astype(np.uint8)
    v_true = v_true.astype(np.uint8)

    # Label connected components (8-connectivity)
    labels_true, num_true = label(v_true, return_num=True, connectivity=2)
    labels_pred, num_pred = label(v_pred, return_num=True, connectivity=2)

    if num_true == 0 and num_pred == 0:
        return 1.0
    if num_true == 0 or num_pred == 0:
        return 0.0

    tp = 0
    # Match ground-truth connected components
    for i in range(1, num_true + 1):
        true_mask = (labels_true == i)
        intersecting_preds = np.unique(labels_pred[true_mask])
        intersecting_preds = intersecting_preds[intersecting_preds != 0]

        is_matched = False
        for p in intersecting_preds:
            pred_mask = (labels_pred == p)
            intersection = np.sum(true_mask & pred_mask)
            union = np.sum(true_mask | pred_mask)
            iou = intersection / (union + 1e-8)
            # Because shelterbelts are very narrow and elongated, a low IoU threshold (e.g., 0.1) is typically used to consider their main structure successfully detected
            if iou >= iou_threshold:
                is_matched = True
                break
        if is_matched:
            tp += 1

    precision = tp / (num_pred + 1e-8)
    recall = tp / (num_true + 1e-8)

    cc_f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return cc_f1


class LossHistory():
    def __init__(self, log_dir, model, input_shape, input_channels=8):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.writer     = SummaryWriter(self.log_dir)
        try:
            # -------------------------------------------------------#
            #   Modification 1: Correct the channel count of dummy_input
            #   Prevent channel-mismatch errors in TensorBoard add_graph
            # -------------------------------------------------------#
            dummy_input     = torch.randn(2, input_channels, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, \
            miou_out_path=".temp_miou_out", eval_flag=True, period=1):
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.image_ids          = image_ids
        self.dataset_path       = dataset_path
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.miou_out_path      = miou_out_path
        self.eval_flag          = eval_flag
        self.period             = period
        
        self.image_ids          = [image_id.split()[0] for image_id in image_ids]
        
        # -------------------------------------------------------------
        # Modification 3: Initialize all metric lists for compatibility with the CSV table in train.py
        # Store metrics as values in [0, 1]; train.py multiplies them by 100 for formatted output
        # -------------------------------------------------------------
        self.mious      = [0.0]
        self.mPAs       = [0.0]
        self.Accs       = [0.0]
        self.f1s        = [0.0]
        self.cldices    = [0.0]
        self.cc_f1s     = [0.0]
        self.epoches    = [0]
        
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_miou_png(self, image):
        """
        Process an 8-channel NumPy image using aspect-ratio-preserving letterbox resizing, predict, and restore the original size.
        """
        original_h, original_w = image.shape[:2]
        input_h, input_w       = self.input_shape

        # 1. Calculate the scaling factor
        scale = min(input_w/original_w, input_h/original_h)
        nw    = int(original_w * scale)
        nh    = int(original_h * scale)

        # 2. Resize the image (OpenCV supports multichannel resizing)
        image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)

        # ---------------------------------------------------------#
        #   Modification 2: Correct the padding value
        #   For float32 scientific data, zero-padding is safer than padding with 128
        # ---------------------------------------------------------#
        new_image = np.full((input_h, input_w, image.shape[2]), 0, dtype=np.float32)
        
        dx = (input_w - nw) // 2
        dy = (input_h - nh) // 2
        new_image[dy:dy+nh, dx:dx+nw, :] = image_resized

        # 4. HWC -> CHW, Expand Dims
        image_data = np.transpose(new_image, (2, 0, 1))
        image_data = np.expand_dims(image_data, 0)

        # 5. Predict
        with torch.no_grad():
            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()

            # Model inference
            pr = self.net(images)[0]
            # Softmax
            pr = F.softmax(pr.permute(1,2,0), dim=-1).cpu().numpy()
            
            # 6. Remove the letterbox padding
            pr = pr[dy:dy+nh, dx:dx+nw]

            # 7. Resize back to the original image dimensions
            pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            
            # Argmax
            pr = pr.argmax(axis=-1)
    
        image = Image.fromarray(np.uint8(pr))
        return image
    
    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net    = model_eval
            gt_dir      = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/")
            pred_dir    = os.path.join(self.miou_out_path, 'detection-results')
            
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
                
            print("Get miou.")
            for image_id in tqdm(self.image_ids):
                #-------------------------------#
                #   Modification: Load .npy images from files
                #-------------------------------#
                image_path  = os.path.join(self.dataset_path, "VOC2007/JPEGImages", image_id + ".npy")
                
                try:
                    image = np.load(image_path)
                    image = image.astype(np.float32)

                    # ==========================================================
                    # Critical fix: Apply the same normalization as in the data loader!
                    # Planet data must be divided by 10000.0
                    # ==========================================================
                    image = image / 10000.0
                    image = np.clip(image, 0, 1.0)
                    # ==========================================================
                    
                    # Obtain the predicted mask (PNG)
                    image_pred  = self.get_miou_png(image)
                    # Save the predicted mask
                    image_pred.save(os.path.join(pred_dir, image_id + ".png"))
                except Exception as e:
                    print(f"Error evaluating {image_id}: {e}")
                    continue
                        
            print("Calculate standard metrics...")
            # Calculate standard metrics such as mIoU
            hist, IoUs, PA_Recall, Precision, kappa, f1_scores, dsc_scores, macc, flops = \
            compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, None)
            
            # Record basic metrics as decimal values
            temp_miou = np.nanmean(IoUs)
            temp_mpa  = np.nanmean(PA_Recall)
            temp_acc  = np.nanmean(macc) if isinstance(macc, (list, np.ndarray)) else macc
            temp_f1   = np.nanmean(f1_scores) if f1_scores is not None else 0

            # -------------------------------------------------------------
            # New step: Calculate clDice and CC-F1 specifically for shelterbelts (Class 1)
            # -------------------------------------------------------------
            print("Calculate topology metrics (clDice & CC-F1) for shelterbelts...")
            cldice_list = []
            ccf1_list = []
            for image_id in tqdm(self.image_ids, desc="Topology Analysis"):
                gt_path = os.path.join(gt_dir, image_id + ".png")
                pred_path = os.path.join(pred_dir, image_id + ".png")

                try:
                    gt_img = np.array(Image.open(gt_path))
                    pred_img = np.array(Image.open(pred_path))

                    # Extract binary masks for Class 1 (the target shelterbelt class)
                    gt_binary = (gt_img == 1).astype(np.uint8)
                    pred_binary = (pred_img == 1).astype(np.uint8)

                    cldice_list.append(calc_cldice(pred_binary, gt_binary))
                    ccf1_list.append(calc_cc_f1(pred_binary, gt_binary, iou_threshold=0.1))
                except Exception as e:
                    # Skip missing or corrupted images
                    pass
            
            temp_cldice = np.nanmean(cldice_list) if len(cldice_list) > 0 else 0.0
            temp_ccf1 = np.nanmean(ccf1_list) if len(ccf1_list) > 0 else 0.0
            
            # Save metrics as class attributes for train.py to access
            self.mious.append(temp_miou)
            self.mPAs.append(temp_mpa)
            self.Accs.append(temp_acc)
            self.f1s.append(temp_f1)
            self.cldices.append(temp_cldice)
            self.cc_f1s.append(temp_ccf1)
            self.epoches.append(epoch)

            # Backward compatibility: Save mIoU to a text file
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou * 100))
                f.write("\n")
            
            # Plot the mIoU curve as before
            plt.figure()
            plt.plot(self.epoches, [m * 100 for m in self.mious], 'red', linewidth = 2, label='train miou')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou (%)')
            plt.title('Validation mIoU Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            print("Evaluation done.")
            shutil.rmtree(self.miou_out_path)
