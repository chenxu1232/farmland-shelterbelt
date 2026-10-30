import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from thop import profile  # Used to calculate FLOPs


def f_score(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   Calculate the Dice coefficient
    # --------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score


# Calculate the Kappa coefficient
def compute_kappa(hist):
    n_classes = hist.shape[0]
    sum0 = np.sum(hist, axis=0)
    sum1 = np.sum(hist, axis=1)
    total = np.sum(hist)

    # Calculate the number of correctly classified pixels on the diagonal
    po = np.trace(hist) / total

    # Calculate chance agreement
    pe = np.sum(sum0 * sum1) / (total * total)

    kappa = (po - pe) / (1 - pe)
    return kappa


# Calculate the F1-score
def compute_f1_score(hist):
    n_classes = hist.shape[0]
    f1_scores = []

    for i in range(n_classes):
        tp = hist[i, i]
        fp = np.sum(hist[:, i]) - tp
        fn = np.sum(hist[i, :]) - tp

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)

        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        f1_scores.append(f1)

    return np.array(f1_scores)


# Calculate the Dice Similarity Coefficient (DSC)
def compute_dsc(hist):
    n_classes = hist.shape[0]
    dsc_scores = []

    for i in range(n_classes):
        tp = hist[i, i]
        fp = np.sum(hist[:, i]) - tp
        fn = np.sum(hist[i, :]) - tp

        dsc = 2 * tp / (2 * tp + fp + fn + 1e-10)
        dsc_scores.append(dsc)

    return np.array(dsc_scores)


# Calculate mean accuracy (mAcc)
def compute_macc(hist):
    n_classes = hist.shape[0]
    acc_per_class = np.diag(hist) / (np.sum(hist, axis=1) + 1e-10)
    return np.mean(acc_per_class)


# ---------------------------------------------------
# Modification: Change the default input_size channel count from 3 to 8
# ---------------------------------------------------
def compute_flops(model, input_size=(1, 8, 512, 512)):
    device = next(model.parameters()).device
    # Generate a dummy eight-channel input to estimate model computation
    dummy_input = torch.randn(input_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    return flops / 1e9  # Convert to GFLOPs


# Assume the label image has width W and height H
def fast_hist(a, b, n):
    # --------------------------------------------------------------------------------#
    #   a: Flattened ground-truth labels of shape (H*W,); b: Flattened predictions of shape (H*W,)
    # --------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    # --------------------------------------------------------------------------------#
    #   np.bincount counts occurrences of the n**2 possible indices from 0 to n**2-1; reshape to (n, n)
    #   Diagonal entries represent correctly classified pixels
    # --------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)


def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)


def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None, model=None):
    print('Num classes', num_classes)
    # -----------------------------------------#
    #   Initialize a zero-valued confusion matrix
    # -----------------------------------------#
    hist = np.zeros((num_classes, num_classes))

    # ------------------------------------------------#
    #   Build the list of validation ground-truth label paths
    #   Build the list of validation prediction paths
    # ------------------------------------------------#
    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]

    # ------------------------------------------------#
    #   Read each prediction and ground-truth label pair
    # ------------------------------------------------#
    for ind in range(len(gt_imgs)):
        # ------------------------------------------------#
        #   Read a predicted segmentation mask and convert it to a NumPy array
        #   Modification: Add .convert('L') to read a grayscale image
        # ------------------------------------------------#
        pred = np.array(Image.open(pred_imgs[ind]).convert('L'))
        # ------------------------------------------------#
        #   Read the corresponding ground-truth label and convert it to a NumPy array
        #   Modification: Add .convert('L') to read a grayscale image
        # ------------------------------------------------#
        label = np.array(Image.open(gt_imgs[ind]).convert('L'))

        # Skip the image if the prediction and label have different flattened lengths
        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        # ------------------------------------------------#
        #   Compute the confusion matrix for an image and accumulate it
        # ------------------------------------------------#
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        # Every 10 images, report the current mean IoU and other metrics across all classes
        if name_classes is not None and ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                ind,
                len(gt_imgs),
                100 * np.nanmean(per_class_iu(hist)),
                100 * np.nanmean(per_class_PA_Recall(hist)),
                100 * per_Accuracy(hist)
            )
            )

    # ------------------------------------------------#
    #   Calculate per-class IoU values over all validation images
    # ------------------------------------------------#
    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)

    # Calculate additional metrics
    kappa = compute_kappa(hist)
    f1_scores = compute_f1_score(hist)
    dsc_scores = compute_dsc(hist)
    macc = compute_macc(hist)

    # Calculate FLOPs if a model is provided
    flops = None
    if model is not None:
        flops = compute_flops(model)

    # ------------------------------------------------#
    #   Print IoU and related metrics for each class
    # ------------------------------------------------#
    if name_classes is not None:
        for ind_class in range(num_classes):
            print('===>' + name_classes[ind_class] +
                  ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) +
                  '; Recall-' + str(round(PA_Recall[ind_class] * 100, 2)) +
                  '; Precision-' + str(round(Precision[ind_class] * 100, 2)) +
                  '; F1-score-' + str(round(f1_scores[ind_class] * 100, 2)) +
                  '; DSC-' + str(round(dsc_scores[ind_class] * 100, 2)))

    # -----------------------------------------------------------------#
    #   Compute the mean IoU across classes over the validation set, ignoring NaN values
    # -----------------------------------------------------------------#
    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) +
          '%; mPA: ' + str(round(np.nanmean(PA_Recall) * 100, 2)) +
          '%; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)) +
          '%; Kappa: ' + str(round(kappa * 100, 2)) +
          '%; mAcc: ' + str(round(macc * 100, 2)) +
          '%; mF1-score: ' + str(round(np.nanmean(f1_scores) * 100, 2)) +
          '%; mDSC: ' + str(round(np.nanmean(dsc_scores) * 100, 2)))

    if flops is not None:
        print('===> FLOPs: {:.2f}G'.format(flops))

    return np.array(hist, int), IoUs, PA_Recall, Precision, kappa, f1_scores, dsc_scores, macc, flops


def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size=12, plt_show=True):
    fig = plt.gcf()
    axes = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val)
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values) - 1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()


def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, kappa, f1_scores, dsc_scores, macc, flops,
                 name_classes, tick_font_size=12):
    # Save the existing metric plots
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.4f}%".format(np.nanmean(IoUs) * 100), "Intersection over Union",
                   os.path.join(miou_out_path, "mIoU.png"), tick_font_size=tick_font_size, plt_show=True)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.4f}%".format(np.nanmean(PA_Recall) * 100), "Pixel Accuracy",
                   os.path.join(miou_out_path, "mPA.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))

    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.4f}%".format(np.nanmean(PA_Recall) * 100), "Recall",
                   os.path.join(miou_out_path, "Recall.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.4f}%".format(np.nanmean(Precision) * 100), "Precision",
                   os.path.join(miou_out_path, "Precision.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    # Save plots for the additional metrics
    draw_plot_func(f1_scores, name_classes, "mF1-score = {0:.4f}%".format(np.nanmean(f1_scores) * 100), "F1-score",
                   os.path.join(miou_out_path, "F1_score.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save F1-score out to " + os.path.join(miou_out_path, "F1_score.png"))

    draw_plot_func(dsc_scores, name_classes, "mDSC = {0:.4f}%".format(np.nanmean(dsc_scores) * 100),
                   "Dice Similarity Coefficient",
                   os.path.join(miou_out_path, "DSC.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save DSC out to " + os.path.join(miou_out_path, "DSC.png"))

    # Save all metrics to a CSV file
    with open(os.path.join(miou_out_path, "all_metrics.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['mIoU', f'{np.nanmean(IoUs) * 100:.2f}%'])
        writer.writerow(['mPA', f'{np.nanmean(PA_Recall) * 100:.2f}%'])
        writer.writerow(['Accuracy', f'{per_Accuracy(hist) * 100:.2f}%'])
        writer.writerow(['Kappa', f'{kappa * 100:.2f}%'])
        writer.writerow(['mAcc', f'{macc * 100:.2f}%'])
        writer.writerow(['mF1-score', f'{np.nanmean(f1_scores) * 100:.2f}%'])
        writer.writerow(['mDSC', f'{np.nanmean(dsc_scores) * 100:.2f}%'])
        if flops is not None:
            writer.writerow(['FLOPs', f'{flops:.2f}G'])

    print("Save all metrics out to " + os.path.join(miou_out_path, "all_metrics.csv"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer_list = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
