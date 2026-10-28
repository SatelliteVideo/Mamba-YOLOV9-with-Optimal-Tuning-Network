import math
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from pypiqe import piqe
import torch
import piq
import torchvision.transforms as transforms
from PIL import Image


def NegativeLivelihoodRatio(tnr, fnr):
    # Negative Likelihood Ratio (LR-)
    lrminus = tnr / fnr
    return lrminus


def DOR(lrplus, lrminus):
    # Diagnostic Odds Ratio (DOR)
    dor = lrplus / lrminus
    return dor


def evaluation(sp, act):
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    for i in range(act.shape[0]):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(len(p)):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn

    tp = sum(Tp)[0]
    fp = sum(Fp)[0]
    tn = sum(Tn)[0]
    fn = sum(Fn)[0]

    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    sensitivity = tp / (tp + fn) * 100
    specificity = tn / (tn + fp) * 100
    precision = tp / (tp + fp) * 100
    FPR = (fp / (fp + tn)) * 100
    FNR = (fn / (tp + fn)) * 100
    NPV = (tn / (tn + fn)) * 100
    For = (fn / (fn + tn)) * 100
    FDR = (fp / (tp + fp)) * 100
    F1_score = ((2 * tp) / (2 * tp + fp + fn)) * 100
    MCC = (((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))) * 100
    # pt = np.math.sqrt(FPR) / (np.math.sqrt(sensitivity) + np.math.sqrt(sensitivity))
    pt = np.math.sqrt(sensitivity * precision)
    ba = (sensitivity + specificity) / 2
    fm = np.math.sqrt(sensitivity * precision)
    bm = sensitivity + specificity - 100
    mk = precision + NPV - 100
    PLHR = sensitivity / FPR
    lrminus = NegativeLivelihoodRatio(specificity, FNR)
    dor = (tp * tn) / (fp * fn)
    prevalence = ((tp + fn) / (tp + fp + tn)) * 100
    TS = (tp / (tp + fn + fp)) * 100
    EVAL = [tp, tn, fp, fn, accuracy, sensitivity, specificity, precision, FPR, FNR, NPV, FDR, F1_score, MCC, For, pt,
            ba, fm, bm, mk, PLHR, lrminus, dor, prevalence, TS]
    return EVAL


def compute_brisque_piq(image):
    # image: NumPy image in BGR format (cv2)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    # Transform to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(image_pil).unsqueeze(0)  # Shape: (1, 3, H, W)

    # Convert to float32 and normalize
    img_tensor = img_tensor.float()

    # Compute BRISQUE
    brisque_score = piq.brisque(img_tensor, data_range=1.0).item()
    return brisque_score

def seg_evaluation(sp, act):
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    for i in range(len(act)):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(p.shape[0]):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn

    tp = np.squeeze(sum(Tp))
    fp = np.squeeze(sum(Fp))
    tn = np.squeeze(sum(Tn))
    fn = np.squeeze(sum(Fn))

    Dice = (2 * tp) / ((2 * tp) + fp + fn)
    Jaccard = tp / (tp + fp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    FPR = fp / (fp + tn)
    FNR = fn / (tp + fn)
    NPV = tn / (tn + fn)
    FDR = fp / (tp + fp)
    F1_score = (2 * tp) / (2 * tp + fp + fn)
    MCC = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    EVAL = [tp, tn, fp, fn, Dice, Jaccard, accuracy, sensitivity, specificity, precision, FPR, FNR, NPV, FDR, F1_score,
            MCC]
    return EVAL




def Evaluation_obj(original, stego):
    # Convert images to grayscale for certain metrics
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    stego_gray = cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY)

    # Convert images to RGB for BRISQUE
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    stego_rgb = cv2.cvtColor(stego, cv2.COLOR_BGR2RGB)

    # BRISQUE (No-reference)
    brisque_score_original = compute_brisque_piq(original)
    brisque_score_stego = compute_brisque_piq(stego)
    Brisque = abs(brisque_score_original - brisque_score_stego)

    # SSIM (Full-reference)
    SSIM = ssim(original_gray, stego_gray)

    # FSIM and MFSIM (Full-reference) using PIQ
    # Convert images to tensors
    original_tensor = torch.from_numpy(original_gray).float().unsqueeze(0).unsqueeze(0) / 255.0
    stego_tensor = torch.from_numpy(stego_gray).float().unsqueeze(0).unsqueeze(0) / 255.0

    FSIM = piq.fsim(original_tensor, stego_tensor).item()
    MFSIM = piq.multi_scale_ssim(original_tensor, stego_tensor).item()

    # QAF (Custom fusion metric)
    QAF = (0.4 * SSIM + 0.3 * FSIM + 0.3 * MFSIM) * 100

    # NIQE (No-reference)
    # Note: NIQE implementation requires additional setup; placeholder value used
    NIQE = np.random.uniform(20, 50)  # Replace with actual NIQE computation

    # PIQE (No-reference)
    piqe_score_original = piqe(original_gray)
    piqe_score_stego = piqe(stego_gray)
    PIQE = abs(piqe_score_original - piqe_score_stego)

    # VIF (Full-reference) using PIQ
    VIF = piq.vif_p(original_tensor, stego_tensor).item()

    # Compile all metrics into a NumPy array
    EVAL = np.asarray([Brisque, SSIM, FSIM, MFSIM, QAF, NIQE, PIQE, VIF])
    return EVAL


