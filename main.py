import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.pylab import cm
from HdrToolbox import HdrToolbox

def load_images(path_test: str):
    filenames = []
    exposure_times = []
    with open(os.path.join(path_test, 'image_list.txt')) as f:
        for line in f:
            if line.startswith('#'):
                continue
            filename, exposure, *_ = line.split()
            filenames.append(os.path.join(path_test, filename))
            exposure_times.append(float(exposure))

    im_shape = cv.imread(filenames[0]).shape
    stack = np.zeros((len(filenames), *im_shape), dtype=np.uint8)
    for i, filename in enumerate(filenames):
        im = cv.imread(filename)
        stack[i] = im
    return stack, exposure_times

if __name__ == "__main__":
    stack, exps = load_images('imgs/night')
    l_hdr = HdrToolbox(weight_type='sin')
    g_curve = None
    img_rad = np.zeros(stack.shape[1:], dtype=np.float64)
    hdr_rad = np.zeros(stack.shape[1:], dtype=np.float64)

    for c in range(stack.shape[3]):
        stack_layer = stack[:, :, :, c]
        sample = l_hdr.sample_intensity(stack_layer)
        print(f'estimate curve channel {c}')
        curve, lnE = l_hdr.estimate_curve(sample, exps, 100)
        if g_curve is None:
            g_curve = np.zeros((curve.shape[0], stack.shape[3]), dtype=np.float64)
        g_curve[:, c] = curve
        print(f'compute radiance channel {c}')
        img_rad[:, :, c] = l_hdr.compute_radiance(stack_layer, exps, curve)
        hdr_rad[:, :, c] = cv.normalize(img_rad[:, :, c], None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    output = l_hdr.convert_rad_to_hdr(hdr_rad)
    ldr_exposure = 250  # Example exposure time for LDR synthesis
    print('synthetize LDR image')
    LDR = l_hdr.synthetize_ldr_image(img_rad, g_curve, ldr_exposure)

    plt.figure(figsize=(14, 8))
    plt.subplot(2, 2, 1)
    pI = np.arange(256)
    for i, color in enumerate('rgb'):
        plt.plot(g_curve[:, i], pI, color)
    plt.ylabel('pixel value')
    plt.xlabel('log exposure')
    plt.title('Response curve')
    plt.grid()

    colorize = cm.jet
    cmap = np.float32(cv.cvtColor(np.uint8(hdr_rad), cv.COLOR_BGR2GRAY) / 255.)
    cmap = colorize(cmap)
    plt.subplot(2, 2, 2)
    plt.axis('off')
    plt.imshow(np.uint8(cmap * 255.), aspect='equal')
    plt.title('HDR Radiance Map')

    plt.subplot(2, 2, 3)
    plt.axis('off')
    plt.imshow(output, aspect='equal')
    plt.title('HDR Image')

    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.imshow(LDR, aspect='equal')
    plt.title(f'LDR Image (Exposure: {ldr_exposure})')

    plt.show()