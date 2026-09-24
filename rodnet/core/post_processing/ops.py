import numpy as np



def detect_peaks(image, threshold=0.3):
    peaks_row = []
    peaks_col = []
    height, width = image.shape
    for h in range(1, height - 1):
        for w in range(2, width - 2):
            area = image[h - 1:h + 2, w - 2:w + 3]
            center = image[h, w]
            flag = np.where(area >= center)
            if flag[0].shape[0] == 1 and center > threshold:
                peaks_row.append(h)
                peaks_col.append(w)

    return peaks_row, peaks_col


def detect_peaks_fast(image, threshold=0.3):
    height, width = image.shape
    if height < 3 or width < 5:
        return [], []
    center = image[1:height - 1, 2:width - 2]
    neighbors = []
    for dh in (-1, 0, 1):
        for dw in (-2, -1, 0, 1, 2):
            if dh == 0 and dw == 0:
                continue
            neighbors.append(
                image[
                    1 + dh:height - 1 + dh,
                    2 + dw:width - 2 + dw
                ]
            )
    neighbor_max = np.maximum.reduce(neighbors)
    peak_mask = (center > threshold) & (center > neighbor_max)
    peaks_row, peaks_col = np.nonzero(peak_mask)
    return (peaks_row + 1).tolist(), (peaks_col + 2).tolist()