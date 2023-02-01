# System libs
import csv, numpy, scipy.io, PIL.Image
from semseg_model.utils import colorEncode, labelEncode

colors = scipy.io.loadmat('data/color150.mat')['colors']
names = {}
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

def visualize_result(img, pred, index=None):
    # filter prediction class if requested
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
        print(f'{names[index+1]}:')
        
    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(numpy.uint8)
    # Label
    label = labelEncode(pred, names, colors)

    # aggregate images and save
    im_vis = numpy.concatenate((img, pred_color), axis=1)
    return PIL.Image.fromarray(im_vis), label