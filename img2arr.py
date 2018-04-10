import io
import matplotlib.pyplot as plt
import librosa.display as disp
import librosa
from PIL import Image
import numpy as np
# def save_ax_nosave(ax, **kwargs):
#     ax.axis("off")
#     ax.figure.canvas.draw()
#     trans = ax.figure.dpi_scale_trans.inverted()
#     bbox = ax.bbox.transformed(trans)
#     buff = io.BytesIO()
#     plt.savefig(buff, format="png", dpi=ax.figure.dpi, bbox_inches='tight',pad_inches=0)
#     ax.axis("on")
#     buff.seek(0)
#     im = plt.imread(buff )
#     return im

def give_img_arr(y):
    cqt = librosa.cqt(y)
    ax = disp.specshow(cqt)
    fig = ax.figure
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    im = Image.fromarray(data)
    return im
