import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import collections
import numpy as np
# referred: https://github.com/1adrianb/face-alignment/blob/master/examples/detect_landmarks_in_image.py
def plot_2Dfeatures_on_image(pred_features: np.array, image) -> plt.figure:
    plot_style = dict(marker='o',
                    markersize=4,
                    linestyle='-',
                    lw=2)

    pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
    pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
                'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
                'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
                'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
                'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
                'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
                'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
                'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
                'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
                }

    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(image)

    for pred_type in pred_types.values():
        ax.plot(pred_features[pred_type.slice, 0],
                pred_features[pred_type.slice, 1],
                color=pred_type.color, **plot_style)

    ax.axis('off')
    return fig

def plot_3Dfeatures(preds):
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.scatter(preds[:, 0] * 1.2,
                    preds[:, 1],
                    preds[:, 2],
                    c='cyan',
                    alpha=1.0,
                    edgecolor='b')
    pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
    pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
                'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
                'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
                'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
                'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
                'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
                'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
                'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
                'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
                }

    for pred_type in pred_types.values():
        ax.plot3D(preds[pred_type.slice, 0] * 1.2,
                preds[pred_type.slice, 1],
                preds[pred_type.slice, 2], color='blue')

    ax.view_init(elev=90., azim=90.)
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.show()
    return fig



def plot_landmarks_on_image(image, landmarks, color='red', fig=None):
    if len(image.shape) == 2:  
        image = np.stack((image,)*3, axis=-1)  
    if fig is None:
        fig = plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off') 
    h, w = image.shape[:2]
    for landmark in landmarks:
        x, y = landmark[:2]
        plt.scatter(x*w, y*h, s=5, color=color, alpha=0.7)  # Adjust size and color as needed
    plt.axis('off')
    plt.xlim(0, w)  # width
    plt.ylim(h, 0)  # height, invert y-axis
    plt.tight_layout()
    return fig