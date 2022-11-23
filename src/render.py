import cv2
import matplotlib.pyplot as plt

def display_plt(image, label, gt, cls, name_path):
    w = raw_image.shape[0]
    h = raw_image.shape[1]
    plt.figure(figsize = (14,14))
    label = np.argmax(label[0])
    plt.title(list(cls.keys())[label] + '_' + gt, fontproperties="SimHei")
    plt.imshow(image)
    plt.savefig(name_path)

def display_cv2(image, label, gt, cls, name_path):
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = np.argmax(label[0])
    cv2.putText(image, "Ground Truth:  " + list(cls.keys())[label], (0, 150), font, 3, (0, 0, 255), 15)
    cv2.putText(image, "转换: " + str(gt), (0, 250), font, 3, (0, 255, 0), 15,
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, True)
#     cv2.imshow('imshow',image)
    cv2.imwrite(name_path, image)