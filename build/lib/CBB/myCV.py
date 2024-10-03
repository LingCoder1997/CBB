import cv2
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import graycomatrix, graycoprops
import os


# 提取 ORB 特征点和描述符
def extract_orb_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(fastThreshold=0, edgeThreshold=0)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

# 加载样本图像并提取特征
def load_sample_images(sample_dir):
    samples = []
    labels = []
    keypoints_list = []
    descriptors_list = []
    for filename in os.listdir(sample_dir):
        if filename.endswith(".jpg"):
            image = cv2.imread(os.path.join(sample_dir, filename))
            image = cv2.resize(image, (100,100))
            keypoints, descriptors = extract_orb_features(image)
            if descriptors is not None:  # 检查描述符是否为空
                samples.append(image)
                keypoints_list.append(keypoints)
                descriptors_list.append(descriptors)
                labels.append(filename.split(".")[0])  # 假设文件名格式为 "label_index.jpg"
    return samples, keypoints_list, descriptors_list, labels

# 使用 ORB 进行图像匹配
def match_image(test_image, samples, keypoints_list, descriptors_list):
    orb = cv2.ORB_create(fastThreshold=0, edgeThreshold=0)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    test_keypoints, test_descriptors = extract_orb_features(test_image)

    if test_descriptors is None:
        raise ValueError("No descriptors found in test image")

    best_match_idx = -1
    best_num_matches = 0

    for i, descriptors in enumerate(descriptors_list):
        # 调试输出描述符的类型和形状
        if descriptors is None:
            continue  # 跳过描述符为空的样本

        matches = bf.match(test_descriptors, descriptors)
        if len(matches) > best_num_matches:
            best_num_matches = len(matches)
            best_match_idx = i

    return best_match_idx, best_num_matches, test_keypoints, test_descriptors

# 显示匹配结果
def display_match(test_image, best_match_idx, samples, keypoints_list, labels, test_keypoints, test_descriptors, descriptors_list):
    if best_match_idx == -1:
        print("No matches found.")
        return

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(test_descriptors, descriptors_list[best_match_idx])
    matches = sorted(matches, key=lambda x: x.distance)
    result = cv2.drawMatches(test_image, test_keypoints, samples[best_match_idx], keypoints_list[best_match_idx], matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow(f"Best match: {labels[best_match_idx]}", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


