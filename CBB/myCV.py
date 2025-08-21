import cv2
import numpy
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import graycomatrix, graycoprops
import os
import SimpleITK as sitk

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

def cropToTumorMask(imageNode, maskNode, boundingBox):
  """
  Create a sitkImage of the segmented region of the image based on the input label.

  Create a sitkImage of the labelled region of the image, cropped to have a
  cuboid shape equal to the ijk boundaries of the label.

  :param boundingBox: The bounding box used to crop the image. This is the bounding box as returned by
    :py:func:`checkMask`.
  :param label: [1], value of the label, onto which the image and mask must be cropped.
  :return: Cropped image and mask (SimpleITK image instances).

  """

  size = numpy.array(maskNode.GetSize())

  ijkMinBounds = boundingBox[0::2]
  ijkMaxBounds = size - boundingBox[1::2]- 1

  # Ensure cropped area is not outside original image bounds
  ijkMinBounds = numpy.maximum(ijkMinBounds, 0)
  ijkMaxBounds = numpy.maximum(ijkMaxBounds, 0)

  # Crop Image
  cif = sitk.CropImageFilter()
  try:
    cif.SetLowerBoundaryCropSize(ijkMinBounds)
    cif.SetUpperBoundaryCropSize(ijkMaxBounds)
  except TypeError:
    # newer versions of SITK/python want a tuple or list
    cif.SetLowerBoundaryCropSize(ijkMinBounds.tolist())
    cif.SetUpperBoundaryCropSize(ijkMaxBounds.tolist())
  croppedImageNode = cif.Execute(imageNode)
  croppedMaskNode = cif.Execute(maskNode)

  return croppedImageNode, croppedMaskNode

def get_bin_edges(image, binWidth=25, binCount=None):
    if binCount is not None:
        binEdges = numpy.histogram(image, binCount)[1]
        binEdges[-1] += 1  # Ensures that the maximum value is included in the topmost bin when using numpy.digitize
    else:
        minimum = min(image)
        maximum = max(image)

    lowBound = minimum - (minimum % binWidth)
    highBound = maximum + 2 * binWidth

    binEdges = numpy.arange(lowBound, highBound, binWidth)

    if len(binEdges) == 1:  # Flat region, ensure that there is 1 bin
      binEdges = [binEdges[0] - .5, binEdges[0] + .5]  # Simulates binEdges returned by numpy.histogram if bins = 1

    return binEdges

def bin_image(image, parameterMatrixCoordinates, binWidth=25, binCount=None):
    discretizedParameterMatrix = numpy.zeros(image.shape, dtype='int')
    binEdges = get_bin_edges(image[parameterMatrixCoordinates],  binWidth=binWidth, binCount=binCount)
    discretizedParameterMatrix[parameterMatrixCoordinates] = numpy.digitize(image[parameterMatrixCoordinates],
                                                                            binEdges)
    return discretizedParameterMatrix, binEdges

def cal_entropy(image,mask):
    _,p_i = np.unique(image[mask],return_counts=True)
    p_i = p_i.reshape((1, -1))
    sumBins = np.sum(p_i, 1, keepdims=True).astype('float')
    sumBins[sumBins == 0] = 1
    p_i = p_i.astype('float') / sumBins
    eps = np.spacing(1)
    entropy = -1.0 * np.sum(p_i * np.log2(p_i + eps), 1)
    return entropy
def get_glcm_mat(bin_image, gray_levels=None):
    if gray_levels is None:
        gray_levels = len(np.unique(bin_image))
    cur_gray = bin_image[:,:,:-1].flatten()
    neightbor_gray = bin_image[:,:,1:].flatten()

    glcm = np.zeros((gray_levels,gray_levels), dtype=np.int64)
    np.add.at(glcm,(cur_gray,neightbor_gray),1)
    return glcm

if __name__ == '__main__':
    bin_mat = [
        [1, 2, 5, 2, 3],
        [3, 2, 1, 3, 1],
        [1, 3, 5, 5, 2],
        [1, 1, 1, 1, 2],
        [1, 2, 4, 3, 5]
    ]

    bin_mat = np.array(bin_mat)
    unique_bin = np.unique(bin_mat)
    value_to_index = {v: i for i, v in enumerate(unique_bin)}
    mapped_bin_mat = np.vectorize(value_to_index.get)(bin_mat)
    print(mapped_bin_mat)
    bin_len = len(unique_bin)

    glcm_mat = np.zeros((bin_len,bin_len),dtype=int)
    print(glcm_mat)
    left_mat = mapped_bin_mat[:, 1:].flatten()
    right_mat = mapped_bin_mat[:,:-1].flatten()
    np.add.at(glcm_mat,(left_mat, right_mat),1)
    np.add.at(glcm_mat, (right_mat, left_mat), 1)
    print(glcm_mat)





