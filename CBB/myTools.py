import SimpleITK as sitk
import os
import os.path as osp
import numpy as np
import nibabel as nib
import pandas as pd
import pydicom
from CBB.myos import is_Exist, check_path, format_number


def sep_100(var):
    return 0 if var < 100 else 1

def sep_6(var):
    return 0 if var < 6 else 1


def dcm2nii(dcm_path, nii_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(dicom_names)
    image2 = reader.Execute()

    image_array = sitk.GetArrayFromImage(image2)

    origin = image2.GetOrigin()
    spacing = image2.GetSpacing()
    direction = image2.GetDirection()

    image3 = sitk.GetImageFromArray(image_array)
    image3.SetSpacing(spacing)
    image3.SetDirection(direction)
    image3.SetOrigin(origin)
    sitk.WriteImage(image3, nii_path)

def dicom_series_to_nrrd(input_folder, output_nrrd):

    dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(input_folder)

    reader = sitk.ImageSeriesReader()

    reader.SetFileNames(dicom_names)

    dicom_series = reader.Execute()

    sitk.WriteImage(dicom_series, output_nrrd)


def dicomConverter(dicom_dir,save_path):
    import pydicom

    dicom_files = sorted([file for file in os.listdir(dicom_dir) if file.endswith(".dcm")])
    dicom_files = list(map(lambda x : osp.join(dicom_dir,x), dicom_files))
    first_dicom = pydicom.dcmread(dicom_files[0])
    rows,columns = int(first_dicom.Rows), int(first_dicom.Columns)

    output = np.zeros((rows,columns,len(dicom_files)), dtype=np.uint8)
    for i, dcm in enumerate(dicom_files):
        dicom = pydicom.dcmread(dcm)
        output[:, :, i] = dicom.pixel_array

    import SimpleITK as sitk

    image_3d = sitk.GetImageFromArray(output)

    image_3d.SetSpacing((float(first_dicom.PixelSpacing[0]), float(first_dicom.PixelSpacing[1]),float(first_dicom.PixelSpacing[2])))
    image_3d.SetOrigin((float(first_dicom.ImagePositionPatient[0]),float(first_dicom.ImagePositionPatient[1]),float(first_dicom.ImagePositionPatient[2])))

    sitk.WriteImage(image_3d, save_path)

def nii2dicom(nii_path, save_dir):
    is_Exist(nii_path)
    check_path(save_dir)

    nii_data = nib.load(nii_path)
    nii_array = nii_data.get_fdata()
    header = nii_data.header
    for i in range(nii_array.shape[2]):

        slice_data = nii_array[:, :, i]

        # 创建 DICOM 数据对象
        dcm = pydicom.dcmread(pydicom.config.default_file_meta)

        # 设置 DICOM 属性
        dcm.PatientName = "Anonymous"
        dcm.PatientID = "123456"
        dcm.StudyInstanceUID = pydicom.uid.generate_uid()
        dcm.SeriesInstanceUID = pydicom.uid.generate_uid()
        dcm.SOPInstanceUID = pydicom.uid.generate_uid()
        dcm.Modality = "CT"
        dcm.PixelSpacing = (1, 1)
        dcm.Rows, dcm.Columns = slice_data.shape
        dcm.ImagePositionPatient = (0, 0, i)
        dcm.SliceThickness = 1
        dcm.ImageOrientationPatient = (1, 0, 0, 0, 1, 0)
        dcm.BitsStored = 16
        dcm.BitsAllocated = 16
        dcm.SamplesPerPixel = 1
        dcm.PixelData = slice_data.tobytes()

def read_data(path):
    import SimpleITK as sitk
    is_Exist(path)
    try:
        image = sitk.ReadImage(path)
        image_array = sitk.GetArrayFromImage(image)
        return image_array
    except Exception as e:
        print(f"Error reading image file {path}: {e}")
        return None

def save_data(image_array, path, reference_image_path):
    try:
        is_Exist(reference_image_path)
        reference_image = sitk.ReadImage(reference_image_path)
        image = sitk.GetImageFromArray(image_array)
        image.CopyInformation(reference_image)  # Preserve the meta-information
        sitk.WriteImage(image, path)
        print(f"Image successfully saved to {path}")
    except Exception as e:
        print(f"Error saving image to {path}: {e}")

def consist_check(dicom_dir, mask_dir, label_file):
    print("Attention, label file should only be in the format that each sample lays in one line and pos and cons seperated by a empty line")
    is_Exist(label_file)
    is_Exist(dicom_dir)
    is_Exist(mask_dir)

    with open(label_file, "r") as f:
        lines = f.readlines()
        seperator_index = lines.index("\n")
        pos_index,neg_index = lines[:seperator_index], lines[seperator_index+1:]
        pos_index = list(map(lambda x:x.strip(), pos_index))
        neg_index = list(map(lambda x:x.strip(), neg_index))
        idx = pos_index+neg_index
        labels = [1] * len(pos_index) + [0] * len(neg_index)
        temp_db = pd.DataFrame(
            {"name" : idx,"label":labels}
        )

    dicoms = os.listdir(dicom_dir)
    dicoms = list(map(lambda x:x.split(".")[0], dicoms))

    masks = os.listdir(mask_dir)
    masks = list(map(lambda x:x.split(".")[0], masks))

    dicom_found = set(idx).intersection(set(dicoms))
    mask_dicom_found = set(dicom_found).intersection(set(masks))

    if not sorted(list(mask_dicom_found)) == sorted(list(idx)):
        print("Error! Please check the following samples: ")
        print(set(idx).difference(set(mask_dicom_found)), sep="\n")
        assert 0
    else:
        print(f"Detect {len(mask_dicom_found)} samples")

    return mask_dicom_found, temp_db




if __name__ == '__main__':
    sample_nii = r"D:\Pycharm_workplace\COVID19\COVID19_INFECTION_DB\Data\radiopaedia_4_85506_1.nii.gz"
    output_dir = r"./sample"
    nii2dicom(sample_nii,output_dir)