import sklearn.linear_model
from matplotlib import pyplot as plt
from sklearn.linear_model import LassoCV
import SimpleITK as sitk
import numpy as np
import os
import os.path as osp
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
import joblib
from CBB.myDecorator import timer
from CBB.myos import is_Exist, get_full_paths, check_path


def generate_LASSO(min=-3, max=1):
    min, max = min, max
    alphas = np.logspace(min, max, 50)
    extractor = LassoCV(alphas=alphas, cv=5, max_iter=300000)
    return extractor

def generate_RELF():
    from skrebate import ReliefF
    return ReliefF()


@timer
def feature_extraction(data_path,mask_path, param = r"D:\Pycharm_workplace\COVID19\param\Params.yaml"):
    from radiomics import featureextractor
    image = sitk.ReadImage(data_path)
    mask = sitk.ReadImage(mask_path)
    if sitk.GetArrayViewFromImage(mask).max() > 1:
        print("Warning! Get the mask file in 0/255 Format, rescale the pixel values")
        mask = mask//255

    image_spacing = image.GetSpacing()
    mask_spacing = mask.GetSpacing()
    if not image_spacing == mask_spacing:
        print("Warning! Spacing inconsistent image spacing: {}; mask spacing: {}".format(image_spacing, mask_spacing))
    extractor = featureextractor.RadiomicsFeatureExtractor(param)
    feature_vector = extractor.execute(image, mask)
    return feature_vector


def generate_feature_file(data_dir, mask_dir, save_path=None):
    is_Exist(data_dir)
    is_Exist(mask_dir)

    masks, dcms = sorted(get_full_paths(mask_dir)), sorted(get_full_paths(data_dir))

    total_db = None
    success = 0
    failed = []

    if (save_path is not None and osp.exists(save_path)):
        check_file = save_path
        dul_check = True
        features_finished = pd.read_csv(check_file, engine='python')
    elif osp.exists(r"./features.csv"):
        check_file = r"./features.csv"
        dul_check=True
        features_finished = pd.read_csv(check_file, engine='python')
    else:
        dul_check=False

    for (dcm, mask) in zip(dcms,masks):
        sample_name = osp.basename(dcm).split(".")[0]
        mask_name = osp.basename(mask).split(".")[0]
        assert sample_name == mask_name,"Error! The executed dicom is not consist with mask {}".format(sample_name,mask_name)
        print("Processing sample {}".format(sample_name))
        if dul_check:
            names = features_finished['name']
            if sample_name in list(names):
                print("Skip finished sample {}".format(sample_name))
                continue
        try:
            features = feature_extraction(dcm, mask)
            df = pd.DataFrame(features.values(), index=features.keys()).transpose()
            df['name'] = sample_name
            total_db = df if total_db is None else pd.concat([total_db, df])
            success += 1
        except:
            print("Failed to extract features from sample {}".format(sample_name))
            failed.append(sample_name)

    print("Feature extraction finished! Successfully proceed {} samples, Failed in {}".format(success, failed))
    total_db.set_index('name', inplace=True)
    if dul_check and features_finished is not None:
        features_finished.set_index('name',inplace=True)
        total_db = pd.concat([features_finished, total_db])
    if save_path is not None:
        total_db.to_csv(save_path)
    else:
        if not dul_check:
            total_db.to_csv("./features.csv")
        else:
            total_db.to_csv("./features_update.csv")

def test_model(model, data, label=None, cm=False, ROC=False):
    from sklearn.preprocessing import StandardScaler
    data=data.fillna(0)
    if label is None:
        print("Label column is not given use 'label' as default!")
        label = 'label'

    if isinstance(model, str):
        model = joblib.load(model)

    features = model['feature_names']
    model = model['model']

    X = data[features]
    y = data[label]
    X = StandardScaler().fit_transform(X)
    X = pd.DataFrame(X, columns=features)

    y_pred = model.predict(X)
    y_pred_prob = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    auc_score = roc_auc_score(y, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    print(f'Accuracy: {accuracy:.2f}')
    print(f'AUC: {auc_score:.2f}')
    print(f'Specificity: {specificity:.2f}')
    print(f'Sensitivity: {sensitivity:.2f}')

    if cm:
        check_path("./result")
        cm_matrix = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
        plt.yticks(tick_marks, ['Negative', 'Positive'])
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        fmt = 'd'
        thresh = cm_matrix.max() / 2.
        for i, j in np.ndindex(cm_matrix.shape):
            plt.text(j, i, format(cm_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm_matrix[i, j] > thresh else "black")

        plt.savefig('result/confusion_matrix.jpg')
        plt.close()
        print("Confusion matrix saved as './result/confusion_matrix.jpg'")

    if ROC:
        check_path("./result")
        fpr, tpr, _ = roc_curve(y, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.savefig('result/ROC.jpg')
        plt.close()
        print('ROC curve saved as result/ROC.jpg')





