import numpy as np
import pandas as pd
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import warnings
import math
from scipy.misc import toimage, imresize
from skimage import measure, morphology
from skimage.transform import resize

warnings.filterwarnings("ignore")

INPUT_FOLDER = "E:/Datasets/Lung Cancer/Stage 1/Pictures/stage1/"
MASK_FOLDER = "D:/Datasets/Lung Cancer/Data/Stage 1/Pictures/stage1_masks/"
PREPD_FOLDER = "D:/Datasets/Lung Cancer/Data/Stage 1/Pictures/stage1_prepd/"
patients = os.listdir(INPUT_FOLDER)
patients.sort()

def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)

def get_batches(patients):
    for ix, patient in enumerate(patients):
        scan = load_scan(INPUT_FOLDER + patient)
        slices = get_pixels_hu(scan)
        if ix % 10 == 0:
            print("Processed patient {0} of {1}".format(ix, len(patients)))
        yield scan, slices, patient
        
def save_array(path, arr):
    np.save(path, arr)
    
def load_array(path):
    return np.load(path)

def resample(image, scan, new_spacing=[1,1,1]):
    thickness = [scan[0].SliceThickness]
    if not thickness[0]:
        thickness = [1.0]  # because weird error
    spacing = np.array(thickness + scan[0].PixelSpacing, dtype=np.float32)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25
IMG_SIZE_PX = 256
SLICE_COUNT = 64
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None
    
def segment_lung_mask(image, fill_lung_structures=True):
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    background_label = labels[0,0,0]
    binary_image[background_label == labels] = 2
    if fill_lung_structures:
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            if l_max is not None:
                binary_image[i][labeling != l_max] = 1
    binary_image -= 1
    binary_image = 1-binary_image
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: 
        binary_image[labels != l_max] = 0
    for i, axial_slice in enumerate(binary_image):
        binary_image[i] = morphology.dilation(axial_slice, np.ones([10,10]))
    return binary_image

def save_masks(scan, patient):
    masks = segment_lung_mask(scan, True)
    np.save(MASK_FOLDER + "{}.npy".format(patient), masks)
    return masks

def apply_masks(imgs, masks):
    out_images = []
    for i in range(len(imgs)):
        mask = masks[i]
        img = imgs[i]
        img= mask*img 
        img = resize(img, [IM_SIZE, IM_SIZE])
        out_images.append(img)
    return np.array(out_images)

def save_preprocessed(patient, scan, masks):
    normalized = normalize(scan)
    centered = zero_center(normalized)
    masked = apply_masks(centered, masks)
    save_array(PREPD_FOLDER + "{}.npy".format(patient), masked)
    
gen = get_batches(patients)
for scan, slices, patient in gen:
    try:
        resampled = resample(slices, scan)
        masks = save_masks(resampled, patient)
        save_preprocessed(patient, resampled, masks)
    except Exception as e:
        print(patient, e)
		
labels = pd.read_csv("D:/Datasets/Lung Cancer/Data/Stage 1/stage1_solution.csv", index_col=0)
train_ids = [id.replace(".npy", "") for id in os.listdir("D:/Datasets/Lung Cancer/Data/Stage 1/Pictures/stage1_prepd/")]
train_ids.sort()

def chunks( l,n ):
    count=0
    for i in range(0, len(l), n):
        if(count < SLICE_COUNT):
            yield l[i:i + n]
            count=count+1
def mean(l):
    return sum(l)/len(l)

def save_array(path, arr):
    np.save(path, arr)
    
def load_array(path):
    return np.load(path)
	
def rotate(patient):
        sample_list = []
        label = labels.get_value(patient,"cancer")
        sample = load_array("D:/Datasets/Lung Cancer/Data/Stage 1/Pictures/stage1_prepd/{}.npy".format(patient))
        sample = np.array([imresize(toimage(im), size=(IMG_SIZE_PX, IMG_SIZE_PX)) for im in sample])
        sample = np.array([skimage.transform.rotate(im, 15, mode='reflect') for im in sample])
        chunk_sizes = math.floor(len(sample) / SLICE_COUNT)
        for sample_chunk in chunks(sample,chunk_sizes):
            sample_chunk = list(map(mean,zip(*sample_chunk)))
            sample_list.append(sample_chunk)
        return np.array(sample_list),label
     
def process(patient):
        sample_list = []
        label = labels.get_value(patient,"cancer")
        sample = load_array("D:/Datasets/Lung Cancer/Data/Stage 1/Pictures/stage1_prepd/{}.npy".format(patient))
        sample = np.array([imresize(toimage(im), size=(IMG_SIZE_PX, IMG_SIZE_PX)) for im in sample])
        chunk_sizes = math.floor(len(sample) / SLICE_COUNT)
        for sample_chunk in chunks(sample,chunk_sizes):
            sample_chunk = list(map(mean,zip(*sample_chunk)))
            sample_list.append(sample_chunk)
        return np.array(sample_list),label
		
for ix,patient in enumerate(train_ids):
    try:
        img_data,label = process(patient)
        print(img_data.shape,label,patient)
        save_array("D:/Datasets/Lung Cancer/Data/Stage 1/Pictures/64-256-256_data/{}.npy".format(patient), 
                   [[img_data],[label]])
    except KeyError as e:
        print("This is unlabeled data!: {}".format(patient))

for ix, patient in enumerate(train_ids):
    try:
        label = labels.get_value(patient, "cancer")
        if label == 1:
            img_data, label = rotate(patient)
            print(img_data.shape, label, patient)
            save_array("D:/Datasets/Lung Cancer/Data/Stage 1/Pictures/64-256-256_data/{}.npy".format(patient+'rot2'), 
                           [[img_data],[label]])
    except KeyError as e:
        print("This is unlabeled data!: {}".format(patient))