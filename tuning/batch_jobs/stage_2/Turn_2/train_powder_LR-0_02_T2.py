## regular module imports
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pickle
import skimage
import skimage.io
import sys
import pandas as pd
import pycocotools.mask as RLE
import seaborn as sns


## detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
)
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.structures import BoxMode

root = '../../../../../'
ocean_images = root + '../../../../ocean/projects/dmr200021p/sprice/tuning/'
sys.path.append(root)

from ampis import data_utils, visualize, analyze
from ampis.applications import powder
from ampis.structures import InstanceSet
from ampis.visualize import display_iset

#CONSTANTS
#--------------------------------------------------------------
EXPERIMENT_NAME = 'satellite' # can be 'particle' or 'satellite'
NUM_ITERATIONS = 15000
CHECKPOINT_NUM = 1000
NUM_CYCLES = 15
OUTPUT_FOLDER = 'batch_temp2'
OUTPUT_FILE = '../LR-WD_S2_T2.txt'
LR = 0.02
#WD = 0.0001
WD_list = [0.000001, 0.000003, 0.000005, 0.000007, 0.000009, 0.00001, 0.00003, 0.00005, 0.00007,0.00009]
BB = 'ResNet50'
#--------------------------------------------------------------
for i in WD_list:
    WD = i
    ##LOADING DATA
    json_path_train = Path('..', '..', 'SALAS_Rep', 'satellite_training.json')  # path to training data
    json_path_val = Path('..', '..', 'SALAS_Rep', 'satellite_validation.json')  # path to training data
    assert json_path_train.is_file(), 'training file not found!'
    assert json_path_val.is_file(), 'validation file not found!'

    ## REGISTERING DATA

    DatasetCatalog.clear()  # resets catalog, helps prevent errors from running cells multiple times

    # store names of datasets that will be registered for easier access later
    dataset_train = f'{EXPERIMENT_NAME}_Train'
    dataset_valid = f'{EXPERIMENT_NAME}_Val'

    # register the training dataset
    DatasetCatalog.register(dataset_train, lambda f = json_path_train: data_utils.get_ddicts(label_fmt='via2',  # annotations generated from vgg image annotator
                                                                                                        im_root=f,  # path to the training data json file
                                                                                                        dataset_class='Train'))  # indicates this is training data
    # register the validation dataset
    DatasetCatalog.register(dataset_valid, lambda f = json_path_val: data_utils.get_ddicts(label_fmt='via2',  # annotations generated from vgg image annotator
                                                                                                    im_root=f,  # path to validation data json file
                                                                                                    dataset_class='Validation'))  # indicates this is validation data
                                
    print(f'Registered Datasets: {DatasetCatalog.list()}')

    ## There is also a metadata catalog, which stores the class names.
    for d in [dataset_train, dataset_valid]:
        MetadataCatalog.get(d).set(**{'thing_classes': [EXPERIMENT_NAME]})


    #VERIFYING IMAGES LOADED PROPERLY

    for i in np.random.choice(DatasetCatalog.get(dataset_train), 3, replace=False):
        visualize.display_ddicts(i, None, dataset_train, suppress_labels=True)
        

    for i in DatasetCatalog.get(dataset_valid):
        visualize.display_ddicts(i, None, dataset_valid, suppress_labels=True)

    #MODEL CONFIGS
    cfg = get_cfg() # initialize cfg object
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))  # load default parameters for Mask R-CNN
    cfg.INPUT.MASK_FORMAT = 'polygon'  # masks generated in VGG image annotator are polygons
    cfg.DATASETS.TRAIN = (dataset_train,)  # dataset used for training model
    cfg.DATASETS.VALIDATION = (dataset_valid,)
    cfg.DATASETS.TEST = (dataset_train, dataset_valid)  # we will look at the predictions on both sets after training
    cfg.SOLVER.IMS_PER_BATCH = 1 # number of images per batch (across all machines)
    cfg.SOLVER.CHECKPOINT_PERIOD = CHECKPOINT_NUM  # number of iterations after which to save model checkpoints
    cfg.MODEL.DEVICE='cuda'  # 'cpu' to force model to run on cpu, 'cuda' if you have a compatible gpu
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # Since we are training separate models for particles and satellites there is only one class output
    cfg.TEST.DETECTIONS_PER_IMAGE = 400 if EXPERIMENT_NAME == 'particle' else 250  # maximum number of instances that can be detected in an image (this is fixed in mask r-cnn)
    cfg.SOLVER.MAX_ITER = NUM_ITERATIONS  # maximum number of iterations to run during training
                                # Increasing this may improve the training results, but will take longer to run (especially without a gpu!)

    #-------------------------------------------------
    cfg.SOLVER.BASE_LR = LR
    cfg.SOLVER.WEIGHT_DECAY = WD
    #-------------------------------------------------

    # model weights will be downloaded if they are not present
    weights_path = Path('..','..','models','model_final_f10217.pkl')
    if weights_path.is_file():
        print('Using locally stored weights: {}'.format(weights_path))
    else:
        weights_path = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        print('Weights not found, weights will be downloaded from source: {}'.format(weights_path))
    cfg.MODEL.WEIGHTs = str(weights_path)
    cfg.OUTPUT_DIR = str(Path(ocean_images, 'weights', OUTPUT_FOLDER))
    # make the output directory
    os.makedirs(Path(cfg.OUTPUT_DIR), exist_ok=True)


    #MODEL TRAINING
    trainer = DefaultTrainer(cfg)  # create trainer object from cfg
    trainer.resume_or_load(resume=False)  # start training from iteration 0
    trainer.train()  # train the model!


    pickle_folder = []
    #CREATING PREDICTOR
    model_checkpoints = sorted(Path(cfg.OUTPUT_DIR).glob('*.pth'))  # paths to weights saved druing training
    #cfg.DATASETS.TEST = (dataset_train, dataset_valid)  # predictor requires this field to not be empty
    for cycle in range(len(model_checkpoints)):
        cfg.MODEL.WEIGHTS = str(model_checkpoints[-cycle])  # use the last model checkpoint saved during training. If you want to see the performance of other checkpoints you can select a different index from model_checkpoints.
        print("USING MODEL WEIGHT: " + str(model_checkpoints[-cycle]))
        predictor = DefaultPredictor(cfg)  # create predictor object
        #SAVING RESULTS
        results = []
        for ds in cfg.DATASETS.VALIDATION:
            print(f'Dataset: {ds}')
            for dd in DatasetCatalog.get(ds):
                print(f'\tFile: {dd["file_name"]}')
                img = cv2.imread(dd['file_name'])  # load image
                outs = predictor(img)  # run inference on image
                
                # format results for visualization and store for later
                results.append(data_utils.format_outputs(dd['file_name'], ds, outs))

                # visualize results
                visualize.display_ddicts(outs, None, ds, gt=False, img_path=dd['file_name'])
        t_name = ((str(model_checkpoints[-cycle]).split('/'))[-1]).split('_')[-1].split('.pth')[0]
        # save to disk
        title = 'results_checkpoint_' + str(t_name) + '.pickle'
        with open(Path(ocean_images, 'weights', OUTPUT_FOLDER, title), 'wb') as f:
            pickle.dump(results, f)
        pickle_folder.append(title)
        #CALCULATING SEGMENTATION SCORES
        average_p = []
        average_r = []
        for i in range(1):
            #Loading Ground Truth Labels
            satellites_gt_path = Path(ocean_images, 'satellite_auto_validation_v1.2.json')
            for path in [satellites_gt_path]:
                assert path.is_file(), f'File not found : {path}'
            satellites_gt_dd = data_utils.get_ddicts('via2', satellites_gt_path, dataset_class='train')
            #Loading Prediction Labels
            satellites_path = Path(ocean_images, 'weights', OUTPUT_FOLDER, title)
            assert satellites_path.is_file()
            with open(satellites_path, 'rb') as f:
                satellites_pred = pickle.load(f)
            iset_satellites_gt = [InstanceSet().read_from_ddict(x, inplace=False) for x in satellites_gt_dd]
            iset_satellites_pred = [InstanceSet().read_from_model_out(x, inplace=False) for x in satellites_pred]
            #Creating Instance Set Objects
            iset_satellites_gt, iset_satellites_pred = analyze.align_instance_sets(iset_satellites_gt, iset_satellites_pred)
            #Re-ordering instance sets to be concurrent
            for gt, pred in zip(iset_satellites_gt, iset_satellites_pred):
                pred.HFW = gt.HFW
                pred.HFW_units = gt.HFW_units
                print(f'gt filename: {Path(gt.filepath).name}\t pred filename: {Path(pred.filepath).name}')
            #Creating Detection Scores
            dss_satellites = [analyze.det_seg_scores(gt, pred, size=gt.instances.image_size)
                            for gt, pred in zip(iset_satellites_gt, iset_satellites_pred)]
            labels = []
            counts = {'train': 0, 'validation': 0}
            for iset in iset_satellites_gt:
                counts[iset.dataset_class] += 1
                labels.append(iset.filepath.name)
            x=[*([1] * len(labels)), *([2] * len(labels))]
            # y values are the bar heights

            scores = [*[x['det_precision'] for x in dss_satellites],
                *[x['det_recall'] for x in dss_satellites]]
            labels = labels * 2
            print('x: ', x)
            print('y: ', [np.round(x, decimals=2) for x in scores])
            #print('labels: ', labels)
            fig, ax = plt.subplots(figsize=(6,3), dpi=150)
            sns.barplot(x=x, y=scores, hue=labels, ax=ax)
            ax.legend(bbox_to_anchor=(1,1))
            ax.set_ylabel('detection score')
            ax.set_xticklabels(['precision','recall'])
            print("Average Precision Score: ", str(sum([*[x['det_precision'] for x in dss_satellites]])/len([*[x['det_precision'] for x in dss_satellites]])))
            print("Average Recall Score:    ", str(sum([*[x['det_recall'] for x in dss_satellites]])/len([*[x['det_recall'] for x in dss_satellites]])))
            #Analyzing Prediction Scores on a pixel level
            temp_p = []
            temp_r = []
            total_area = 1024*768
            for instance in range(len(iset_satellites_pred)):
                fp_area = 0
                fn_area = 0
                tp_area = 0
                iset_satellites_pred[instance].compute_rprops(keys=['area'])
                for i in dss_satellites[instance]['det_fp']:
                    try: 
                        fp_area += int(iset_satellites_pred[instance].rprops['area'][i])
                    except:
                        pass

                for i in dss_satellites[instance]['det_fn']:
                    try: 
                        fn_area += int(iset_satellites_pred[instance].rprops['area'][i])
                    except:
                        pass

                #print(dss_satellites[0]['seg_tp'])
                for i in dss_satellites[instance]['det_tp']:
                    try: 
                        tp_area += int(iset_satellites_pred[instance].rprops['area'][i[1]])
                    except:
                        pass
                print("Precision:", str(tp_area/(tp_area+fp_area)))
                print('Recall:', str(tp_area/(tp_area+fn_area)))
                temp_p.append(tp_area/(tp_area+fp_area))
                temp_r.append(tp_area/(tp_area+fn_area))
                print('---')
            average_p.append(temp_p)
            average_r.append(temp_r)
            '''counter = 0   
            for iset in iset_satellites_gt:
                gt = iset_satellites_gt[counter]
                pred = iset_satellites_pred[counter]
                iset_det, colormap = analyze.det_perf_iset(gt, pred)
                img = skimage.color.gray2rgb(skimage.io.imread(iset.filepath))
                #display_iset(img, iset=iset_det)
                counter += 1'''
        del (average_p[0])[-1]
        del (average_r[0])[-1]
        
        iteration_name = ((str(model_checkpoints[-cycle]).split('/'))[-1]).split('_')[-1].split('.pth')[0]
        if iteration_name == 'final':
            print("Ignoring Final Model")
        else:
            return_list = [LR, WD, BB, str(int(iteration_name)), str(sum(average_p[0])/len(average_p[0])), str(sum(average_r[0])/len(average_r[0]))]
            with open(OUTPUT_FILE, "a") as output:
                output.write(str(return_list))
            f = open(OUTPUT_FILE, "a")
            f.write(',\n')
            f.close()
    for model in range(len(model_checkpoints)):
        print("Deleting: " + str(model_checkpoints[-model]))
        os.remove(str(model_checkpoints[-model]))
    for file in range(len(pickle_folder)):
        temp = ocean_images + "weights/" + OUTPUT_FOLDER + "/" + pickle_folder[file]
        print("Deleting: " + temp)
        os.remove(temp)
    print("Removing: " + ocean_images + "weights/" + OUTPUT_FOLDER  + "/" +"metrics.json")
    os.remove(ocean_images + "weights/" + OUTPUT_FOLDER  + "/" +"metrics.json")
    print("Removing: " + ocean_images + "weights/" + OUTPUT_FOLDER  + "/" +"last_checkpoint")
    os.remove(ocean_images + "weights/" + OUTPUT_FOLDER  + "/" + "last_checkpoint")

