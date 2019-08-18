import os
import caffe
from collections import OrderedDict
from glob import glob
import numpy as np
from PIL import Image
import random
import sys
import time

base_dir = os.getcwd()
sys.path.append(base_dir)

from utils import check_array, check_dirs, get_options
from DeepImageSynthesis import LossFunctions
from DeepImageSynthesis.ImageSyn import ImageSyn
from DeepImageSynthesis.Misc import *


def main():
    opt = get_options()

    # seed
    random.seed(opt.seed)

    # load model
    VGGweights = os.path.join(base_dir, 'Models/vgg_normalised.caffemodel')
    VGGmodel = os.path.join(base_dir, 'Models/VGG_ave_pool_deploy.prototxt')
    imagenet_mean = np.array([ 0.40760392,  0.45795686,  0.48501961]) #mean for color channels (bgr)
    # im_dir = os.path.join(base_dir, 'Images/')
    caffe.set_mode_gpu() #for cpu mode do 'caffe.set_mode_cpu()'
    caffe.set_device(opt.cuda_id)

    # set dataset
    dataset_path = os.path.join(opt.data_root, opt.dataset,
                                'train' if opt.use_train else 'val')
    file_list = glob(os.path.join(dataset_path, '*/*.JPEG'))
    num_data = len(file_list)
    print('Number of data:', num_data)
    random.shuffle(file_list)

    # check the target dir
    target_dataset_path = os.path.join(opt.data_root, opt.tar_dir,
                                       'train' if opt.use_train else 'val')
    dir_list = glob(os.path.join(dataset_path, '*'))
    check_dirs(target_dataset_path, dir_list)

    # iteration for dataset
    print("Start Iteration")
    for itr, source_path in enumerate(file_list):
        # save textured image
        class_dir = source_path.split('/')[-2]
        filename = source_path.split('/')[-1]
        save_path = os.path.join(opt.data_root, opt.tar_dir, 
                                 'train' if opt.use_train else 'val',
                                 class_dir, filename)
        if os.path.exists(save_path):
            print(itr+1, "is passed.")
            continue

        start = time.time()
        #load source image
        source_img_org = caffe.io.load_image(source_path)
        # check_array(np.array(img), 'img')
        [source_img, net] = load_image(source_path, VGGmodel,
                                       VGGweights, imagenet_mean)
        # check_array(source_img, 'img after processed')
        im_size = np.asarray(source_img.shape[-2:])


        #pass image through the network and save the constraints on each layer
        constraints = OrderedDict()
        net.forward(data = source_img)
        for l,layer in enumerate(opt.tex_layers):
            constraints[layer] = constraint([LossFunctions.gram_mse_loss],
                                            [{'target_gram_matrix': gram_matrix(net.blobs[layer].data),
                                             'weight': opt.tex_weights[l]}])
            
        #get optimisation bounds
        bounds = get_bounds([source_img],im_size)

        #generate new texture
        result = ImageSyn(net, constraints, bounds=bounds,
                          callback=lambda x: show_progress(x,net), 
                          minimize_options={'maxiter': opt.max_iter,
                                            'maxcor': opt.max_cor,
                                            'ftol': 0, 'gtol': 0})


        #match histogram of new texture with that of the source texture and show both images
        new_texture = result['x'].reshape(*source_img.shape[1:]).transpose(1,2,0)[:,:,::-1]
        new_texture = histogram_matching(new_texture, source_img_org)
        new_texture = (new_texture * 255).astype(np.uint8)

        # save textured image
        Image.fromarray(new_texture).save(save_path)

        print('itr: {:d}/{:d} '
              'elapsed: {:.1f} '.format(
                  itr+1, num_data,
                  time.time() - start
                  )
              )

        if itr+1 == opt.stop_itr:
            break

if __name__ == '__main__':
    main()
