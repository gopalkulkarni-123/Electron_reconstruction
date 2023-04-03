import os
from time import time
from sys import stdout

import h5py as h5
import numpy as np
import torch

from lib.networks.utils import AverageMeter

def  gen(iterator, model, loss_func, **kwargs):
    train_mode = kwargs.get('train_mode')
    util_mode = kwargs.get('util_mode')
    #is_saving = kwargs.get('saving')
    is_saving = True
    if is_saving:
        #saving generated point clouds, ground-truth point clouds and sampled labels.
        clouds_fname = 'airplanes_and_chairs.h5'
        clouds_fname = os.path.join(kwargs['logging_path'], clouds_fname)
        print(clouds_fname)
        clouds_file = h5.File(clouds_fname, 'w')
        sampled_clouds = clouds_file.create_dataset(
            'sampled_clouds',
            shape=(kwargs['N_sets'] * len(iterator.dataset), 5, kwargs['sampled_cloud_size']),
            dtype=np.float32)

    elif util_mode == 'generating':
        gen_clouds_buf = []
        ref_clouds_buf = []

    model.eval()
    torch.set_grad_enabled(False)

    end = time()

    for i, batch in enumerate(iterator):
        g_clouds = torch.zeros((kwargs.get('batch_size'), 5, kwargs.get('sampled_cloud_size')))
        p_clouds = torch.zeros((1, 5, kwargs.get('sampled_cloud_size')))
        cloud_labels = torch.zeros((kwargs.get('batch_size')))


        inf_end = time()
        n_components = kwargs.get('n_components')
        n = kwargs.get('sampled_cloud_size')

        # for test, generate samples
        with torch.no_grad():
            if train_mode == 'p_rnvp_mc_g_rnvp_vae':
                output_prior, samples, labels, log_weights = model(g_clouds, p_clouds, cloud_labels, images=None, n_sampled_points=n, labeled_samples=True)
            elif train_mode == 'p_rnvp_mc_g_rnvp_vae_ic':
                images = batch['image'].cuda(non_blocking=True)
                output_prior, samples, labels, log_weights = model(g_clouds, p_clouds, images, n_sampled_points=n, labeled_samples=True)
        
        #inf_time.update((time() - inf_end) / g_clouds.shape[0], g_clouds.shape[0])

        r_clouds = samples
        if kwargs['unit_scale_evaluation']:
            if kwargs['cloud_scale']:
                r_clouds *= kwargs['cloud_scale_scale']
                p_clouds *= kwargs['cloud_scale_scale']
        if kwargs['orig_scale_evaluation']:
            if kwargs['cloud_scale']:
                r_clouds *= kwargs['cloud_scale_scale']
                p_clouds *= kwargs['cloud_scale_scale']

            if kwargs['cloud_translate']:
                shift = torch.from_numpy(np.array(kwargs['cloud_translate_shift']).reshape(1, -1, 1)).cuda()
                r_clouds += shift
                p_clouds += shift

            if not kwargs['cloud_rescale2orig']:
                r_clouds *= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
                p_clouds *= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
            if not kwargs['cloud_recenter2orig']:
                r_clouds += batch['orig_c'].unsqueeze(2).cuda()
                p_clouds += batch['orig_c'].unsqueeze(2).cuda()

        if is_saving:
            # saving generated point clouds, ground-truth point clouds and sampled labels.
            if i<=10:
                sampled_clouds[i] = r_clouds[0].detach().cpu().numpy().astype(np.float32)
        end = time()

    if util_mode == 'generating':
        # compute mmd-cd, mmd-emd, mmd-f1 for generation task
        gen_clouds_buf = torch.transpose(torch.cat(gen_clouds_buf, dim=0), 2, 1).contiguous()
        ref_clouds_buf = torch.transpose(torch.cat(ref_clouds_buf, dim=0), 2, 1).contiguous()

        gen_clouds_buf = gen_clouds_buf.cpu().numpy()
        gen_clouds_inds = set(np.arange(gen_clouds_buf.shape[0]))
        nan_gen_clouds_inds = set(np.isnan(gen_clouds_buf).sum(axis=(1, 2)).nonzero()[0])
        gen_clouds_inds = list(gen_clouds_inds - nan_gen_clouds_inds)
        dup_gen_clouds_inds = np.random.choice(gen_clouds_inds, size=len(nan_gen_clouds_inds))
        gen_clouds_buf[list(nan_gen_clouds_inds)] = gen_clouds_buf[dup_gen_clouds_inds]
        gen_clouds_buf = torch.from_numpy(gen_clouds_buf).cuda()

        res = {}    
    return res