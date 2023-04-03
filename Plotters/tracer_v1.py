import os
import time
from sys import stdout
import torch
import h5py as h5
import numpy as np

from lib.networks.utils import AverageMeter, save_model
from lib.visualization.utils import add_figures_reconstruction_tb, add_svr_reconstruction_tb
from plotter_nparray import plot


def train(iterator, model, loss_func, optimizer, scheduler, epoch, iter, warmup, train_writer, **kwargs):
    num_workers = kwargs.get('num_workers')
    train_mode = kwargs.get('train_mode')
    model_name = os.path.join(kwargs['logging_path'], kwargs.get('model_name'))

    batch_time = AverageMeter()
    data_time = AverageMeter()

    LB = AverageMeter()
    PNLL = AverageMeter()
    GNLL = AverageMeter()
    GENT = AverageMeter()

    model.train()
    torch.set_grad_enabled(True)

    end = time.time()
#for loop starts
    for i, batch in enumerate(iterator):
        if iter + i >= len(iterator):
            break
        data_time.update(time.time() - end)
        scheduler(optimizer, epoch, iter + i)

        print("iteration number is",i)
        #print("batch is ",batch)
        g_clouds = batch['cloud'].cuda(non_blocking=True)
        #print("g_cloud in training.py is",g_clouds)                 #testing. Can be removed
        p_clouds = batch['eval_cloud'].cuda(non_blocking=True)
        #print("p_cloud in training.py is",p_clouds)                 #testing . Can be removed
        # returns shape distributions list in prior flows, samples list in decoder flows
        # and log weights of all flows in decoder flows.
        output_prior, output_decoder, mixture_weights_logits = model(g_clouds, p_clouds, images=None, n_sampled_points=None, labeled_samples=False, warmup=warmup)

        loss, pnll, gnll, gent = loss_func(output_prior, output_decoder, mixture_weights_logits)
        with torch.no_grad():
            if torch.isnan(loss):
                print('Loss is NaN! Stopping without updating the net...')
                exit()

        PNLL.update(pnll.item(), g_clouds.shape[0])
        GNLL.update(gnll.item(), g_clouds.shape[0])
        GENT.update(gent.item(), g_clouds.shape[0])
        LB.update((pnll + gnll - gent).item(), g_clouds.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        if (iter + i + 1) % (num_workers) == 0 and kwargs['logging']:
            line = 'Epoch: [{0}][{1}/{2}]'.format(epoch + 1, iter + i + 1, len(iterator))
            line += '\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(batch_time=batch_time)
            line += '\tLB {LB.val:.2f} ({LB.avg:.2f})'.format(LB=LB)
            line += '\tPNLL {PNLL.val:.2f} ({PNLL.avg:.2f})'.format(PNLL=PNLL)
            line += '\tGNLL {GNLL.val:.2f} ({GNLL.avg:.2f})'.format(GNLL=GNLL)
            line += '\tGENT {GENT.val:.2f} ({GENT.avg:.2f})'.format(GENT=GENT)
            line += '\n'
            stdout.write(line)
            stdout.flush()

        end = time.time()

        if (iter + i + 1) % (100 * num_workers) == 0 and kwargs['logging']:
            if kwargs['distributed']:
                sd = model.module.state_dict()
            else:
                sd = model.state_dict()
            save_model({
                'epoch': epoch,
                'iter': iter + i + 1,
                'model_state': sd,
                'optimizer_state': optimizer.state_dict()
            }, model_name)
#for loop ends

    # write to tensorboard
    if kwargs['logging']:
        train_writer.add_scalar('train/loss', LB.avg, epoch)
        train_writer.add_scalar('train/PNLL', PNLL.avg, epoch)
        train_writer.add_scalar('train/GNLL', GNLL.avg, epoch)
        train_writer.add_scalar('train/GENT', GENT.avg, epoch)

    if kwargs['logging']:
        if kwargs['distributed']:
            sd = model.module.state_dict()
        else:
            sd = model.state_dict()
        save_model({
            'epoch': epoch + 1,
            'iter': 0,
            'model_state': sd,
            'optimizer_state': optimizer.state_dict()
        }, model_name)


def eval(iterator, model, loss_func, optimizer, epoch, iter, warmup, min_loss, eval_writer, **kwargs):
    train_mode = kwargs.get('train_mode')

    LB = AverageMeter()
    PNLL = AverageMeter()
    GNLL = AverageMeter()
    GENT = AverageMeter()

    model.eval()
    torch.set_grad_enabled(False)

    for i, batch in enumerate(iterator):
        if iter + i >= len(iterator):
            break
        g_clouds = batch['cloud'].cuda(non_blocking=True)
        p_clouds = batch['eval_cloud'].cuda(non_blocking=True)
        output_prior, output_decoder, mixture_weights_logits = model(g_clouds, p_clouds, images=None, n_sampled_points=None, labeled_samples=False, warmup=warmup)

        with torch.no_grad():
            loss, pnll, gnll, gent = loss_func(output_prior, output_decoder, mixture_weights_logits)

        PNLL.update(pnll.item(), g_clouds.shape[0])
        GNLL.update(gnll.item(), g_clouds.shape[0])
        GENT.update(gent.item(), g_clouds.shape[0])
        LB.update((pnll + gnll - gent).item(), g_clouds.shape[0])

        with torch.no_grad():
            if torch.isnan(loss):
                print('Loss is NaN! Stopping without updating the net...')
                exit()
            if torch.isinf(loss):
                print('Loss is INF! Stopping without updating the net...')
                exit()

    if kwargs.get('logging'):
        print('[epoch %d]: eval loss %f' % (epoch, LB.avg))

    # write to tensorboard
    if kwargs.get('logging'):
        eval_writer.add_scalar('val/loss', LB.avg, epoch)
        eval_writer.add_scalar('val/PNLL', PNLL.avg, epoch)
        eval_writer.add_scalar('val/GNLL', GNLL.avg, epoch)
        eval_writer.add_scalar('val/GENT', GENT.avg, epoch)

    # Add reconstruction visualization to tensorboard
    if kwargs.get('logging_img') and epoch % kwargs.get('logging_img_frequency') == 0 and kwargs.get('logging'):
        npy_path = os.path.join(kwargs.get('logging_path'), '')

        if kwargs.get('distributed'):
            tmp_mode = model.module.mode
            model.module.mode = 'autoencoding'
            all_samples, all_gts, all_labels = reconstruct(iterator, model, max_batches=1, warmup=False, **kwargs)
            model.module.mode = tmp_mode
        else:
            tmp_mode = model.mode
            model.mode = 'autoencoding'
            all_samples, all_gts, all_labels = reconstruct(iterator, model, max_batches=1, warmup=False, **kwargs)
            model.mode = tmp_mode

        # save numpy data
        all_labels = all_labels.detach().cpu().numpy()
        all_samples = all_samples.detach().cpu().numpy()
        all_gts = all_gts.detach().cpu().numpy()

        add_figures_reconstruction_tb(all_gts, all_samples, all_labels, eval_writer, epoch)

    if LB.avg < min_loss:
        min_loss = LB.avg
        best_modelname = 'best_model_' + kwargs.get('model_name')
        best_model_name = os.path.join(kwargs['logging_path'], best_modelname)
        if kwargs.get('logging'):
            if kwargs['distributed']:
                sd = model.module.state_dict()
            else:
                sd = model.state_dict()
            save_model({
                'epoch': epoch + 1,
                'iter': 0,
                'model_state': sd,
                'optimizer_state': optimizer.state_dict()
            }, best_model_name)
    return min_loss

def main():
    parser = define_options_parser()
    args = parser.parse_args()
    ngpus_per_node = torch.cuda.device_count()
    if args.distributed and ngpus_per_node > 1:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = find_free_port()  # '6666'
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        args.distributed = False
        main_worker(0, 1, args)

