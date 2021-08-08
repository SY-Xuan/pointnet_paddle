import os
import sys
import paddle
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from ModelNetDataset import ModelNetDataset
from models.pointnet_utils import PointNetCls, feature_transform_regularizer
from paddle.optimizer import Adam
import paddle.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--feature_transform', action='store_true', default=False)
    return parser.parse_args()

def test(model, args, num_class=40):
    data_path = 'dataset/modelnet40_normal_resampled/'
    test_dataset = ModelNetDataset(root=data_path, args=args, split='test', process_data=args.process_data)
    testDataLoader = paddle.io.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=20)

    mean_correct = []

    model.eval()
    
    for j, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):

        # points, target = points.cuda(), target.cuda()

        points = points.transpose((0, 2, 1))
        with paddle.no_grad():
            pred, _ = model(points)
        pred_choice = paddle.argmax(pred, axis=1)
        
        correct = pred_choice.equal(target).astype("float32").sum()
        mean_correct.append(correct.numpy()[0] / float(points.shape[0]))

    instance_acc = np.mean(mean_correct)

    return instance_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, "ModelNet"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'dataset/modelnet40_normal_resampled/'

    train_dataset = ModelNetDataset(root=data_path, args=args, split='train', process_data=args.process_data)
    trainDataLoader = paddle.io.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=20, drop_last=True)

    '''MODEL LOADING'''
    num_class = 40
    classifier = PointNetCls(num_class, normal_channel=args.use_normals)

    # classifier = classifier.cuda()

    try:
        checkpoint = paddle.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.set_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=args.learning_rate, step_size=20, gamma=0.7)
    optimizer = Adam(
        parameters=classifier.parameters(),
        learning_rate=scheduler,
        epsilon=1e-08,
        weight_decay=args.decay_rate
    )
    
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier.train()

        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):

            points = points.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = paddle.to_tensor(points)
            points = points.transpose((0, 2, 1))

            # points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = F.nll_loss(pred, target)
            if args.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            pred_choice = paddle.argmax(pred, axis=1)
            correct = pred_choice.equal(target).astype("float32").sum()

            mean_correct.append(correct.numpy()[0] / float(points.shape[0]))
            
            global_step += 1
        
        scheduler.step()

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        instance_acc = test(classifier, args, num_class=num_class)

        if (instance_acc >= best_instance_acc):
            best_instance_acc = instance_acc
            best_epoch = epoch + 1

        log_string('Test Instance Accuracy: %f' % (instance_acc))
        log_string('Best Instance Accuracy: %f' % (best_instance_acc))

        if (instance_acc >= best_instance_acc):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': best_epoch,
                'instance_acc': instance_acc,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            paddle.save(state, savepath)
        global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    paddle.device.set_device("gpu")
    main(args)