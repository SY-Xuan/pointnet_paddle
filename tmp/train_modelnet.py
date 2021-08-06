from __future__ import print_function
import argparse
import os
import random
import paddle
from paddle.optimizer import Adam
from dataset import ModelNetDataset
from model import PointNetCls, feature_transform_regularizer
import paddle.nn.functional as F
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='modelnet40', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)


if opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')


dataloader = paddle.io.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    drop_last=True,
    num_workers=int(opt.workers))

testdataloader = paddle.io.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.set_state_dict(paddle.load(opt.model))

scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.001, step_size=20, gamma=0.5)
optimizer = Adam(parameters=classifier.parameters(), learning_rate=scheduler, betas=(0.9, 0.999))

num_batch = len(dataset) / opt.batchSize

best = 0
for epoch in range(opt.nepoch):
    classifier.train()

    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose((0, 2, 1))
        points, target = points.cuda(), target.cuda()

        pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        pred_choice = paddle.argmax(pred, axis=1)
        correct = pred_choice.equal(target).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.numpy(), correct.numpy() / float(opt.batchSize)))

        if i % 10 == 0:
            # j, data = next(enumerate(testdataloader, 0))
            # points, target = data
            # target = target[:, 0]
            # points = points.transpose((0, 2, 1))
            # points, target = points.cuda(), target.cuda()
            # classifier = classifier.eval()
            # pred, _, _ = classifier(points)
            # loss = F.nll_loss(pred, target)
            # pred_choice = paddle.argmax(pred, axis=1)
            # correct = pred_choice.equal(target).cpu().sum()

            # print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.numpy(), correct.numpy()/float(opt.batchSize)))
            total_correct = 0
            total_testset = 0
            for i,data in tqdm(enumerate(testdataloader, 0)):
                points, target = data
                target = target[:, 0]
                points = points.transpose((0, 2, 1))
                points, target = points.cuda(), target.cuda()
                classifier.eval()
                pred, _, _ = classifier(points)
                pred_choice = paddle.argmax(pred, axis=1)
                correct = pred_choice.equal(target).cpu().sum()
                total_correct += float(correct.numpy())
                total_testset += points.shape[0]
            print("final accuracy {} best accuray {}".format(total_correct / float(total_testset), best))
            if best < (total_correct / float(total_testset)):
                best = (total_correct / float(total_testset))
                paddle.save(classifier.state_dict(), '%s/cls_model_best.pth' % (opt.outf))
    scheduler.step()
    