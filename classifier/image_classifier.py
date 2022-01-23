from data.dataset import ImageNet
from model.resnet import *
from model.vggnet import *
from model.densenet import *
from model.ror import *
from model.preactivation_resnet import *
from model.resnext import *
from model.convnext import *
from utils.color import *
from utils.helper import *
from model.utils.labelsmoothingcrossentropy import *
from utils.scheduler import *
import tkinter
from tkinter import filedialog
import torchinfo
import torch.optim as optim
import os
import time
import numpy as np
import matplotlib.pyplot as plt


class ImageClassifier:

    def __init__(self):
        self.learning_rate_scheduler = None
        self.dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.pretrained_path = None
        self.b_pretrained = False
        self.learning_rate = 0
        self.best_prec1 = 0
        self.batch_size = 0
        self.epochs = 0
        self.step = 0
        self.training_accuracy_top1 = list()
        self.training_accuracy_top5 = list()
        self.validation_accuracy_top1 = list()
        self.validation_accuracy_top5 = list()

    def load_dataset(self):
        root = tkinter.Tk()
        root.withdraw()

        # Load Train Folderpath
        self.get_print_line()
        self.get_print_request("Select Training Folder Path")
        train_path = filedialog.askdirectory(parent=root, initialdir="/", title="Please Select A Train Data Folder")
        if train_path == "" or train_path == ():
            self.get_print_fail("Not Select Training Folder Path!")
            self.get_print_fail("Please Restart SW Now!!")
            self.get_print_line()
            exit(1)
        self.get_print_response("Train Path : {}".format(train_path))
        self.get_print_line()

        self.get_print_line()
        self.get_print_request("Select Validation Folder Path")
        valid_path = filedialog.askdirectory(parent=root, initialdir="/", title="Please Select A Valid Data Folder")
        if valid_path == "" or valid_path == ():
            self.get_print_fail("Not Select Validation Folder Path!")
            self.get_print_fail("Please Restart SW Now!!")
            self.get_print_line()
            exit(1)
        self.get_print_response("Validation Path : {}".format(valid_path))
        self.get_print_line()

        self.get_print_line()
        self.get_print_info("Load ImageNet Dataset!")
        self.dataset = ImageNet(train_path, valid_path)
        self.get_print_info("Training Data Size : {}, Validation Data Size : {}".format(self.dataset.get_train_size(),
                                                                                        self.dataset.get_valid_size()))
        self.get_print_info("Load ImageNet Dataset Finish!!")
        self.get_print_line()

        self.get_print_line()
        self.get_print_request("Select Pretrained Model File")
        pretrained_path = filedialog.askopenfilename(parent=root, initialdir="/",
                                                     title="Please Select Pretrained Model File!")
        if pretrained_path == "" or pretrained_path == ():
            self.get_print_fail("Not Select Pretrained Model File!")
            self.get_print_line()
        else:
            self.pretrained_path = pretrained_path
            self.b_pretrained = True
            self.get_print_fail("Pretrained File : {}".format(self.pretrained_path))
            self.get_print_line()
        self.get_print_line()

    def set_dataset_detail(self):
        self.get_print_line()
        self.get_print_request("Please enter the batch size!")
        self.get_print_request("Batch Size :")
        self.batch_size = int(input())
        self.get_print_line()
        self.get_print_line()
        self.get_print_request("Please enter the num_worker size!")
        self.get_print_request("Number of Worker Size : ")
        num_worker = int(input())
        self.get_print_line()
        self.get_print_line()
        self.train_loader = self.dataset.get_train_loader(batch_size=self.batch_size, num_worker=num_worker)
        self.valid_loader = self.dataset.get_valid_loader(batch_size=self.batch_size, num_worker=num_worker)
        self.get_print_info("DataLoader Detail Finish!!")
        self.get_print_line()

    def set_model(self):
        self.get_print_line()
        self.get_print_request("Please enter the number")
        self.get_print_request("0 : ResNet-18, 1 : ResNet-34, 2 : ResNet-50, 3 : ResNet-101, 4 : ResNet-152")
        self.get_print_request("5 : VGGNet-11, 6 : VGGNet-13, 7 : VGGNet-16, 8 : VGGNet-19")
        self.get_print_request("9 : DenseNet-121, 10 : DenseNet-169, 11 : DenseNet-201, 12 : DenseNet-264")
        self.get_print_request("13 : Pre-ResNet-18, 14 : Pre-ResNet-34, 15 : Pre-ResNet-50, 16 : Pre-ResNet-101, "
                               "17 : Pre-ResNet-152")
        self.get_print_request("18 : ResNeXt-50, 19 : ResNeXt-101, 20 : ResNeXt-152")
        self.get_print_request("21 : RoR3-18, 22 : RoR3-34, 23 : RoR3-50, 24 : RoR3-101, 25 : RoR3-152")
        self.get_print_request("26 : ConvNeXt-T, 27 : ConvNeXt-S, 28 : ConvNeXt-B, 29 : ConvNeXt-L, 30 : ConvNeXt-XL")
        number = int(input())
        if number == 0:
            self.model = resnet(18, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 1:
            self.model = resnet(34, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 2:
            self.model = resnet(50, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 3:
            self.model = resnet(101, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 4:
            self.model = resnet(152, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 5:
            self.model = vggnet(11, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 6:
            self.model = vggnet(13, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 7:
            self.model = vggnet(16, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 8:
            self.model = vggnet(19, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 9:
            self.model = densenet(121, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 10:
            self.model = densenet(169, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 11:
            self.model = densenet(201, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 12:
            self.model = densenet(264, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 13:
            self.model = preactivation_resnet(18, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 14:
            self.model = preactivation_resnet(34, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 15:
            self.model = preactivation_resnet(50, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 16:
            self.model = preactivation_resnet(101, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 17:
            self.model = preactivation_resnet(152, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 18:
            self.model = resnext(50, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 19:
            self.model = resnext(101, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 20:
            self.model = resnext(152, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 21:
            self.model = ror(18, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 22:
            self.model = ror(34, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 23:
            self.model = ror(50, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 24:
            self.model = ror(101, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 25:
            self.model = ror(152, 1000, self.b_pretrained, self.pretrained_path)
        elif number == 26:
            self.model = convnext('T', 1000, self.b_pretrained, self.pretrained_path)
        elif number == 27:
            self.model = convnext('S', 1000, self.b_pretrained, self.pretrained_path)
        elif number == 28:
            self.model = convnext('B', 1000, self.b_pretrained, self.pretrained_path)
        elif number == 29:
            self.model = convnext('L', 1000, self.b_pretrained, self.pretrained_path)
        elif number == 30:
            self.model = convnext('XL', 1000, self.b_pretrained, self.pretrained_path)
        else:
            self.get_print_fail("Not Corret Number!")
            self.get_print_fail("Please Restart SW Now!!")
            self.get_print_line()
            exit(1)
        self.model.to(self.dev)
        self.get_print_response("You Select Model : {}".format(self.model.get_name()))
        self.get_print_line()

        self.get_print_line()
        self.get_print_info("Start Model Check!")
        self.get_print_info("Get Input Data Size : (1, 3, 224, 224)")
        torchinfo.summary(self.model, (1, 3, 224, 224), device=self.dev)
        self.get_print_info("Finish Model Check!!")
        self.get_print_line()

        self.get_print_line()
        self.get_print_info("Image Classifier Model Selection Finish!!")
        self.get_print_line()

    def set_loss(self):
        self.get_print_line()
        self.get_print_info("Set Loss Function")
        self.get_print_request("Please enter the number")
        self.get_print_request("0 : CrossEntropyLoss, 1 : LabelSmoothingCrossEntropy")
        number = int(input())
        if number == 0:
            self.criterion = nn.CrossEntropyLoss().to(self.dev)
            self.get_print_response("You Select Cross Entropy Loss")
        elif number == 1:
            self.get_print_response("You Select Label Smoothing Cross Entropy Loss")
            self.get_print_request("Please enter the Label Smoothing Rate")
            smooth_rate = float(input())
            self.criterion = LabelSmoothingCrossEntropy(smoothing=smooth_rate).to(self.dev)
            self.get_print_response("Label Smoothing Cross Entropy Loss -> Smooth Rate : {}".format(smooth_rate))
        self.get_print_info("Loss Function Setting Finish!!")
        self.get_print_line()

    def set_optimizer(self):
        self.get_print_line()
        self.get_print_info("Start Optimzier Setting")
        self.get_print_line()

        if self.model.get_name() == 'ConvNeXt_T':
            total_batch_size = self.batch_size
            num_training_steps_per_epoch = self.dataset.get_train_size() // total_batch_size
            self.learning_rate_scheduler = cosine_scheduler(base_value=4e-3, final_value=1e-6, epochs=300,
                                                            niter_per_ep=num_training_steps_per_epoch, warmup_epochs=20,
                                                            start_warmup_value=0, warmup_steps=-1)
            self.optimizer = optim.AdamW(self.model.parameters(), lr=4e-3, weight_decay=0.05)
        else:

            self.get_print_line()
            self.get_print_request("Please enter the learning rate!")
            self.get_print_request("Learning Rate : ")
            self.learning_rate = float(input())
            self.get_print_line()

            self.get_print_line()
            self.get_print_request("Please enter the momentum!")
            self.get_print_request("Momentum : ")
            momentum = float(input())
            self.get_print_line()

            self.get_print_line()
            self.get_print_request("Please enter the weight_decay")
            self.get_print_request("Weight Decay : ")
            weight_decay = float(input())

            self.get_print_line()
            self.get_print_request("Please enter the number")
            self.get_print_request("0 : SGD, 1 : Adam, 2 : AdaGrad, 3 : RMSProp, 4 : AdamW")
            number = int(input())
            if number == 0:
                self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=momentum,
                                           weight_decay=weight_decay)
                self.get_print_response("You Select SGD -> Learning Rate : {}, Momentum : {}, Weight Decay : {}".format(
                    self.learning_rate, momentum, weight_decay))
            elif number == 1:
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
                self.get_print_response("You Select Adam -> Learning Rate : {}, Weight Decay : {}".format(
                    self.learning_rate, weight_decay))
            elif number == 2:
                self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate,
                                               weight_decay=weight_decay)
                self.get_print_response("You Select Adagrad -> Learning Rate : {}, Weight Decay : {}".format(
                    self.learning_rate, weight_decay))
            elif number == 3:
                self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, momentum=momentum,
                                               weight_decay=weight_decay)
                self.get_print_response(
                    "You Select RMSProp -> Learning Rate : {}, Momentum : {}, Weight Decay : {}".format(
                        self.learning_rate, momentum, weight_decay))
            elif number == 4:
                self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
                self.get_print_response("You Select AdamW -> Learning Rate : {}, Weight Decay : {}".format(
                    self.learning_rate, weight_decay))
            else:
                self.get_print_fail("Not Corret Number!")
                self.get_print_fail("Please Restart SW Now!!")
                self.get_print_line()
                exit(1)
        self.get_print_info("Finish Optimizer Setting!!!")
        self.get_print_line()

    def train(self):
        self.get_print_line()
        self.get_print_info("Check the Pretrained Model!")
        pretrained_path = "./Log"
        start_epoch = 0
        best_prec1 = 0
        if os.path.exists(pretrained_path):
            if os.path.isfile(os.path.join(pretrained_path, self.model.get_name() + '.pth')):
                checkpoint = torch.load(os.path.join(pretrained_path, self.model.get_name() + '.pth'))
                start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.get_print_info("Load the Pretrained Model!")
            else:
                self.get_print_info("Don't Exist Pretrained Model!")
        else:
            self.get_print_info("Don't Exist Pretrained Model")
            os.makedirs(pretrained_path)
        self.get_print_line()

        self.get_print_line()
        self.get_print_request("Please enter the epoch")
        self.get_print_request("Epoch : ")
        total_epcoh = int(input())

        self.get_print_line()
        self.get_print_info("Train Start!")
        self.get_print_line()
        for epoch in range(start_epoch, total_epcoh):
            if self.model.get_name() == 'ConvNeXt_T':
                pass
            else:
                self.get_print_line()
                lr = adjust_learning_rate(self.optimizer, epoch, self.learning_rate)
                self.get_print_info("Training Epoch {} Start!".format(epoch + 1))
                self.get_print_info("Current Learning Rate : {}".format(lr))
                self.get_print_line()

            self.get_print_line()
            self.train_per_epoch(self.train_loader, self.model, self.criterion, self.optimizer, epoch, 10)
            self.get_print_line()

            self.get_print_line()
            prec1, prec5 = self.validate(self.valid_loader, self.model, self.criterion, 10)
            self.get_print_line()

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': self.model.get_name(),
                'state_dict': self.model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': self.optimizer.state_dict()
            }, is_best, './Log/' + self.model.get_name() + ".pth")
            self.get_print_line()
            self.get_print_info("Epoch {} Save CheckPoint!!".format(epoch + 1))
            self.get_print_line()

    def train_per_epoch(self, train_loader, model, criterion, optimizer, epoch, print_freq):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        model.train()

        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if self.model.get_name() == 'ConvNeXt_T':
                for i, param_group in enumerate(optimizer.param_groups):
                    if self.learning_rate_scheduler is not None:
                        param_group["lr"] = self.learning_rate_scheduler[self.step]
                self.step = self.step + 1
            else:
                self.get_print_line()
                lr = adjust_learning_rate(self.optimizer, epoch, self.learning_rate)
                self.get_print_info("Training Epoch {} Start!".format(epoch + 1))
                self.get_print_info("Current Learning Rate : {}".format(lr))
                self.get_print_line()

            if torch.cuda.is_available():
                target = target.to(self.dev)
                input = input.to(self.dev)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                self.get_print_info("Epoch: [{0}][{1}/{2}]\t"
                                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                                    "Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t".format(
                    epoch + 1, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
        self.training_accuracy_top1.append(top1.avg)
        self.training_accuracy_top5.append(top5.avg)

    def validate(self, val_loader, model, criterion, print_freq):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                target = target.to(self.dev)
                input = input.to(self.dev)
            with torch.no_grad():
                # compute output
                output = model(input)
                loss = criterion(output, target)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1[0], input.size(0))
                top5.update(prec5[0], input.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    self.get_print_warning('Test: [{0}/{1}]\t'
                                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5))

        self.get_print_info(f' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}')
        self.validation_accuracy_top1.append(top1.avg)
        self.validation_accuracy_top5.append(top5.avg)
        return top1.avg, top5.avg

    def visualization_loss_graph(self):
        self.get_print_line()
        self.get_print_info("Plot Loss Graph")
        self.get_print_line()
        train_top1 = np.array(self.training_accuracy_top1)
        train_top5 = np.array(self.training_accuracy_top5)
        valid_top1 = np.array(self.validation_accuracy_top1)
        valid_top5 = np.array(self.validation_accuracy_top5)
        train_x = np.arange(train_top5.shape[0])
        valid_x = np.arange(valid_top5.shape[0])

        self.get_print_line()
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.plot(train_x, train_top1, 'r', label="Training Accuracy Top 1 : {}".format(train_top1[-1]))
        plt.plot(train_x, train_top5, 'g', label="Training Accuracy Top 5 : {}".format(train_top5[-1]))
        plt.xlabel("Epochs", size=12)
        plt.ylabel("Average Loss", size=12)
        plt.title("Training Loss per Epoch", size=15)
        plt.legend()
        plt.savefig("training_loss.png")
        self.get_print_info("Traiing Loss Graph Save Finish!")
        self.get_print_line()

        self.get_print_line()
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.plot(valid_x, valid_top1, 'r', label="Validation Accuracy Top 1 : {}".format(valid_top1[-1]))
        plt.plot(valid_x, valid_top5, 'g', label="Validation Accuracy Top 5 : {}".format(valid_top5[-1]))
        plt.xlabel("Epochs", size=12)
        plt.ylabel("Average Loss", size=12)
        plt.title("Validation Loss per Epoch", size=15)
        plt.legend()
        plt.savefig("validation_loss.png")
        self.get_print_info("Validation Loss Graph Save Finish!")
        self.get_print_line()

    def finish(self):
        self.get_print_line()
        self.get_print_ok("ImageNet Training Finish!!")
        self.get_print_ok("Goodbye~~")
        self.get_print_line()
        exit(0)

    def get_print_line(self):
        print(Colors.LINE + "---------------------------------------------------- \n" + Colors.RESET)

    def get_print_info(self, infomation):
        print(Colors.INFOMATION + infomation + Colors.RESET)

    def get_print_fail(self, fail):
        print(Colors.FAIL + fail + Colors.RESET)

    def get_print_warning(self, warning):
        print(Colors.WARNING + warning + Colors.RESET)

    def get_print_request(self, request):
        print(Colors.REQUEST + request + Colors.RESET)

    def get_print_response(self, response):
        print(Colors.RESPONSE + response + Colors.RESET)

    def get_print_ok(self, ok):
        print(Colors.OK + ok + Colors.RESET)
