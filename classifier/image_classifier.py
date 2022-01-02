from data.dataset import ImageNet
from model.resnet import *
from utils.color import *
from utils.helper import *
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
        self.dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.learning_rate = 0
        self.best_prec1 = 0
        self.training_loss = list()
        self.validation_loss = list()

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

    def set_dataset_detail(self):
        self.get_print_line()
        self.get_print_request("Please enter the batch size!")
        self.get_print_request("Batch Size :")
        batch_size = int(input())
        self.get_print_line()
        self.get_print_line()
        self.get_print_request("Please enter the num_worker size!")
        self.get_print_request("Number of Worker Size : ")
        num_worker = int(input())
        self.get_print_line()
        self.get_print_line()
        self.train_loader = self.dataset.get_train_loader(batch_size=batch_size, num_worker=num_worker)
        self.valid_loader = self.dataset.get_valid_loader(batch_size=batch_size, num_worker=num_worker)
        self.get_print_info("DataLoader Detail Finish!!")
        self.get_print_line()

    def set_model(self):
        self.get_print_line()
        self.get_print_request("Please enter the number")
        self.get_print_request("0 : ResNet-18, 1 : ResNet-34, 2 : ResNet-50, 3: ResNet-101, 4 : ResNet-152")
        number = int(input())
        if number == 0:
            self.model = resnet(18, 1000, False, None)
        elif number == 1:
            self.model = resnet(34, 1000, False, None)
        elif number == 2:
            self.model = resnet(50, 1000, False, None)
        elif number == 3:
            self.model = resnet(101, 1000, False, None)
        elif number == 4:
            self.model = resnet(152, 1000, False, None)
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
        self.criterion = nn.CrossEntropyLoss().to(self.dev)
        self.get_print_info("CrossEntroyLoss Setting Finish!!")
        self.get_print_line()

    def set_optimizer(self):
        self.get_print_line()
        self.get_print_info("Start Optimzier Setting")
        self.get_print_line()

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
        self.get_print_request("0 : SGD, 1 : Adam, 2 : AdaGrad, 3 : RMSProp")
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
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
            self.get_print_response("You Select Adagrad -> Learning Rate : {}, Weight Decay : {}".format(
                self.learning_rate, weight_decay))
        elif number == 3:
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, momentum=momentum,
                                           weight_decay=weight_decay)
            self.get_print_response("You Select Adam -> Learning Rate : {}, Momentum : {}, Weight Decay : {}".format(
                self.learning_rate, momentum, weight_decay))
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
                self.model.load_state_dict(checkpoint['optimizer'])
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
            self.get_print_line()
            lr = adjust_learning_rate(self.optimizer, epoch, self.learning_rate)
            self.get_print_info("Training Epoch {} Start!".format(epoch+1))
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
            self.get_print_info("Epoch {} Save CheckPoint!!".format(epoch+1))
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
        self.training_loss.append(losses.avg)

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
                    self.get_print_info('Test: [{0}/{1}]\t'
                                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5))

        self.get_print_info(f' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}')
        self.validation_loss.append(losses.avg)
        return top1.avg, top5.avg

    def visualization_loss_graph(self):
        self.get_print_line()
        self.get_print_info("Plot Loss Graph")
        self.get_print_line()
        train_loss = np.array(self.training_loss)
        valid_loss = np.array(self.validation_loss)
        train_x = np.arange(train_loss.shape[0])
        valid_x = np.arange(valid_loss.shape[0])

        self.get_print_line()
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.plot(train_x, train_loss, 'r', label="Training Loss")
        plt.xlabel("Epochs", size=12)
        plt.ylabel("Average Loss", size=12)
        plt.title("Training Loss per Epoch", size=15)
        plt.savefig("training_loss.png")
        self.get_print_info("Traiing Loss Graph Save Finish!")
        self.get_print_line()

        self.get_print_line()
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.plot(valid_x, valid_loss, 'r', label="Validation Loss")
        plt.xlabel("Epochs", size=12)
        plt.ylabel("Average Loss", size=12)
        plt.title("Validation Loss per Epoch", size=15)
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
