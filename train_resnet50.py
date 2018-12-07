from __future__ import print_function
import sys
import os
import random
import re
import math
caffe_root='/home/jiangzhiqi/Documents/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

def parseCommand():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("src_prototxt_dir", help="the directory of the source deploy file which we will generate prototxt files by modifying")
    parser.add_argument("save_dir", help="the directory of the generated files required when training, such as lmdb, prototxt files, train.txt, etc.")
    parser.add_argument("caffe_tool_dir", help="the directory of caffe/build/tools")
    parser.add_argument("data_dir", help="the directory of the labeled source data used to train")
    parser.add_argument("--resize_height", default=0, type=int, help="image height in lmdb. If 0, save images in original size")
    parser.add_argument("--resize_width", default=0, type=int, help="image width in lmdb. If 0, save images in original size")
    parser.add_argument("--base_lr", default=0.01, type=int, help="base_lr in solver.prototxt")
    parser.add_argument("--weight_decay", default=0.005, type=float, help="weight_decay in solver.prototxt")
    parser.add_argument("--batch_size_train", default=30, type=int, help="batch size in train")
    parser.add_argument("--batch_size_val", default=16, type=int, help="batch size in val")
    parser.add_argument("--train_soon", help="whether to train", action="store_true")
    parser.add_argument("--gpu", default="0", help="gpus or gpu on which to train")
    parser.add_argument("--pretrained_model_dir", default="", help="the directory of the pretrained model used to finetune")
    parser.add_argument("--log_dir", default="logs", help="the directory of log")
    args = parser.parse_args()
    return args


class FileOperation(object):
    def getAllFiles(self,root_dir):
        list_files = os.listdir(root_dir)
        list_file_path=[]
        for file_name in list_files:
            file_path=root_dir+"/"+file_name
            if not os.path.isdir(file_path):
                list_file_path.append(file_path)
            else:
                list_file_path.extend(self.getAllFiles(file_path))
        return list_file_path


    def getAllImageFiles(self,root_dir):
        list_file_path=self.getAllFiles(root_dir)
        all_image_list=[]
        for file_path in list_file_path:
            extension_name = file_path.split(".")[-1].lower()
            if extension_name=="jpg" or extension_name=="bmp" or extension_name=="png" \
            or extension_name=="jpeg":
                all_image_list.append(file_path)
        return all_image_list


    def getDirs(self,root_dir):
        dir_list = os.listdir(root_dir)
        result_list=[]
        for file_name in dir_list:
            file_path=root_dir+"/"+file_name
            if os.path.isdir(file_path):
                result_list.append(file_path)
        return result_list


    def createDir(self,path):
        if not os.path.exists(path):
            os.makedirs(path)


    def addLabel(self,data_list,label):
        result=[]
        for data in data_list:
            result_data=data+" "+str(label)
            result.append(result_data)
        return result


    def splitImagesToTrainTest(self,image_list,test_percent):
        random.shuffle(image_list)
        num_files=len(image_list)
        train_num=int(num_files*(1-test_percent))
        train_list=image_list[0:train_num]
        test_list =image_list[train_num:num_files]
        return train_list,test_list


    def saveDataFile(self,data_list,save_file): 
    	random.shuffle(data_list)
        f_save_file= open(save_file,'w')
        for data in data_list:
            # print data
            f_save_file.write(data+'\n')
        f_save_file.close()



class ResNetClassificationModel(object):
    def __init__(self, data_dir, save_dir, resize_width, resize_height, caffe_tool_dir, batch_size_train,\
        batch_size_val, src_prototxt_dir, base_lr, weight_decay, pretrained_model_dir, gpu, train_soon, log_dir):
        self.__data_dir=data_dir
        self.__save_dir=os.path.abspath(save_dir)
        self.__resize_width=resize_width
        self.__resize_height=resize_height
        self.__caffe_tool_dir=caffe_tool_dir
        self.__batch_size_train=batch_size_train
        self.__batch_size_val=batch_size_val
        self.__src_prototxt_dir=src_prototxt_dir
        self.__base_lr=base_lr
        self.__weight_decay=weight_decay
        self.__pretrained_model_dir=pretrained_model_dir
        self.__gpu=gpu
        self.__train_soon=train_soon
        self.__log_dir=os.path.join(save_dir, log_dir)

        self.__num_classes=0
        self.__num_train_image=0
        self.__num_val_image=0
        self.__file_operation=FileOperation()


    def createDeployPrototxt(self):
        num_classes=self.__num_classes
        src_prototxt_dir=self.__src_prototxt_dir
        save_dir=self.__save_dir

        src_resnet50_deploy_prototxt="{}/ResNet-50-deploy.prototxt".format(src_prototxt_dir)
        dst_resnet50_deploy_prototxt="{}/my_deploy.prototxt".format(save_dir)
        file_object=open(src_resnet50_deploy_prototxt)
        net_deploy=caffe_pb2.NetParameter()
        strText=file_object.read()

        text_format.Merge(strText, net_deploy)
        net_deploy.layer.remove(net_deploy.layer[-1])

        fc1000_relu=caffe_pb2.LayerParameter()
        fc1000_relu.name="fc1000_relu"
        fc1000_relu.type="ReLU"
        fc1000_relu.bottom.append("fc1000")
        fc1000_relu.top.append("fc1000")

        fcn=caffe_pb2.LayerParameter()
        fcn.name="fcn"
        fcn.type="InnerProduct"
        fcn.bottom.append("fc1000")
        fcn.top.append("fcn")
        fcn.inner_product_param.num_output=num_classes

        prob=caffe_pb2.LayerParameter()
        prob.name="prob"
        prob.type="Softmax"
        prob.bottom.append("fcn")
        prob.top.append("prob")

        net_deploy.layer.extend([fc1000_relu, fcn, prob])

        with open(dst_resnet50_deploy_prototxt, 'w') as f:
                print(text_format.MessageToString(net_deploy), file=f)


    def createTrainValPrototxt(self):
        num_classes=self.__num_classes
        batch_size_train=self.__batch_size_train
        batch_size_val=self.__batch_size_val
        src_prototxt_dir=self.__src_prototxt_dir
        save_dir=self.__save_dir

        src_resnet50_deploy_prototxt="{}/ResNet-50-deploy.prototxt".format(src_prototxt_dir)
        dst_resnet50_trainval_prototxt="{}/my_trainval.prototxt".format(save_dir)
        train_lmdb="{}/train_lmdb".format(save_dir)
        val_lmdb="{}/train_lmdb".format(save_dir)
        mean_file="{}/mean.binaryproto".format(save_dir)

        file_object=open(src_resnet50_deploy_prototxt)
        net=caffe_pb2.NetParameter()
        strText=file_object.read()
        text_format.Merge(strText, net)


        net.layer.remove(net.layer[-1])
        net.input.pop()
        for i in range(4):
            net.input_dim.pop()
            
        data_layer_train=caffe_pb2.LayerParameter()
        data_layer_train.name="data"
        data_layer_train.type="Data"
        data_layer_train.top.extend(["data", "label"])
        data_layer_train.include.add().phase=caffe_pb2.Phase.Value("TRAIN")
        data_layer_train.transform_param.mirror=True
        data_layer_train.transform_param.crop_size=224
        data_layer_train.transform_param.mean_file=mean_file
        data_layer_train.data_param.source=train_lmdb
        data_layer_train.data_param.batch_size=batch_size_train
        data_layer_train.data_param.backend=caffe_pb2.DataParameter.DB.Value('LMDB')

        data_layer_test=caffe_pb2.LayerParameter()
        data_layer_test.name="data"
        data_layer_test.type="Data"
        data_layer_test.top.extend(["data", "label"])
        data_layer_test.include.add().phase=caffe_pb2.Phase.Value("TEST")
        data_layer_test.transform_param.mirror=False
        data_layer_test.transform_param.crop_size=224
        data_layer_test.transform_param.mean_file=mean_file
        data_layer_test.data_param.source=val_lmdb
        data_layer_test.data_param.batch_size=batch_size_val
        data_layer_test.data_param.backend=caffe_pb2.DataParameter.DB.Value('LMDB')

        net_trainval=caffe_pb2.NetParameter()
        net_trainval.name="ResNet-50"
        net_trainval.layer.extend([data_layer_train, data_layer_test])

        net_trainval.layer.extend(net.layer)

        fc1000_relu=caffe_pb2.LayerParameter()
        fc1000_relu.name="fc1000_relu"
        fc1000_relu.type="ReLU"
        fc1000_relu.bottom.append("fc1000")
        fc1000_relu.top.append("fc1000")

        fcn=caffe_pb2.LayerParameter()
        fcn.name="fcn"
        fcn.type="InnerProduct"
        fcn.bottom.append("fc1000")
        fcn.top.append("fcn")
        fcn.inner_product_param.num_output=num_classes

        accuracy=caffe_pb2.LayerParameter()
        accuracy.name="accuracy"
        accuracy.type="Accuracy"
        accuracy.bottom.extend(["fcn", "label"])
        accuracy.top.append("accuracy")

        loss=caffe_pb2.LayerParameter()
        loss.name="loss"
        loss.type="SoftmaxWithLoss"
        loss.bottom.extend(["fcn", "label"])
        loss.top.append("loss")

        net_trainval.layer.extend([fc1000_relu, fcn, accuracy, loss])

        for i in range(len(net_trainval.layer)):
            if net_trainval.layer[i].type == "Convolution":
                if net_trainval.layer[i].convolution_param.bias_term == True:
                    net_trainval.layer[i].param.extend([caffe_pb2.ParamSpec(lr_mult=0, decay_mult=0), caffe_pb2.ParamSpec(lr_mult=0, decay_mult=0)])
                else:
                    net_trainval.layer[i].param.extend([caffe_pb2.ParamSpec(lr_mult=0, decay_mult=0)])
            if net_trainval.layer[i].type == "Scale":
                net_trainval.layer[i].param.extend([caffe_pb2.ParamSpec(lr_mult=0, decay_mult=0), caffe_pb2.ParamSpec(lr_mult=0, decay_mult=0)])
            if net_trainval.layer[i].type == "BatchNorm":
                net_trainval.layer[i].param.extend([caffe_pb2.ParamSpec(lr_mult=0, decay_mult=0), caffe_pb2.ParamSpec(lr_mult=0, decay_mult=0), \
                    caffe_pb2.ParamSpec(lr_mult=0, decay_mult=0)])
                net_trainval.layer[i].batch_norm_param.use_global_stats=False
                 
        for i in range(len(net_trainval.layer)):
            if re.match(r'res5\w_branch[1-2]\w?$', net_trainval.layer[i].name):
                net_trainval.layer[i].param[0].lr_mult=0.1
                net_trainval.layer[i].param[0].decay_mult=1
            if re.match(r'scale5\w_branch[1-2]\w?$', net_trainval.layer[i].name):
                net_trainval.layer[i].param[0].lr_mult=0.1
                net_trainval.layer[i].param[0].decay_mult=1
                net_trainval.layer[i].param[1].lr_mult=0.2
                net_trainval.layer[i].param[1].decay_mult=0
                
        with open(dst_resnet50_trainval_prototxt, 'w') as f:
                print(text_format.MessageToString(net_trainval), file=f)


    def createSolverPrototxt(self):
        num_val_image=self.__num_val_image
        save_dir=self.__save_dir
        batch_size_val=self.__batch_size_val
        base_lr=self.__base_lr
        weight_decay=self.__weight_decay
        log_dir=self.__log_dir

        solver_file="{}/solver.prototxt".format(save_dir)
        test_iter=int(math.ceil(float(num_val_image)/batch_size_val))
        dst_resnet50_trainval_prototxt="{}/my_trainval.prototxt".format(save_dir)
        solver_param = {
            'net': dst_resnet50_trainval_prototxt,
            'test_iter': [test_iter],
            'test_interval': 100,
            'base_lr': base_lr,
            'lr_policy': "step",
            'gamma': 0.5,
            'stepsize': 1000, 
            'display': 20,
            'max_iter': 10000,
            'momentum': 0.9,
            'weight_decay': weight_decay,
            'snapshot': 5000,
            'snapshot_prefix': os.path.join(log_dir, "snapshot"),
            'solver_mode': 'GPU',
            }
        solver = caffe_pb2.SolverParameter(**solver_param)
        with open(solver_file, 'w') as f:
            print(solver, file=f)

    def createPrototxts(self):
        self.createTrainValPrototxt()
        self.createDeployPrototxt()
        self.createSolverPrototxt()


    #model 1:The first 3 chars are label. 2:Each directory has a label.
    def createTxtFiles(self, test_percent, mode):
        data_dir=self.__data_dir
        save_dir=self.__save_dir
        file_operation=self.__file_operation

        dir_list=file_operation.getDirs(data_dir)
        train_txt=save_dir+"/"+"train.txt"
        val_txt=save_dir+"/"+"val.txt"
        classes=set()
        train_data=[]
        val_data=[]
        label=-1
        for dir_name in dir_list:
            dir_last_name=dir_name.split("/")[-1]
            if mode==1:
                label=int(dir_last_name[0:3])
            if mode==2:
                label=label+1
            classes.add(label)
            image_list=file_operation.getAllImageFiles(dir_name)
            train_list, val_list=file_operation.splitImagesToTrainTest(image_list,test_percent)
            train_data.extend(file_operation.addLabel(train_list, label))
            val_data.extend(file_operation.addLabel(val_list, label))
        file_operation.saveDataFile(train_data, train_txt)
        file_operation.saveDataFile(val_data, val_txt)

        self.__num_train_image = len(train_data)
        self.__num_val_image = len(val_data)
        self.__num_classes=len(classes)


    def createDataForTrain(self):
        data_dir=self.__data_dir
        save_dir=self.__save_dir
        resize_width=self.__resize_width
        resize_height=self.__resize_height
        caffe_tool_dir=self.__caffe_tool_dir

        self.createTxtFiles(0.2, 2)
        lmdb_list=[["train.txt", "train_lmdb"], ["val.txt", "val_lmdb"]]
        for ld in lmdb_list:
            create_lmdb_cmd="GLOG_logtostderr=1 "+os.path.join(caffe_tool_dir, "convert_imageset")+" --resize_height="+str(resize_height)+\
            " --resize_width="+str(resize_width)+" --shuffle --encoded --encode_type=png "+"/ "+os.path.join(save_dir, ld[0])+" "+\
            os.path.join(save_dir, ld[1])
            os.system(create_lmdb_cmd)

        mean_cmd=os.path.join(caffe_tool_dir, "compute_image_mean")+" "+os.path.join(save_dir, "train_lmdb")+" "+os.path.join(save_dir, "mean.binaryproto")
        os.system(mean_cmd)


    def train(self):
        save_dir=self.__save_dir
        pretrained_model_dir=self.__pretrained_model_dir
        gpu=self.__gpu
        train_soon=self.__train_soon
        caffe_tool_dir=self.__caffe_tool_dir
        file_operation=self.__file_operation
        log_dir=self.__log_dir

        solver_file="{}/solver.prototxt".format(save_dir)
        file_operation.createDir(log_dir)
        if train_soon:
            cmd=caffe_tool_dir+"/caffe train --solver="+solver_file+" --weights="+\
            os.path.join(pretrained_model_dir, "ResNet-50-model.caffemodel")+" --log_dir="+log_dir+" --gpu="+gpu
            os.system(cmd)


def main():
    args=parseCommand()
    data_dir=args.data_dir
    save_dir=args.save_dir
    resize_width=args.resize_width
    resize_height=args.resize_height
    caffe_tool_dir=args.caffe_tool_dir
    batch_size_train=args.batch_size_train
    batch_size_val=args.batch_size_val
    src_prototxt_dir=args.src_prototxt_dir
    base_lr=args.base_lr
    weight_decay=args.weight_decay
    pretrained_model_dir=args.pretrained_model_dir
    gpu=args.gpu
    train_soon=args.train_soon
    log_dir=args.log_dir

    model=ResNetClassificationModel(data_dir, save_dir, resize_width,\
        resize_height, caffe_tool_dir, batch_size_train, batch_size_val,\
        src_prototxt_dir, base_lr, weight_decay, pretrained_model_dir, gpu, train_soon, log_dir)
    model.createDataForTrain()
    model.createPrototxts()
    model.train()

if __name__ == "__main__":
    main()

#python train_resnet50.py /home/jiangzhiqi/Documents/deep-residual-networks/ResNet /home/jiangzhiqi/Documents/SSD-github/caffe/jiangzhiqi/vehicle_type_dir/generated ~/Documents/caffe/build/tools ~/Documents/SSD-github/caffe/jiangzhiqi/vehicle_type_dir/data --resize_height=256 --resize_width=256 --batch_size_train=2 --batch_size_val=1 --pretrained_model_dir=/home/jiangzhiqi/Documents/deep-residual-networks/ResNet --log_dir=logs --train
	