
import os
import sys
import shutil
import time
import numpy as np
from optparse import OptionParser
from tqdm import tqdm
import copy

from models import build_omni_model, build_omni_model_from_checkpoint, save_checkpoint
from utils import metric_AUROC, cosine_scheduler, collate_fn_masked, collate_fn_unmasked, unmasked_bce_loss
from sklearn.metrics import accuracy_score

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from trainer import train_one_epoch, test_classification, evaluate, mid_epoch_eval, write_logs

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from functools import partial
import torch.nn as nn
# import wandb

sys.setrecursionlimit(40000)

def ark_engine(args, model_path, output_path, dataset_list, datasets_config, dataset_train_list, dataset_val_list, dataset_test_list):
    device = torch.device(args.device)
    cudnn.benchmark = True

    # logs
    exp = 'Ark'
    for dataset in dataset_list:
        exp += '_' + dataset 
    model_path = os.path.join(model_path, exp)
    model_path = os.path.join(model_path, args.exp_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    log_file = os.path.join(model_path, "train.log")
    mid_epoch_eval_file = os.path.join(output_path, exp+"_"+args.exp_name+"_mid_epoch_eval.txt")
    output_file = os.path.join(output_path, exp+"_"+args.exp_name+"_results.txt")

    # set collate_fn and criterion
    if args.unmasked:
        collate_fn = collate_fn_unmasked
        criterion = unmasked_bce_loss
    else:
        collate_fn = collate_fn_masked
        criterion = nn.BCEWithLogitsLoss()
        print("##DEBUG## Using Masked Collate and BCEWithLogitsLoss")

    # dataloaders for pretraining
    data_loader_list_train = []
    for d in dataset_train_list:
        data_loader_list_train.append(DataLoader(dataset=d, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn))
    data_loader_list_val = []
    for dv in dataset_val_list:
        data_loader_list_val.append(DataLoader(dataset=dv, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn))
    data_loader_list_test = []
    for dt in dataset_test_list: 
        data_loader_list_test.append(DataLoader(dataset=dt, batch_size=int(args.batch_size/2), shuffle=False,
                                        num_workers=int(args.workers/2), pin_memory=True, collate_fn=collate_fn))
        
    print("##DEBUG## Dataset_list:",dataset_list)
    num_stances_list = [len(datasets_config[dataset]['stance_list']) for dataset in dataset_list]
    print("num_stances_list:", num_stances_list)
    for i, dataset in enumerate(dataset_list):
        print(f"##DEBUG## Head {i}: {dataset} -> {len(datasets_config[dataset]['stance_list'])} classes")

    # training setups
    print ("##DEBUG## Dataset List and Num Stances List: ", dataset_list, num_stances_list)
    if args.from_checkpoint:
        model = build_omni_model_from_checkpoint(args, num_stances_list, 'state_dict')
        teacher = build_omni_model_from_checkpoint(args, num_stances_list, 'teacher')     
    else:
        model = build_omni_model(args, num_stances_list)
        teacher = build_omni_model(args, num_stances_list)    
    print(model)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        teacher = torch.nn.DataParallel(teacher)
    model.to(device)
    teacher.to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.model_name} network.")

    # momentum parameter is increased to 1. during training with a cosine schedule
    if args.ema_mode == "epoch":
        momentum_schedule = cosine_scheduler(args.momentum_teacher, 1,
                                               args.pretrain_epochs, len(dataset_list))
        coef_schedule = cosine_scheduler(0, 0.5, args.pretrain_epochs, len(dataset_list))
    elif args.ema_mode == "iteration":
        iters_per_epoch = 0
        for d in data_loader_list_train:
            iters_per_epoch += len(d)
        momentum_schedule = cosine_scheduler(args.momentum_teacher, 1,
                                               args.pretrain_epochs, iters_per_epoch) 
        coef_schedule = cosine_scheduler(0, 0.5, args.pretrain_epochs, iters_per_epoch)

    print ("args.individual: ", args.individual)
    
    #suppress teacher loss coef to zero if args.individual is set
    if args.individual:
        coef_schedule = cosine_scheduler(0, 0, args.pretrain_epochs, len(dataset_list)) 

    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    start_epoch = 0
    init_loss = 999999
    best_val_loss = init_loss
    save_model_path = os.path.join(model_path, exp)

    if args.resume:
        resume = save_model_path + '.pth.tar'
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            init_loss = checkpoint['lossMIN']
            state_dict = checkpoint['state_dict']
            teacher_state_dict = checkpoint['teacher']

            model.load_state_dict(state_dict, strict=True)
            teacher.load_state_dict(teacher_state_dict, strict=True)
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch={:04d}, val_loss={})"
                    .format(resume, start_epoch, init_loss))
            start_epoch += 1
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
        # wandb.init(
        #     # set the wandb project where this run will be logged
        #     project=exp+'_'+args.exp_name,
        #     resume=True
        # )


    with open(log_file, 'a') as log:
            log.write(str(args))
    log.close()

    student_accuracy_test_results, teacher_accuracy_test_results_teacher = [],[]
    student_AUC_test_results, teacher_AUC_test_results = [],[]
    it = start_epoch * len(dataset_list)
    for epoch in range(start_epoch, args.pretrain_epochs):
        for i, data_loader in enumerate(data_loader_list_train): 
            train_one_epoch(model, i, dataset_list[i], data_loader, device, criterion, optimizer, epoch, args.ema_mode, teacher, momentum_schedule, coef_schedule, it)
            #mid_epoch_eval(model, teacher, dataset_list[i], dataset_list, data_loader_list_test, device, epoch, datasets_config, mid_epoch_eval_file)
            it += 1
        val_loss_list = []

        for i, dv in enumerate(data_loader_list_val):
            val_loss = evaluate(model, i, dv, device, criterion, dataset_list[i])
            val_loss_list.append(val_loss)
            # wandb.log({"val_loss_{}".format(dataset_list[i]): val_loss})
        
        avg_val_loss = np.average(val_loss_list)
        if args.val_loss_metric == "average":
            val_loss_metric = avg_val_loss
        else:
            val_loss_metric = val_loss_list[dataset_list.index(args.val_loss_metric)]
        lr_scheduler.step(val_loss_metric)

        # log metrics to wandb
        # wandb.log({"avg_val_loss": avg_val_loss})

        print("Epoch {:04d}: avg_val_loss {:.5f}, saving model to {}".format(epoch, avg_val_loss,save_model_path))
        save_checkpoint({
                'epoch': epoch,
                'lossMIN': val_loss_list,
                'state_dict': model.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                },  filename=save_model_path)

        with open(log_file, 'a') as log:
            log.write("Epoch {:04d}: avg_val_loss = {:.5f} \n".format(epoch, avg_val_loss))
            log.write("     Datasets  : " + str(dataset_list) + "\n")
            log.write("     Val Losses: " + str(val_loss_list) + "\n")
            log.close()

        if epoch % args.test_epoch == 0 or epoch+1 == args.pretrain_epochs:
            save_checkpoint({
                    'epoch': epoch,
                    'lossMIN': val_loss_list,
                    'state_dict': model.state_dict(),
                    'teacher': teacher.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    },  filename=save_model_path+str(epoch))
            
            student_accuracy_result, teacher_accuracy_result, student_AUC_result, teacher_AUC_result = write_logs(output_file, model, device, teacher, epoch, datasets_config, dataset_list, data_loader_list_test, val_loss_list)
            
            student_accuracy_test_results.append(student_accuracy_result)
            teacher_accuracy_test_results_teacher.append(teacher_accuracy_result)
            student_AUC_test_results.append(student_AUC_result)
            teacher_AUC_test_results.append(teacher_AUC_result)

    with open(output_file, 'a') as writer:
        writer.write("\n\nFinal Results after Omni-pretraining on datasets {} for {} epochs:\n".format(dataset_list, args.pretrain_epochs))
        writer.write("Omni-pretraining stage: \nStudent Accuracy = \n{} \nTeacher Accuracy = \n{}\n".format(np.array2string(np.array(student_accuracy_test_results), precision=4, separator='\t'),np.array2string(np.array(teacher_accuracy_test_results_teacher), precision=4, separator='\t')))
        writer.write("Omni-pretraining stage: \nStudent meanAUC = \n{} \nTeacher meanAUC = \n{}\n".format(np.array2string(np.array(student_AUC_test_results), precision=4, separator='\t'),np.array2string(np.array(teacher_AUC_test_results), precision=4, separator='\t')))
    writer.close()


    
