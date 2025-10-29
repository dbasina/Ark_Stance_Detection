from utils import MetricLogger, ProgressLogger, stance_detect_accuracy, metric_AUROC
from sklearn.metrics import accuracy_score
import time
import torch
import numpy as np
import sys
import copy
from tqdm import tqdm
# import wandb

def train_one_epoch(model, use_head_n, dataset, data_loader_train, device, criterion, optimizer, epoch, ema_mode, teacher, momentum_schedule, coef_schedule, it):
    batch_time = MetricLogger('Time', ':6.3f')
    losses_cls = MetricLogger('Loss_'+dataset+' cls', ':.4e')
    losses_mse = MetricLogger('Loss_'+dataset+' mse', ':.4e')
    progress = ProgressLogger(
        len(data_loader_train),
        [batch_time, losses_cls, losses_mse],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    MSE = torch.nn.MSELoss()
    # coefficient scheduler from  0 to 0.5 
    coff = coef_schedule[it]
    print("Teacher_Loss_Coef: ",coff)
    end = time.time()
    for i, batch in enumerate(data_loader_train):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        stance = batch["stance"].to(device)
        mask_index = batch['mask_index'].to(device)
        
        with torch.no_grad():
            feat_t, pred_t = teacher(input_ids, attention_mask, token_type_ids, mask_index, use_head_n)
        feat_s, pred_s = model(input_ids, attention_mask, token_type_ids, mask_index, use_head_n)

        # Debug: Print prediction shape on first iteration  
        if i == 0:
            print(f"##DEBUG## Prediction shape: {pred_s.shape}")

        loss_cls = criterion(pred_s, stance)
        loss_const = MSE(feat_s, feat_t)
        
        loss = (1-coff) * loss_cls + coff * loss_const

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_cls.update(loss_cls.item(), input_ids.size(0))
        losses_mse.update(loss_const.item(), input_ids.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            progress.display(i)

        if ema_mode == "iteration":
            ema_update_teacher(model, teacher, momentum_schedule, it)
            it += 1

    if ema_mode == "epoch":
        ema_update_teacher(model, teacher, momentum_schedule, it)
        it += 1

def ema_update_teacher(model, teacher, momentum_schedule, it):
    with torch.no_grad():
        m = momentum_schedule[it]  # momentum parameter
        for param_q, param_k in zip(model.parameters(), teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

def evaluate(model, use_head_n, data_loader_val, device, criterion, dataset):
    model.eval()

    with torch.no_grad():
        batch_time = MetricLogger('Time', ':6.3f')
        losses = MetricLogger('Loss', ':.4e')
        progress = ProgressLogger(
        len(data_loader_val),
        [batch_time, losses], prefix='Val_'+dataset+': ')

        end = time.time()
        for i, (batch) in enumerate(data_loader_val):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            stance = batch["stance"].to(device)
            mask_index = batch['mask_index'].to(device)

            _, outputs = model(input_ids, attention_mask, token_type_ids, mask_index, use_head_n)
            loss = criterion(outputs, stance)
            losses.update(loss.item(), input_ids.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                progress.display(i)

    return losses.avg


def test_classification_unmasked(model, use_head_n, data_loader_test, device):
    model.eval()
    preds_cpu = []
    labels_cpu = []
    masks_cpu = []
    with torch.no_grad():
        for batch in data_loader_test:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            word_one_hot   = batch["word_one_hot"].to(device)
            stance         = batch["stance"]              
            known_mask     = batch["known_mask"]        

            _, out = model(input_ids, attention_mask, token_type_ids, word_one_hot, use_head_n)

            # if stance_detection:
            #     out = torch.softmax(out, dim=1)   # [B, C] multiclass probs
            # else:
            #     out = torch.sigmoid(out)          # [B, C] multilabel probs
            out = torch.sigmoid(out)
            preds_cpu.append(out.detach().cpu())
            labels_cpu.append(stance.detach().cpu())
            masks_cpu.append(known_mask.detach().cpu())

    y_test = torch.cat(labels_cpu, dim=0)   # [N, C] or [N]
    p_test = torch.cat(preds_cpu,  dim=0)   # [N, C]
    known_mask = torch.cat(masks_cpu, dim=0)
    print("##DEBUG## y_test.shape, p_test.shape", y_test.shape, p_test.shape)
    return y_test, p_test, known_mask

def test_classification(model, use_head_n, data_loader_test, device):

    model.eval()
    preds_cpu = []
    labels_cpu = []
    with torch.no_grad():
        for batch in data_loader_test:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            stance         = batch["stance"].to(device)              
            mask_index     = batch['mask_index'].to(device)

            fused, out = model(input_ids, attention_mask, token_type_ids, mask_index, use_head_n)

            # if stance_detection:
            #     out = torch.softmax(out, dim=1)   # [B, C] multiclass probs
            # else:
            #     out = torch.sigmoid(out)          # [B, C] multilabel probs
            out = torch.sigmoid(out)
            preds_cpu.append(out.detach().cpu())
            labels_cpu.append(stance.detach().cpu())


    test_targets = torch.cat(labels_cpu, dim=0)   # [N, C] or [N]
    test_predictions = torch.cat(preds_cpu,  dim=0)   # [N, C]

    print("##DEBUG## test_targets.shape, test_predictions.shape", test_targets.shape, test_predictions.shape)
    return test_targets, test_predictions
    
def mid_epoch_eval(model,teacher, most_recent_dataset, dataset_list, data_loaders_list, device, epoch, datasets_config, mid_epoch_eval_file): 
    model.eval()
    teacher.eval()
    print ("## STAGE ## Performing mid-epoch evaluation...")
    with torch.no_grad():
        for i,dataset in enumerate(dataset_list):
            data_loader = data_loaders_list[i]
            diseases = datasets_config[dataset]['diseases']
            task_type = datasets_config[dataset]['task_type']
            
            if task_type == "multi-class classification":
                multiclass = True
            else:
                multiclass = False

            y_test_student, p_test_student = test_classification(model, i, data_loader, device, multiclass)
            y_test_teacher, p_test_teacher = test_classification(teacher, i, data_loader, device, multiclass)

            if dataset == "CheXpert":
                test_diseases_name = datasets_config['CheXpert']['test_diseases_name']
                test_diseases = [diseases.index(c) for c in test_diseases_name]

                y_test_student = copy.deepcopy(y_test_student[:,test_diseases])
                p_test_student = copy.deepcopy(p_test_student[:, test_diseases])
                individual_results = metric_AUROC(y_test_student, p_test_student, len(test_diseases))

                y_test_teacher = copy.deepcopy(y_test_teacher[:,test_diseases])
                p_test_teacher = copy.deepcopy(p_test_teacher[:, test_diseases])
                individual_results_teacher = metric_AUROC(y_test_teacher, p_test_teacher, len(test_diseases)) 
            else: 
                individual_results_student = metric_AUROC(y_test_student, p_test_student, len(diseases))
                individual_results_teacher = metric_AUROC(y_test_teacher, p_test_teacher, len(diseases))

            # Compute AUC for all task types using one-vs-rest approach
            
            # individual_results_student = metric_AUROC(y_test_student, p_test_student, len(diseases))
            # individual_results_teacher = metric_AUROC(y_test_teacher, p_test_teacher, len(diseases))
            mean_auroc_student = np.nanmean(individual_results_student)
            mean_auroc_teacher = np.nanmean(individual_results_teacher)
            result_str = f"Epoch {epoch}, Recent Dataset {most_recent_dataset}, Test Dataset {dataset} ({task_type}): Student AUC: {mean_auroc_student:.4f}, Teacher AUC: {mean_auroc_teacher:.4f}"
            
            with open(mid_epoch_eval_file, "a") as f:
                f.write(result_str + "\n")
    print("## STAGE ## Mid-epoch evaluation complete. Results appended to", mid_epoch_eval_file)

def write_logs(output_file, model, device, teacher, epoch, datasets_config, dataset_list, data_loader_list_test, val_loss_list):
    
    with open(output_file, 'a') as writer:
        writer.write("Omni-pretraining stage:\n")
        writer.write("Epoch {:04d}:\n".format(epoch))
        student_accuracy_result, teacher_accuracy_result = [],[]
        student_AUC_result, teacher_AUC_result = [],[]
        for i, dataset in enumerate(dataset_list):

            writer.write("{} Validation Loss = {:.5f}:\n".format(dataset, val_loss_list[i]))
            stances = datasets_config[dataset]['stance_list']
            
            # Get predictions and targets on the test sets
            targets_test_student, predictions_test_student = test_classification(model, i, data_loader_list_test[i], device)
            targets_test_teacher, predictions_test_teacher = test_classification(teacher, i, data_loader_list_test[i], device)
            
            # Compute Accuracies
            acc_student = stance_detect_accuracy(targets_test_student, predictions_test_student)
            acc_teacher = stance_detect_accuracy(targets_test_teacher, predictions_test_teacher)


            student_accuracy_result.append(acc_student)
            teacher_accuracy_result.append(acc_teacher)

            # Compute AUCs. We only consider target aspects that have at least one positive and one negative sample in known_mask
            individual_results = metric_AUROC(targets_test_student, predictions_test_student, len(stances))
            individual_results_teacher = metric_AUROC(targets_test_teacher, predictions_test_teacher, len(stances))
            

            mean_over_all_classes = np.nanmean(individual_results)
            mean_over_all_classes_teacher = np.nanmean(individual_results_teacher)

            student_AUC_result.append(mean_over_all_classes)
            teacher_AUC_result.append(mean_over_all_classes_teacher)

            # Logging Prints and Writes
            # Print accuracy results
            print(">>{}:Student ACCURACY = {}, \nTeacher ACCURACY = {}\n".format(dataset,acc_student, acc_teacher))
            writer.write(
                "{}: \nStudent ACCURACY = {}, \nTeacher ACCURACY = {}\n".format(dataset, np.array2string(np.array(acc_student), precision=4, separator='\t'), np.array2string(np.array(acc_teacher), precision=4, separator='\t')))
            
            # Print individual AUC results
            print(">>{}:Student AUC = {}, \nTeacher AUC = {}\n".format(dataset, np.array2string(np.array(individual_results), precision=4, separator='\t'),np.array2string(np.array(individual_results_teacher), precision=4, separator='\t')))
            writer.write(
                "{}: \nStudent AUC = {}, \nTeacher AUC = {}\n".format(dataset, np.array2string(np.array(individual_results), precision=4, separator='\t'),np.array2string(np.array(individual_results_teacher), precision=4, separator='\t')))
            
            # Print mean AUC results

            print(">>{}: Student mAUC = {:.4f}, Teacher mAUC = {:.4f}".format(dataset, mean_over_all_classes,mean_over_all_classes_teacher))
            writer.write("{}: Student mAUC = {:.4f}, Teacher mAUC = {:.4f}\n\n".format(dataset, mean_over_all_classes,mean_over_all_classes_teacher))

            
        writer.close()

    return student_accuracy_result, teacher_accuracy_result, student_AUC_result, teacher_AUC_result