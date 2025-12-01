import matplotlib.pyplot as plt
import re
import numpy as np
from utils import parse_individual_results, parse_joint_training_results, generate_plot

individual_politic_results = "/home/divesh-basina/Documents/cse573/Project/Ark_Aspect_Stance_Detection/Outputs/stance_bert_masked_Random/Ark_MaskedABSA_politic_MaskedABSA_politic_individual_baseline_results.txt"
individual_race_results = "/home/divesh-basina/Documents/cse573/Project/Ark_Aspect_Stance_Detection/Outputs/stance_bert_masked_Random/Ark_MaskedABSA_race_MaskedABSA_race_individual_baseline_results.txt"
joint_training_results = "/home/divesh-basina/Documents/cse573/Project/Ark_Aspect_Stance_Detection/Outputs/stance_bert_masked_Random/Ark_MaskedABSA_race_MaskedABSA_politic_Joint_Training_results.txt"

# Parse the results files
datasets = ["MaskedABSA_race", "MaskedABSA_politic"]
results = parse_joint_training_results(joint_training_results, datasets)
baseline_politic_student_acc, baseline_politic_teacher_acc, baseline_politic_student_auc, baseline_politic_teacher_auc = parse_individual_results(individual_politic_results)
baseline_race_student_acc, baseline_race_teacher_acc, baseline_race_student_auc, baseline_race_teacher_auc = parse_individual_results(individual_race_results)

joint_politic_student_acc = results["MaskedABSA_politic"]["student_acc"]
joint_politic_teacher_acc = results["MaskedABSA_politic"]["teacher_acc"]
joint_politic_student_auc = results["MaskedABSA_politic"]["student_mauc"]
joint_politic_teacher_auc = results["MaskedABSA_politic"]["teacher_mauc"]

joint_race_student_acc = results["MaskedABSA_race"]["student_acc"]
joint_race_teacher_acc = results["MaskedABSA_race"]["teacher_acc"]
joint_race_student_auc = results["MaskedABSA_race"]["student_mauc"]
joint_race_teacher_auc = results["MaskedABSA_race"]["teacher_mauc"]

# Accuracy Plots
# Plot Politic Accuracy: baseline_politic_student_acc, joint_politic_student_acc, joint_politic_teacher_acc
generate_plot(
    baseline_politic_student_acc,
    joint_politic_student_acc,
    joint_politic_teacher_acc,
    "Politic Accuracy Comparison",
    "Epochs",
    "Accuracy",
    "PerformancePlots/politic_accuracy_comparison.png"
)
# Plot Race Accuracy: baseline_race_student_acc, joint_race_student_acc, joint_race_teacher_acc
generate_plot(
    baseline_race_student_acc,
    joint_race_student_acc,
    joint_race_teacher_acc,
    "Race Accuracy Comparison",
    "Epochs",
    "Accuracy",
    "PerformancePlots/race_accuracy_comparison.png"
)

# AUC Plots
# Plot Politic AUC: baseline_politic_student_auc, joint_politic_student_auc, joint_politic_teacher_auc
generate_plot(
    baseline_politic_student_auc,
    joint_politic_student_auc,
    joint_politic_teacher_auc,
    "Politic AUC Comparison",
    "Epochs",
    "AUC",
    "PerformancePlots/politic_auc_comparison.png"
)

# Plot Race AUC: baseline_race_student_auc, joint_race_student_auc, joint_race_teacher_auc
generate_plot(
    baseline_race_student_auc,
    joint_race_student_auc,
    joint_race_teacher_auc,
    "Race AUC Comparison",
    "Epochs",
    "AUC",
    "PerformancePlots/race_auc_comparison.png"
)