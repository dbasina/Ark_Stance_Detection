import re
import numpy as np

def parse_individual_results(file_path):
    # Read the text file
    with open(file_path, "r") as f:
        text = f.read()

    # Patterns to capture values after each epoch
    student_acc_pattern = r"Student ACCURACY\s*=\s*([\d\.]+)"
    teacher_acc_pattern = r"Teacher ACCURACY\s*=\s*([\d\.]+)"
    student_auc_pattern = r"Student mAUC\s*=\s*([\d\.]+)"
    teacher_auc_pattern = r"Teacher mAUC\s*=\s*([\d\.]+)"

    # Extract using regex and convert to float
    student_acc = np.array([float(x) for x in re.findall(student_acc_pattern, text)])
    teacher_acc = np.array([float(x) for x in re.findall(teacher_acc_pattern, text)])
    student_auc = np.array([float(x) for x in re.findall(student_auc_pattern, text)])
    teacher_auc = np.array([float(x) for x in re.findall(teacher_auc_pattern, text)])

    return student_acc, teacher_acc, student_auc, teacher_auc

def parse_joint_training_results(file_path, datasets):
    # Read the text file
    with open(file_path, "r") as f:
        text = f.read()

    results = {
        ds: {
            "student_acc": [],
            "teacher_acc": [],
            "student_mauc": [],
            "teacher_mauc": []
        } for ds in datasets
    }

    # For each dataset, find the per-epoch block of that dataset and grab ACCURACY & mAUC
    for ds in datasets:
        pattern = rf"""
            {ds}:\s*                       # dataset header
            Student\ ACCURACY\s*=\s*([0-9]*\.[0-9]+)\s*,\s*
            Teacher\ ACCURACY\s*=\s*([0-9]*\.[0-9]+)
            [\s\S]*?                       # skip the AUC-by-class lines
            {ds}:\s*Student\ mAUC\s*=\s*([0-9]*\.[0-9]+)\s*,\s*
            Teacher\ mAUC\s*=\s*([0-9]*\.[0-9]+)
        """
        for m in re.finditer(pattern, text, flags=re.VERBOSE):
            s_acc, t_acc, s_mauc, t_mauc = map(float, m.groups())
            results[ds]["student_acc"].append(s_acc)
            results[ds]["teacher_acc"].append(t_acc)
            results[ds]["student_mauc"].append(s_mauc)
            results[ds]["teacher_mauc"].append(t_mauc)

    # Convert to numpy arrays
    for ds in datasets:
        for k in results[ds]:
            results[ds][k] = np.array(results[ds][k], dtype=float)

    return results

def generate_plot(baseline_student_acc, joint_student_acc, joint_teacher_acc, title, xlabel, ylabel, save_path):
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(baseline_student_acc) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, baseline_student_acc, label='Bert Baseline Student '+ ylabel, marker='o')
    plt.plot(epochs, joint_student_acc, label='ARK Bert Student '+ ylabel, marker='o')
    plt.plot(epochs, joint_teacher_acc, label='ARK Bert Teacher '+ ylabel, marker='o')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(epochs, labels=['']*len(epochs))
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()