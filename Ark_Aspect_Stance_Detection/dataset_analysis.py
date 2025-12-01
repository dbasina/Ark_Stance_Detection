from dataloader import MaskedABSA_race
from utils import get_config, count_aspect_occurrences, generate_plots

plots_save_path = "/home/divesh-basina/Documents/cse573/Project/Ark_Aspect_Stance_Detection/Inference/DatasetPlots"
dataset_config = get_config('datasets_config.yaml')
aspect_list = dataset_config['MaskedABSA_race']['aspect_list']
train_dataset = MaskedABSA_race(file_path=dataset_config['MaskedABSA_race']['data_dir'], split='train', aspects_list=aspect_list, annotation_percent=100)
val_dataset = MaskedABSA_race(file_path=dataset_config['MaskedABSA_race']['data_dir'], split='val', aspects_list=aspect_list, annotation_percent=100)
test_dataset = MaskedABSA_race(file_path=dataset_config['MaskedABSA_race']['data_dir'], split='test', aspects_list=aspect_list, annotation_percent=100)


dataset_list = [train_dataset, val_dataset, test_dataset]
aspect_postive_count, aspect_negative_count = count_aspect_occurrences(dataset_list, aspect_list)

output_file = '/home/divesh-basina/Documents/cse573/Project/Ark_Aspect_Stance_Detection/dataset/race_dataset/maskedABSA_race_aspect_distribution.txt'

# Write table to text file
non_polarized_aspect_indices = []
with open(output_file, 'w') as f:
    # Write header
    f.write(f"{'Aspect':<30} {'totals':<10} {'pro_labels':<12} {'anti_labels':<12}\n")
    f.write("-" * 70 + "\n")

    total_pro = sum(aspect_postive_count)
    total_anti = sum(aspect_negative_count)
    total_samples = total_pro + total_anti

    # Write each aspect row
    for i, aspect in enumerate(aspect_list):
        total = aspect_postive_count[i] + aspect_negative_count[i]
        f.write(f"{aspect:<30} {total:<10} {aspect_postive_count[i]:<12} {aspect_negative_count[i]:<12}\n")
        if aspect_postive_count[i] > 0 and aspect_negative_count[i] > 0:
            non_polarized_aspect_indices.append((i,aspect_postive_count[i],aspect_negative_count[i]))

    # Write final total row
    f.write("-" * 70 + "\n")
    f.write(f"{'Total samples':<30} {total_samples:<10} {total_pro:<12} {total_anti:<12}\n")

    # --- Non-polarized aspects section ---
    f.write("\n\nNon-polarized Aspects (have both 'pro' and 'anti' labels):\n")
    f.write(f"{'Aspect':<30} {'pro_labels':<12} {'anti_labels':<12}\n")
    f.write("-" * 60 + "\n")

    for (index, pro_count, anti_count) in non_polarized_aspect_indices:
        f.write(f"{aspect_list[index]:<30} {pro_count:<12} {anti_count:<12}\n")

generate_plots(plots_save_path, aspect_list, aspect_postive_count, aspect_negative_count)