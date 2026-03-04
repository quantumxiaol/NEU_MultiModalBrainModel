from transformer_experiment_runner import run_transformer_experiment


if __name__ == "__main__":
    run_transformer_experiment(
        atlas_path="./ABIDEdata/pcc_correlation_871_cc400_.mat",
        label_path="./ABIDEdata/871_label_cc400.txt",
        model_save_dir="./modelstfcc400",
        attention_output_root="./modelstfcc400/attention_maps",
    )
