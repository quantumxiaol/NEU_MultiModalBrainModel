from transformer_experiment_runner import run_transformer_experiment


if __name__ == "__main__":
    run_transformer_experiment(
        atlas_path="./ABIDEdata/pcc_correlation_871_ez_.mat",
        label_path="./ABIDEdata/871_label_ez.txt",
        model_save_dir="./modelstfez",
        attention_output_root="./modelstfez/attention_maps",
    )
