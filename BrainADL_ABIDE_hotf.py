from transformer_experiment_runner import run_transformer_experiment


if __name__ == "__main__":
    run_transformer_experiment(
        atlas_path="./ABIDEdata/pcc_correlation_871_ho_.mat",
        label_path="./ABIDEdata/871_label_ho.txt",
        model_save_dir="./modelstfho",
        attention_output_root="./modelstfho/attention_maps",
    )
