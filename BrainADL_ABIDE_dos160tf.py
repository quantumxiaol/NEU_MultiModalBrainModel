from transformer_experiment_runner import run_transformer_experiment


if __name__ == "__main__":
    run_transformer_experiment(
        atlas_path="./ABIDEdata/pcc_correlation_871_dos160_.mat",
        label_path="./ABIDEdata/871_label_dos160.txt",
        model_save_dir="./modelstfdos160",
        attention_output_root="./modelstfdos160/attention_maps",
    )
