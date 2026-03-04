from transformer_experiment_runner import run_transformer_experiment


if __name__ == "__main__":
    run_transformer_experiment(
        atlas_path="./ABIDEdata/pcc_correlation_871_tt_.mat",
        label_path="./ABIDEdata/871_label_tt.txt",
        model_save_dir="./modelstftt",
        attention_output_root="./modelstftt/attention_maps",
    )
