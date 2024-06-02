from datasets import load_dataset, load_metric

def dataloader(train_csv_path, eval_csv_path):
    data_files = {
        "train": train_csv_path,
        "validation": eval_csv_path,
    }

    dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    print(train_dataset)
    print(eval_dataset)
    
    return train_dataset, eval_dataset