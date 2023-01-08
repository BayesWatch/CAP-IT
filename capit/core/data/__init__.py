def build_dataset(
    dataset_name: str,
    data_dir: str,
    sets_to_include=None,
):
    import datasets as hf_datasets

    if sets_to_include is None:
        sets_to_include = ["train", "validation"]

    dataset = {}
    for set_name in sets_to_include:
        data = hf_datasets.load(
            path=dataset_name,
            split=set_name,
            cache_dir=data_dir,
            task="image-classification",
        )
        dataset[set_name] = data

    return dataset
