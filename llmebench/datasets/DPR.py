import datasets
import json

from llmebench.datasets.dataset_base import DatasetBase


class DPRDataset(DatasetBase):
    """
    Generic DPR output dataset loader

    This data loader provides a way to load datasets on HuggingFace Hub and transform
    them into the format required by the framework. Assets using this loader *must*
    provide a `custom_test_split`, which should correspond to a split in the dataset
    as defined on the Hub. Similarly, `custom_train_split` must also be provided for
    few shot assets.

    Attributes
    ----------
    data_dir : str
        Base path of data containing all datasets. Defaults to "data" in the current
        working directory.
    huggingface_dataset_name : str
        Name of the dataset on HuggingFace Hub (e.g. 'sst2')
    column_mapping : dict
        Mapping defining which of the columns in the loaded data are "input" and "label".
        The supplied dict must contain mappings for "input" and "label", and may contain
        other mappings (such as "input_id").
    """

    def __init__(self, huggingface_dataset_name, column_mapping, sub_split, **kwargs):
        self.huggingface_dataset_name = huggingface_dataset_name
        
        # Check for column_mapping
        assert "input" in column_mapping
        assert "label" in column_mapping
        self.column_mapping = column_mapping
        self.sub_split = sub_split

        super(DPRDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {"generic": True}

    @staticmethod
    def get_data_sample():
        return {"input": "Test Input", "label": "0"}
    



    def load_dataset(self, path):
        """Load dataset from JSON or JSONL file."""
        if path.endswith(".json"):
            return json.load(open(path, "r"))
        elif path.endswith(".jsonl"):
            return [json.loads(line.strip()) for line in open(path, "r")]
        else:
            extension = path.split(".")[-1]
            raise ValueError(f"File extension [{extension}] not valid.")


    def load_data(self, data_split, no_labels=False):
        if "nq" not in self.huggingface_dataset_name:
            path = "./datasets/"+self.huggingface_dataset_name.replace("_","")+"/base/dev.json"
        else: 
            path = "./datasets/"+self.huggingface_dataset_name.replace("_","")+"/base/test.json"
            print("loading path ", str(path))
        dataset = self.load_dataset(path)

        data = []
        
        for sample in dataset:
            processed_sample = {}
            if "multi_column" not in self.column_mapping:
                for sample_key, column_name in self.column_mapping.items():
                    processed_sample[sample_key] = sample[column_name]
            else:
                sample["answer"] = sample.pop("answers")
                processed_sample = sample
            data.append(processed_sample)

        return data

    

    @classmethod
    def download_dataset(cls, data_dir, download_url=None, default_url=None):
        # Generic dataset loaders do not refer to a specific dataset to download
        pass




