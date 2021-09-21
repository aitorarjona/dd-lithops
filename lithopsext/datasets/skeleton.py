from lithopsext.datasetbase import DatasetBase


class MyCustomDataset(DatasetBase):
    def __init__(self):
        super().__init__()


class MyCustomDatasetPartition:
    def __init__(self):
        pass

    @property
    def key(self):
        return None

    def get(self):
        return None
   