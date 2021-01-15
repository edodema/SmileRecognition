from src.utils.refine_dataset import RefineDataset

path = 'datasets/affwild'
dr = RefineDataset(path)

dr.refine()