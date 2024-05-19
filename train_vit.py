from datasets import load_dataset
dataset = load_dataset('imagenet-1k', split="train", cache_dir='/workspace/imagenet', trust_remote_code=True)
print(dataset[0])