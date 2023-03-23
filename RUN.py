import os

for algo in ['fucker-sq8', 'fucker-fp16', 'faissFS']:
    for dataset in ['sift-128-euclidean', 'glove-100-angular', 'gist-960-euclidean', 'deep-image-96-angular']:
        for topk in [1, 10, 100]:
            os.system(
                f'python run.py --dataset {dataset} --algorithm {algo} --local --runs 100  --batch --count {topk}')
