# Evaluation
LLaVA-1.5 evaluates models on a diverse set of 12 benchmarks. 
To ensure the reproducibility, they evaluate the models with greedy decoding. 
They do not evaluate using beam search to make the inference process consistent with the chat demo of real-time outputs.

The checkpoints of LLaVA are placed in "outputs/llava_ckpts"


### 1. VQAv2

1. Download [`test2015`](http://images.cocodataset.org/zips/test2015.zip) and put it under `$ROOT_DIR/playground/vqav2`.
2. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/llava_eval/vqav2.sh [CKPT_NAME]
```
3. Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/830/my-submission): `./playground/data/eval/vqav2/answers_upload`.


### 2. GQA

1. Download the [data](https://cs.stanford.edu/people/dorarad/gqa/download.html) and [evaluation scripts](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html) following the official instructions and put under `$ROOT_DIR/playground/gqa/data`. You may need to modify `eval.py` as [this](https://gist.github.com/haotian-liu/db6eddc2a984b4cbcc8a7f26fd523187) due to the missing assets in the GQA v1.2 release.
2. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/llava_eval/gqa.sh
```


### 3. VisWiz

1. Download [`test.json`](https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip) and extract [`test.zip`](https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip) to `test`. Put them under `$ROOT_DIR/playground/vizwiz`.
2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/llava_eval/vizwiz.sh
```
* CKPT_NAME:
* IMG_SCALE:
* CONV_MODE:
* LLM_NAME: 

3. Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/1911/my-submission): `./playground/data/eval/vizwiz/answers_upload`.


### 4. ScienceQA

1. Under `$ROOT_DIR/playground/scienceqa`, download `images`, `pid_splits.json`, `problems.json` from the `data/scienceqa` folder of the ScienceQA [repo](https://github.com/lupantech/ScienceQA).
2. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/llava_eval/sqa.sh
```


### 5. TextVQA

1. Download [`TextVQA_0.5.1_val.json`](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) and [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) and extract to `$ROOT_DIR/playground/textvqa`.
2. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/llava_eval/textvqa.sh
```


### 6. POPE

1. Download `coco` from [COCO 2014 val](http://images.cocodataset.org/zips/val2014.zip) and annotations from [POPE](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco) and put under `$ROOT_DIR/playground/pope`.
2. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/llava_eval/pope.sh
```


### 7. MME

1. Download the data following the official instructions [here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).
2. Downloaded images to `MME_Benchmark_release_version`.
3. put the official `eval_tool` and `MME_Benchmark_release_version` under `./playground/data/eval/MME`.
4. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/llava_eval/mme.sh
```


### 8. MMBench

1. Download [`mmbench_dev_20230712.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv) and put under `$ROOT_DIR/playground/mmbench`.
2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/llava_eval/mmbench.sh
```
3. Submit the results to the [evaluation server](https://opencompass.org.cn/leaderboard-multimodal): `./playground/data/eval/mmbench/answers_upload/mmbench_dev_20230712`.


### 9. MMBench-CN

1. Download [`mmbench_dev_cn_20231003.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv) and put under `$ROOT_DIR/playground/mmbench`.
2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/llama_eval/mmbench_cn.sh
```
3. Submit the results to the evaluation server: `./playground/data/eval/mmbench/answers_upload/mmbench_dev_cn_20231003`.


### 10. SEED-Bench

1. Following the official [instructions](https://github.com/AILab-CVC/SEED-Bench/blob/main/DATASET.md) to download the images and the videos. Put images under `$ROOT_DIR/playground/seed_bench/SEED-Bench-image`.
2. Extract the video frame in the middle from the downloaded videos, and put them under `$ROOT_DIR/playground/seed_bench/SEED-Bench-video-image`. We provide our script `extract_video_frames.py` modified from the official one.
3. Multiple-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/llava_eval/seed.sh
```
4. Optionally, submit the results to the leaderboard: `./playground/data/eval/seed_bench/answers_upload` using the official jupyter notebook.


### 11. LLaVA-Bench-in-the-Wild

1. Extract contents of [`llava-bench-in-the-wild`](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild) to `$ROOT_DIR/playground/llava-bench-in-the-wild`.
2. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/llava_eval/llavabench.sh
```


### 12. MM-Vet

1. Extract [`mm-vet.zip`](https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip) to `$ROOT_DIR/playground/mmvet`.
2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/llava_eval/mmvet.sh
```
3. Evaluate the predictions in `./playground/data/eval/mmvet/results` using the official jupyter notebook.
