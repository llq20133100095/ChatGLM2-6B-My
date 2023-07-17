import os
import sys

import fire
import gradio as gr
import torch
import transformers
import peft
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import time
import torch.distributed as dist
import numpy as np

from utils_custom.prompter import Prompter
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModel

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

cutoff_len = 1248 
train_on_inputs: bool = True  # if False, masks out inputs in loss
add_eos_token: bool = False
base_model = "./chatglm2-6b/"
data_path = "data/record/test_20230317_20230510_is_guilty.json"
lora_weights = None

def ddp_setup():
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    os.environ["MASTER_ADDR"] = "localhost"
    local_rank = int(os.environ['LOCAL_RANK'])
    init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

prompter = Prompter("")
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

# 补充0值在最左边： 0,0,0,0,A,B....
# tokenizer.padding_side = "left"
# tokenizer.pad_token_id = 0

class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        # image = torch.Tensor(sample['input_text']).to(torch.int64)
        image = sample['input_text']
        label = sample['labels']
        return image, label

def build_model(
    load_8bit: bool = True,
    base_model: str = "llama-7b-hf",
    lora_weights: str = "lora-alpaca-zh-wangzhe1248",
    template_name: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = True,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    if device == "cuda":
        model = AutoModel.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device_map,
        )

        # model = PeftModel.from_pretrained(
        #     model,
        #     lora_weights,
        #     torch_dtype=torch.float16,
        #     device_map=device_map,
        # )
    else:
        model = AutoModel.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True, trust_remote_code=True,
        )
        # model = PeftModel.from_pretrained(
        #     model,
        #     lora_weights,
        #     device_map={"": device},
        # )

    # model_vocab_size = model.get_input_embeddings().weight.size(0)
    # tokenzier_vocab_size = len(tokenizer)
    # print(f"Vocab of the base model: {model_vocab_size}")
    # print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
    # if model_vocab_size != tokenzier_vocab_size:
    #     assert tokenzier_vocab_size > model_vocab_size
    #     print("Resize model embeddings to fit tokenizer")
    #     model.resize_token_embeddings(tokenzier_vocab_size)
    # if lora_weights is not None:
    #     print("loading peft model")
    #     model = PeftModel.from_pretrained(
    #         model,
    #         lora_weights,
    #         load_in_8bit=load_8bit,
    #         torch_dtype=torch.float16,
    #         device_map=device_map,
    #     )
    # else:
    #     model = model

    # # unwind broken decapoda-research config
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2

    # if not load_8bit:
    #     model.half()  # seems to fix bugs for some users.

    model = model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # model.config.use_cache = False
    return model

def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    # if (
    #     result["input_ids"][-1] != tokenizer.eos_token_id
    #     and len(result["input_ids"]) < cutoff_len
    #     and add_eos_token
    # ):
    #     result["input_ids"].append(tokenizer.eos_token_id)
    #     result["attention_mask"].append(1)

    # result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
    )
    # tokenized_full_prompt = tokenize(full_prompt)
    tokenized_full_prompt = {}

    tokenized_full_prompt["input_text"] = data_point["instruction"] + "\n" + data_point["input"]
    tokenized_full_prompt["labels"] = data_point["output"]

    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = tokenize(
            user_prompt, add_eos_token=add_eos_token
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably

    return tokenized_full_prompt

def evaluate(
    model,
    input_ids,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=1,
    max_new_tokens=10,
    stream_output=False,
    **kwargs,
):
    # with torch.no_grad():
    device_in = "cuda:" + str(os.environ.get("LOCAL_RANK") or 0)
    # input_ids = input_ids.to(device_in)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    # Without streaming

    # print(input_ids.shape)
    # generation_output = model.generate(
    #     input_ids=input_ids,
    #     generation_config=generation_config,
    #     return_dict_in_generate=True,
    #     output_scores=True,
    #     max_new_tokens=max_new_tokens,
    # )
    # s = generation_output.sequences[0]
    # # print(s.shape)
    # # print(generation_output.shape)
    # all_outpupt = tokenizer.decode(s)
    # output = prompter.get_response(all_outpupt)

    input_ids = str(input_ids[0])
    response, history = model.chat(tokenizer, input_ids, history=[])

    # return prompter.get_response(output)
    # s = generation_output.sequences
    # output = tokenizer.batch_decode(s, skip_special_tokens=True)

    # new_output.append(prompter.get_response(o))
    return input_ids[0], response


if __name__ == "__main__":
    start_time = time.time()
    # fire.Fire(main)
    instruction = ["以下是王者荣耀游戏中，一个玩家的操作行为，判定该玩家是否存在作弊行为"] * 4
    input = [
        "英雄决策是:干将莫邪,第一段时间,跟随靠近敌方:5,敌方主动靠近:5,命中且前8s无视野:5,清完兵线:5,塔下清兵:5,撤退:2,前进:2,等待:1,攻击小兵:5第二段时间,跟随靠近敌方:1,技能:1,敌方主动靠近:1,命中且前8s无视野:5,无视野靠近:3,无视野单人靠近:3,清完兵线:4,塔下清兵:4,撤退:4,前进:1,攻击小兵:3第三段时间,无进入草丛:1,命中有视野:4,命中且前8s无视野:2,命中且前2s有视野:4,命中且前5s有视野:4,到前2s有视野:2,到前3s有视野:2,到前4s有视野:2,无视野靠近:5,有视野靠近:3,无视野单人靠近:5,有视野单人靠近:3,撤退:1,前进:4第四段时间,攻击中立野怪:2,首个入草丛:3,技能探草:2,技能:2,命中有视野:2,多人击杀:3,命中有视野:5,命中且前2s有视野:3,到前2s有视野:5,到前3s有视野:5,到前4s有视野:5,有视野靠近:5,有视野单人靠近:5,多人击杀:3,有大招，大招起手:3,撤退:5,攻击英雄:3", 
        "英雄决策是: 百里守约 第一段时间,敌方攻击，塔下防守: 5, 塔内防守: 5, 技能: 1, 命中有视野: 1, 命中有视野: 5, 命中且前2s有视野: 5, 命中且前5s有视野: 5, 命中且前8s有视野: 5, 到前2s有视野: 5, 到前3s有视野: 5, 到前4s有视野: 5, 有视野靠近: 5, 有视野单人靠近: 5, 没有清完兵线: 3, 塔下清兵: 3, 撤退: 2, 前进: 3, 攻击小兵: 4, 攻击英雄: 2 第二段时间,敌方攻击，塔下防守: 5, 塔内防守: 5, 技能: 1, 命中有视野: 1, 命中有视野: 3, 命中且前2s有视野: 5, 命中且前5s有视野: 5, 命中且前8s有视野: 2, 到前2s有视野: 5, 到前3s有视野: 5, 到前4s有视野: 5, 有视野靠近: 4, 有视野单人靠近: 4, 撤退: 1, 前进: 1, 等待: 3, 攻击英雄: 3 第三段时间,单人靠近敌方: 5, 敌方攻击，塔下防守: 5, 塔内防守: 5, 技能: 2, 命中有视野: 2, 单人击杀: 1, 到前3s无视野: 1, 命中有视野: 4, 命中且前2s有视野: 2, 到前2s有视野: 5, 到前3s有视野: 2, 到前4s有视野: 2, 有视野靠近: 4, 有视野单人靠近: 4, 单人击杀: 1, 残血击杀: 1, 前进: 1, 等待: 4, 攻击英雄: 3 第四段时间,单人靠近敌方: 3, 敌方攻击，塔下防守: 5, 塔内防守: 3, 单人击杀: 3, 清完兵线: 2, 单人击杀: 5, 残血击杀: 5, 塔下清兵: 2, 撤退: 1, 前进: 2, 等待: 2, 攻击小兵: 2, 攻击英雄: 2 ", 
        "英雄决策是: 干将莫邪 第一段时间,跟随靠近敌方: 5, 敌方主动靠近: 5, 命中且前8s无视野: 5, 清完兵线: 5, 塔下清兵: 5, 撤退: 2, 前进: 2, 等待: 1, 攻击小兵: 5 第二段时间,跟随靠近敌方: 1, 技能: 1, 敌方主动靠近: 1, 命中且前8s无视野: 5, 无视野靠近: 3, 无视野单人靠近: 3, 清完兵线: 4, 塔下清兵: 4, 撤退: 4, 前进: 1, 攻击小兵: 3 第三段时间,无进入草丛: 1, 命中有视野: 4, 命中且前8s无视野: 2, 命中且前2s有视野: 4, 命中且前5s有视野: 4, 到前2s有视野: 2, 到前3s有视野: 2, 到前4s有视野: 2, 无视野靠近: 5, 有视野靠近: 3, 无视野单人靠近: 5, 有视野单人靠近: 3, 撤退: 1, 前进: 4 第四段时间,攻击中立野怪: 2, 首个入草丛: 3, 技能探草: 2, 技能: 2, 命中有视野: 2, 多人击杀: 3, 命中有视野: 5, 命中且前2s有视野: 3, 到前2s有视野: 5, 到前3s有视野: 5, 到前4s有视野: 5, 有视野靠近: 5, 有视野单人靠近: 5, 多人击杀: 3, 有大招，大招起手: 3, 撤退: 5, 攻击英雄: 3 ", 
        "英雄决策是: 鲁班七号 \n第一段时间,塔外防守: 3, 技能: 3, 命中无视野: 1, 命中无视野: 1, 命中无视野: 3, 命中且前2s有视野: 2, 有视野靠近: 3, 有视野跟随靠近: 3, 前进: 3, 等待: 2, 攻击英雄: 5 \n第二段时间,塔外防守: 5, 技能: 1, 有视野靠近: 5, 有视野单人靠近: 5, 清完兵线: 5, 塔下清兵: 5, 撤退: 2, 前进: 2, 等待: 1, 攻击小兵: 5, 攻击英雄: 2 \n第三段时间,塔外防守: 5, 技能: 3, 有视野靠近: 3, 有视野单人靠近: 3, 清完兵线: 5, 塔下清兵: 5, 等待: 5, 攻击小兵: 5, 攻击英雄: 2 \n第四段时间,塔外防守: 5, 技能: 1, 命中无视野: 1, 命中无视野: 1, 到前4s无视野: 1, 有视野靠近: 4, 有视野跟随靠近: 4, 有视野敌方靠近: 4, 撤退: 4, 等待: 1, 攻击小兵: 1, 攻击英雄: 3 \n", 
    ]

    ddp_setup()

    """ 1. 导入数据集 """
    data = load_dataset("json", data_files=data_path)
    train_val = data["train"].train_test_split(
        train_size=2000, test_size=2000, shuffle=True, seed=42
    )
    train_data = (
        train_val["train"].map(generate_and_tokenize_prompt)
    )

    # print("input_ids1", len(train_data['input_ids'][0]))
    # print("input_ids2", len(train_data['input_ids'][1]))
    # print("labels", len(train_data['labels']))


    """ 2. 模型构建 """
    model = build_model(
        base_model=base_model, 
        lora_weights=lora_weights
    )

    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()



    # 创建自定义的数据集实例
    custom_dataset = MyDataset(train_data)

    # 初始化分布式采样器
    sampler = torch.utils.data.distributed.DistributedSampler(custom_dataset)

    # 创建数据加载器
    batch_size = 1
    dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size, sampler=sampler)

    # 遍历数据加载器
    prediction = []
    label = []
    for i, batch in enumerate(dataloader):
        inputs, labels = batch
        # if labels[0] == "可疑行为":
        #     continue
    
        # 进行模型训练或推理操作
        all_outpupt, outputs = evaluate(model, inputs)

        if "实锤作弊" in outputs[:4]:
            prediction.append(1)
        elif "可疑行为" in outputs[:4]:
            prediction.append(1)
        else:
            prediction.append(0)

        # print(outputs, labels)

        if labels[0] == "实锤作弊":
            label.append(1)
        elif labels[0] == "可疑行为":
            label.append(1)
        else:
            label.append(0)
    
        if i % 100 == 0:
            print("In %d Use time %f" % (i, time.time() - start_time))

        # if i % 10 == 0 and i != 0:
        #     break

    print("End time", time.time() - start_time)

    device_in = "cuda:" + str(os.environ.get("LOCAL_RANK") or 0)
    prediction = torch.Tensor(prediction).to(device_in)
    label = torch.Tensor(label).to(device_in)

    # 聚集分布式数据
    prediction_data = [torch.zeros_like(prediction) for _ in range(dist.get_world_size())]
    label_data = [torch.zeros_like(label) for _ in range(dist.get_world_size())]
    dist.all_gather(prediction_data, prediction)
    dist.all_gather(label_data, label)


    import torch, gc
    del model
    # gc.collect()
    # torch.cuda.empty_cache()

    # 打印聚集后的完整数据
    labels_nums = 2
    if dist.get_rank() == 0:
        print("Gathered data:")
        new_prediction_data = []
        new_label_data = []
        # for p, l in zip(prediction_data, label_data):
        #     new_prediction_data.append(p.numpy())
        #     new_label_data.append(l.numpy())
        prediction_data = torch.concat(prediction_data, axis=0).cpu()
        label_data = torch.concat(label_data, axis=0).cpu()

        pre, rec, f1, sup = precision_recall_fscore_support(
            label_data.numpy(), prediction_data.numpy())
        
        for i in range(labels_nums):
            print("sup_%d: %d" % (i, sup[i]))
            print(f"Precision: {pre[i]:.3f}, Recall: {rec[i]:.3f}, F1: {f1[i]:.3f}")
            