import torch

from transformers import AutoTokenizer, AutoModel

base_model = "./chatglm2-6b/"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModel.from_pretrained(
    base_model, 
    load_in_8bit=True,
    torch_dtype=torch.float16,
    trust_remote_code=True, 
    device='cuda'
)
model = model.eval()

instruction = "以下是王者荣耀游戏中，一个玩家的操作行为，你需要先回答“实锤作弊”，\"可疑行为\"或者“无违规”，然后再解析原因"
input_data = '''使用英雄干将莫邪, 该英雄属于法师。{"前5秒": {"位移": {"可疑": "首个入侵,首个进入草丛,单人靠近,无视野靠近,无视野靠近:单人靠近", "正常": "有视野靠近,有视野靠近:单人靠近,有视野靠近:敌方主动靠近", "中立": "靠近活着的队友,靠近可见的队手,靠近自己家"}, "行为": {"可疑": "命中无视野：t-2时刻无视野,命中无视野：t-5时刻无视野", "正常": "命中有视野，针对攻击目标,命中有视野：t-2时刻有视野,命中时，t~t-2时段有视野", "中立": "技能,清完兵线"}}, "前10秒": {"位移": {"正常": "有视野靠近,有视野靠近:单人靠近,有视野靠近:敌方主动靠近", "中立": "靠近活着的队友,靠近可见的队手,靠近自己家"}, "行为": {"可疑": "单人击杀,单人击杀：针对攻击目标", "正常": "技能探草丛,命中有视野，针对攻击目标,命中时，t~t-2时段有视野,命中时，t~t-3时段有视野", "中立": "技能"}, "状态": {"可疑": "残血时间"}}, "前15秒": {"位移": {"可疑": "经过草丛，但没有进入草丛", "中立": "靠近自己家,靠近活着的队友,靠近可见的队手"}, "行为": {"可疑": "单人击杀：针对攻击目标", "正常": "处在塔的保护范围下"}}, "前20秒": {"位移": {"可疑": "经过草丛，但没有进入草丛", "中立": "靠近敌方家,靠近可见的队手,远离活着的队友"}, "行为": {"正常": "我方野区,处在塔的保护范围下"}}}'''
instruction += "\n" + input_data

print(instruction)
response, history = model.chat(tokenizer, instruction, history=[])
print(response)

# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
# print(response)



