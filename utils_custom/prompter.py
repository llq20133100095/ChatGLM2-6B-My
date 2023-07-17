"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


    def generate_prompt_list(
        self,
        instruction: str,
        input: Union[None, str] = None,
    ):
        res_list = []
        for ins, inp in zip(instruction, input):
            res = self.template["prompt_input"].format(
                instruction=ins, input=inp
            )
            res_list.append(res)

        return res_list
    
    # def 

if __name__ == "__main__":
    prompter = Prompter("")
    # print(prompter.template)

    instruction = ["以下是王者荣耀游戏中，一个玩家的操作行为，判定该玩家是否存在作弊行为, 并给出原因"] * 4
    input = [
            "英雄决策是: 干将莫邪 \n第一段时间,跟随靠近敌方: 5, 敌方主动靠近: 5, 命中且前8s无视野: 5, 清完兵线: 5, 塔下清兵: 5, 撤退: 2, 前进: 2, 等待: 1, 攻击小兵: 5 \n第二段时间,跟随靠近敌方: 1, 技能: 1, 敌方主动靠近: 1, 命中且前8s无视野: 5, 无视野靠近: 3, 无视野单人靠近: 3, 清完兵线: 4, 塔下清兵: 4, 撤退: 4, 前进: 1, 攻击小兵: 3 \n第三段时间,无进入草丛: 1, 命中有视野: 4, 命中且前8s无视野: 2, 命中且前2s有视野: 4, 命中且前5s有视野: 4, 到前2s有视野: 2, 到前3s有视野: 2, 到前4s有视野: 2, 无视野靠近: 5, 有视野靠近: 3, 无视野单人靠近: 5, 有视野单人靠近: 3, 撤退: 1, 前进: 4 \n第四段时间,攻击中立野怪: 2, 首个入草丛: 3, 技能探草: 2, 技能: 2, 命中有视野: 2, 多人击杀: 3, 命中有视野: 5, 命中且前2s有视野: 3, 到前2s有视野: 5, 到前3s有视野: 5, 到前4s有视野: 5, 有视野靠近: 5, 有视野单人靠近: 5, 多人击杀: 3, 有大招，大招起手: 3, 撤退: 5, 攻击英雄: 3 \n", 
            "英雄决策是: 百里守约 \n第一段时间,敌方攻击，塔下防守: 5, 塔内防守: 5, 技能: 1, 命中有视野: 1, 命中有视野: 5, 命中且前2s有视野: 5, 命中且前5s有视野: 5, 命中且前8s有视野: 5, 到前2s有视野: 5, 到前3s有视野: 5, 到前4s有视野: 5, 有视野靠近: 5, 有视野单人靠近: 5, 没有清完兵线: 3, 塔下清兵: 3, 撤退: 2, 前进: 3, 攻击小兵: 4, 攻击英雄: 2 \n第二段时间,敌方攻击，塔下防守: 5, 塔内防守: 5, 技能: 1, 命中有视野: 1, 命中有视野: 3, 命中且前2s有视野: 5, 命中且前5s有视野: 5, 命中且前8s有视野: 2, 到前2s有视野: 5, 到前3s有视野: 5, 到前4s有视野: 5, 有视野靠近: 4, 有视野单人靠近: 4, 撤退: 1, 前进: 1, 等待: 3, 攻击英雄: 3 \n第三段时间,单人靠近敌方: 5, 敌方攻击，塔下防守: 5, 塔内防守: 5, 技能: 2, 命中有视野: 2, 单人击杀: 1, 到前3s无视野: 1, 命中有视野: 4, 命中且前2s有视野: 2, 到前2s有视野: 5, 到前3s有视野: 2, 到前4s有视野: 2, 有视野靠近: 4, 有视野单人靠近: 4, 单人击杀: 1, 残血击杀: 1, 前进: 1, 等待: 4, 攻击英雄: 3 \n第四段时间,单人靠近敌方: 3, 敌方攻击，塔下防守: 5, 塔内防守: 3, 单人击杀: 3, 清完兵线: 2, 单人击杀: 5, 残血击杀: 5, 塔下清兵: 2, 撤退: 1, 前进: 2, 等待: 2, 攻击小兵: 2, 攻击英雄: 2 \n", 
            "英雄决策是: 干将莫邪 \n第一段时间,跟随靠近敌方: 5, 敌方主动靠近: 5, 命中且前8s无视野: 5, 清完兵线: 5, 塔下清兵: 5, 撤退: 2, 前进: 2, 等待: 1, 攻击小兵: 5 \n第二段时间,跟随靠近敌方: 1, 技能: 1, 敌方主动靠近: 1, 命中且前8s无视野: 5, 无视野靠近: 3, 无视野单人靠近: 3, 清完兵线: 4, 塔下清兵: 4, 撤退: 4, 前进: 1, 攻击小兵: 3 \n第三段时间,无进入草丛: 1, 命中有视野: 4, 命中且前8s无视野: 2, 命中且前2s有视野: 4, 命中且前5s有视野: 4, 到前2s有视野: 2, 到前3s有视野: 2, 到前4s有视野: 2, 无视野靠近: 5, 有视野靠近: 3, 无视野单人靠近: 5, 有视野单人靠近: 3, 撤退: 1, 前进: 4 \n第四段时间,攻击中立野怪: 2, 首个入草丛: 3, 技能探草: 2, 技能: 2, 命中有视野: 2, 多人击杀: 3, 命中有视野: 5, 命中且前2s有视野: 3, 到前2s有视野: 5, 到前3s有视野: 5, 到前4s有视野: 5, 有视野靠近: 5, 有视野单人靠近: 5, 多人击杀: 3, 有大招，大招起手: 3, 撤退: 5, 攻击英雄: 3 \n", 
            "英雄决策是: 百里守约 \n第一段时间,敌方攻击，塔下防守: 5, 塔内防守: 5, 技能: 1, 命中有视野: 1, 命中有视野: 5, 命中且前2s有视野: 5, 命中且前5s有视野: 5, 命中且前8s有视野: 5, 到前2s有视野: 5, 到前3s有视野: 5, 到前4s有视野: 5, 有视野靠近: 5, 有视野单人靠近: 5, 没有清完兵线: 3, 塔下清兵: 3, 撤退: 2, 前进: 3, 攻击小兵: 4, 攻击英雄: 2 \n第二段时间,敌方攻击，塔下防守: 5, 塔内防守: 5, 技能: 1, 命中有视野: 1, 命中有视野: 3, 命中且前2s有视野: 5, 命中且前5s有视野: 5, 命中且前8s有视野: 2, 到前2s有视野: 5, 到前3s有视野: 5, 到前4s有视野: 5, 有视野靠近: 4, 有视野单人靠近: 4, 撤退: 1, 前进: 1, 等待: 3, 攻击英雄: 3 \n第三段时间,单人靠近敌方: 5, 敌方攻击，塔下防守: 5, 塔内防守: 5, 技能: 2, 命中有视野: 2, 单人击杀: 1, 到前3s无视野: 1, 命中有视野: 4, 命中且前2s有视野: 2, 到前2s有视野: 5, 到前3s有视野: 2, 到前4s有视野: 2, 有视野靠近: 4, 有视野单人靠近: 4, 单人击杀: 1, 残血击杀: 1, 前进: 1, 等待: 4, 攻击英雄: 3 \n第四段时间,单人靠近敌方: 3, 敌方攻击，塔下防守: 5, 塔内防守: 3, 单人击杀: 3, 清完兵线: 2, 单人击杀: 5, 残血击杀: 5, 塔下清兵: 2, 撤退: 1, 前进: 2, 等待: 2, 攻击小兵: 2, 攻击英雄: 2 \n", 
             ]
    prompter.generate_prompt_list(instruction, input)