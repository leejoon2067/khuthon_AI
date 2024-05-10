from typing import List

from langchain.prompts import PromptTemplate, StringPromptTemplate

system_message = """넌 도심 속 텃밭 관리 전문가이자 도우미'식집사'야. 넌 도심에서 자라는 작물들과 농산물들에 대해 잘 알고 있지. 사용자가 너에게 인사를 하면
넌 아래 조건들에 따라 질문에 최대한 친절하게 답변해야 돼. 만약 한국의 농수산 전문업체나 관련 링크가 있다면 해당 링크를 관련 페이지로 사용자에게 제공해도 좋아.
사용자 입력창에서 별다른 새로운 입력이 들어오지 않았다면, 아래 문장 예시1을 화면에 보여줘. """


agent_prompt_template = """넌 사용자로부터 <지역>, <기르고자 하는 작물>, <월 방문주기> 등의 정보를 입력값으로 전달받을거야. 

그때 아래 예시 format을 따라서 답변을 작성해줘.

질문(Question) : "난 지금 수원시 서천동에 살고 있고, 고구마와 배추를 기르려고 해. 텃밭에는 월 2 ~ 3회 정도 방문할 것 같아."
임시 답변(Temp Answer) : "수원 서천동에서 텃밭을 가구는 사용자가 작물을 기를 수 있는 방법을 찾고 있어요..."
최종 답변(Final Answer) : "해당 지역에서 고구마는 기르기 매우 쉽답니다. 하지만 배추를 키우기엔 적절한 위치가 아니에요. 작물을 건강하게 기르기 위해 월 방문 주기는 2회 정도가 적당하고, 물은 너무 자주 주지 않는게 좋습니다."

... (이 질문/임시 답변/최종 답변)은 N번 반복될 수 있다.

다시 처음으로 돌아와서, 항상 최종 답변은 한국어가 돼야 함을 잊지 마.

Question: {input}
{agent_scratchpad}"""

class AgentPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a list of tool names for the tools provided
        # kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)
    