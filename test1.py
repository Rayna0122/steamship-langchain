from langchain import LLMChain, PromptTemplate
from steamship import Steamship

# from langchain import OpenAI
from steamship_langchain import OpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from steamship_langchain.memory import ChatMessageHistory

import logging
client = Steamship(api_key="14611DD0-8AB2-47D3-981E-1CD9D40969C5")

logging.disable(logging.CRITICAL)  # disable warning messages to declutter

template = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

{history}
Human: {human_input}
Assistant:"""
client = Steamship()
prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

chat_memory = ChatMessageHistory(client=client, key="chat-test")

chatgpt_chain = LLMChain(
    llm=OpenAI(client=client, temperature=0),
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory(chat_memory=chat_memory, k=2),
)

chatgpt_chain.predict(
    human_input="现在假装你是一个中国建设银行的智能客服，我是一个顾客，你每次回答都要以“您好，很高兴为您服务”为开头进行回答，一定要有礼貌，耐心解答；如果遇到用“#”前后标记的内容#像这样#，这将不作为顾客的提问，而是请你理解并记忆的内容，而且只回复：“谢谢，我已将上述内容纳入知识库中。”，好现在开始。"
)
chatgpt_chain.predict(human_input="#通过建行电话银行可以办理哪些业务？答：我行电话银行可为客户提供自助语音和人工服务相结合的金融服务。具体向客户提供账户查询、转账汇款、缴费、个贷查询、公积金查询、金融信息查询、账户挂失、信用卡还款、投资理财以及信息咨询、投诉、建议等多种服务。我想使用建行电话银行，怎样才能通过电话银行办理业务？答：请您拨打我行24小时服务热线95533，接通后根据系统提示设置好电话银行查询密码，您即成为我行电话银行普通客户，可以办理账户查询等业务。您也可以到柜台签约成为我行电话银行高级客户。签约建行电话银行高级客户需要携带哪些资料？答：请您携带本人有效身份证件、建行实名制账户到我行任意网点柜台签约电话银行即可。使用建行电话银行是否收取服务费？答：您通过柜台或自助方式开通电话银行服务后，若使用电话银行进行我行有偿服务如转账汇款等，则需按一定标准收取结算手续费（目前电话银行实行按柜面五折标准收取手续费）。建行电话银行如何保障我的资金安全？答：我行电话银行采取了客户个人操作密码唯一性，交易封闭管理等安全措施，流程上还增加简单密码限制、短信验证码等控制，保障了整个操作过程及交易资金安全、可靠。我的电话银行密码忘了，应该怎么办？答：如您遗忘的是电话银行签约密码，需由您本人持有效身份证件和任一电话银行签约账户至我行任意网点柜台重置密码；如您遗忘的是电话银行非签约密码，您可拨打95533转人工进行密码重置。电话银行高级客户与普通客户在服务上有什么区别？答：普通可以通过电话银行办理账户查询、缴费充值等业务，高级客户除了享受普通客户的服务，还可以办理转账汇款等交易，同时在很多项目上享有比普通客户更高的交易限额。#")
chatgpt_chain.predict(human_input="我能否替别人进行信用卡开卡")
chatgpt_chain.predict(human_input="#我能否替别人进行信用卡开卡？答：只要您正确输入所有验证信息，就可以开通本人或他人名下的信用卡。#")