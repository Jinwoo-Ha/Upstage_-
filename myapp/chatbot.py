# chatbot.py

import os
import re
import json
from typing import List, Dict, Any, Optional
from pprint import pprint

from django.conf import settings

from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.tools import DuckDuckGoSearchResults

from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key=settings.UPSTAGE_API_KEY, base_url="https://api.upstage.ai/v1/solar")

# Ground check function
def ground_check(scenario, answer):
    request_input = {
        "messages": [
            {"role": "user", "content": scenario},
            {"role": "assistant", "content": answer}
        ]
    }

    responses = []
    ground = 0

    for i in range(5):
        response = client.chat.completions.create(
            model="solar-1-mini-groundedness-check",
            messages=request_input["messages"]
        )
        gc_result = response.choices[0].message.content
        responses.append(gc_result)

        if gc_result.lower().startswith("grounded"):
            ground += 1
    
    perc_ground = (ground / 5) * 100
    return perc_ground

# Initialize Solar Mini chat model
solar_mini = ChatUpstage(model="solar-1-mini-chat")

# Fact extraction function
def extracted_claimed_facts(text: str, llm: Optional[ChatUpstage] = solar_mini) -> List[Dict[str, Any]]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert fact extractor. Extract a list of claimed facts, focusing on entities and their relationships."),
        ("human", "Extract the claimed facts from the following text:\n\n{input_text}"),
        ("human", "Respond with a JSON array of fact dictionaries only.")
    ])

    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"input_text": text})
    return result

# Search context function
def search_context(text: str, claimed_facts: List[Dict[str, Any]], search_tool: DuckDuckGoSearchResults = DuckDuckGoSearchResults(), llm: Optional[ChatUpstage] = solar_mini) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate 3-5 search keywords based on the given text and facts."),
        ("human", "Text: {text}\n\nExtracted Facts:\n{facts}\n\nProvide only the keywords, separated by commas.")
    ])
    #2. 필수 키가 있는 항목만 처리
    #facts_str = "\n".join([f"- {fact['entity']} {fact['relation']} {fact['value']}" for fact in claimed_facts])
    facts_str = "\n".join([f"- {fact['entity']} {fact['relation']} {fact['value']}" for fact in claimed_facts if 'entity' in fact and 'relation' in fact and 'value' in fact])
    keywords_response = llm.invoke(prompt.format(text=text, facts=facts_str))

    keywords = [kw.strip() for kw in keywords_response.content.split(",") if kw.strip()]
    search_query = " ".join(keywords)
    search_results = search_tool.run(search_query)

    return search_results

# Build knowledge graph function
def build_kg(claimed_facts: List[Dict[str, Any]], context: str, llm: Optional[ChatUpstage] = solar_mini) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Build a knowledge graph from the given context, using claimed facts as schema hints."),
        ("human", "Context:\n{context}\n\nClaimed Facts (use as schema hints):\n{claimed_facts}\n\nConstruct the knowledge graph:")
    ])

    chain = prompt | llm | JsonOutputParser()
    facts_str = "\n".join([f"- {fact['entity']} {fact['relation']} {fact['value']}" for fact in claimed_facts])
    kg = chain.invoke({"context": context, "claimed_facts": facts_str})

    return kg

# Verify facts function
def verify_facts(claimed_facts: List[Dict[str, Any]], context: str, kg: Dict[str, Any], confidence_threshold: float, llm: Optional[ChatUpstage] = solar_mini) -> Dict[str, Dict[str, Any]]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Verify the claimed fact against the knowledge graph and context."),
        ("human", "Claimed Fact: {entity} {relation} {value}\n\nKnowledge Graph:\n{kg}\n\nContext:\n{context}\n\nProvide the verification result:")
    ])

    chain = prompt | llm | JsonOutputParser()
    kg_str = json.dumps(kg, indent=2)
    verified_facts = {}

    for i, fact in enumerate(claimed_facts):
        verification_result = chain.invoke({
            "entity": fact["entity"],
            "relation": fact["relation"],
            "value": fact["value"],
            "kg": kg_str,
            "context": context,
        })

        verified_facts[str(i)] = {
            "claimed": f"{fact['entity']} {fact['relation']} {fact['value']}",
            **verification_result,
        }

    return verified_facts

# Add fact check comments to text
def add_fact_check_comments_to_text(text, verified_facts, llm=solar_mini):
    fact_map = {fact["claimed"]: fact for fact in verified_facts.values()}
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Add fact-check annotations to the given text."),
        ("human", "Original text:\n{text}\n\nVerified facts:\n{fact_map}\n\nAdd fact-check annotations to the original text:")
    ])

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"text": text, "fact_map": fact_map})
    return response

# Fact check function
def fact_check(text_to_check):
    claimed_facts = extracted_claimed_facts(text_to_check)
    relevant_context = search_context(text_to_check, claimed_facts)
    kg = build_kg(claimed_facts, relevant_context)
    verified_facts = verify_facts(claimed_facts, relevant_context, kg, confidence_threshold=0.7)
    fact_checked_text = add_fact_check_comments_to_text(text_to_check, verified_facts)
    return fact_checked_text

# Check fact function
def check_fact(text):
    fact_patterns = re.findall(r'\[Fact: (True|False).+?\]', text)
    success = sum(1 for fact in fact_patterns if "True" in fact)
    return success / len(fact_patterns) * 100 if fact_patterns else 0

# Main chatbot function
def ask_chatbot(user_input, scenario):
    scenario_document = Document(page_content=scenario)
    retriever = BM25Retriever.from_documents([scenario_document])
    llm = ChatUpstage()

    prompt_template = PromptTemplate.from_template(
        """
        너는 다양한 국가에 대한 동화를 읽고 있는 어린이의 이해를 돕기 위한 어시스턴트로, 너의 이름은 '호솔'이야.
        동화를 읽고 있는 어린이의 이해를 돕기 위해 친절하고 다정하게 답변하고, 모든 문장은 반드시 반말로 해.
        질문에 대한 정확한 답변을 내야 하고, 어린이들이 이해하기 쉽게 간단한 단어와 문장 구조를 사용해.
        어린이가 어려워하거나 힘들어하면 격려해줄 수 있어야 해.
        답변은 반드시 반말로 이루어져야 해.
        Your answer should not exceed 30 words.Keep this in mind!

        Please provide most correct answer from the following context. 
        If the answer is not present in the context, create a relevant response based on your understanding.
        ---
        Question: {question}
        ---
        Context: {Context}
        ---
        """
    )
    chain = prompt_template | llm | StrOutputParser()

    context_docs = retriever.invoke(user_input)
    assistant_response = chain.invoke({"question": user_input, "Context": context_docs})

    perc_ground = ground_check(scenario, assistant_response)

    if perc_ground < 50:
        fact_checked_response = fact_check(assistant_response)
        fact_checked_output = check_fact(fact_checked_response)

        if fact_checked_output < 80:
            assistant_response = chain.invoke({"question": user_input, "Context": context_docs})
            fact_checked_response = fact_check(assistant_response)
            fact_checked_output = check_fact(fact_checked_response)

    return assistant_response

# Scenario (you can move this to a separate file or database later)
scenario = """
와! 해원아, 이제 우리는 프랑스로 떠나서 멋진 명소들을 구경할 거야! 비행기를 타고 프랑스로 가는 건 정말 신나는 일이겠지? 도착하면 우리는 프랑스의 대표 공항인 샤를 드 골 공항에 도착할 거야. 그곳은 파리에 위치해 있어서 여행 시작하기에 딱 좋은 장소야!
공항에 도착하면 먼저 세관을 통과해야 해. 여권과 필요한 서류를 잘 챙겨야 해. 그리고 수하물을 찾으러 가야지. 비행기 안에서 조금 불편했을 테니, 우리는 파리 시내로 이동해서 편안한 호텔에 체크인할 거야.
첫 번째 명소로 우리는 에펠탑을 보러 갈 거야! 에펠탑은 파리의 상징이자 세계에서 가장 유명한 건축물 중 하나야. 우리는 에펠탑을 올려다보며 그 웅장함에 감탄할 거야. 그리고 꼭대기에 올라가서 아름다운 파리 전경을 감상할 수도 있어.
다음으로 우리는 루브르 박물관에 갈 거야. 루브르 박물관은 세계에서 가장 크고 유명한 박물관 중 하나야. 거기에는 레오나르도 다 빈치의 '모나리자'를 비롯한 수많은 예술 작품들이 전시되어 있어. 우리는 예술의 아름다움과 역사를 배울 수 있을 거야.
파리에서는 맛있는 음식도 즐길 수 있어. 크로와상과 에스카페드를 맛보면서 프랑스의 맛있는 빵 문화를 경험할 수 있어. 또한, 프랑스 와인 한 잔으로 여행의 분위기를 더욱 즐길 수도 있지!
여행을 하면서 우리는 새로운 문화와 사람들을 만나게 될 거야. 프랑스어를 조금 배워두면 현지인들과 소통하기가 더 쉬울 거야. 또한, 다른 나라의 관습과 예절을 존중하는 것도 중요해.
이 여행에서 우리는 새로운 경험을 쌓고, 세계를 더 넓게 바라볼 수 있을 거야. 프랑스에서의 여정은 우리에게 새로운 영감과 추억을 선사할 거야. 함께 프랑스 여행을 떠나보자!
"""