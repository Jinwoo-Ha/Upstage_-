import os
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import openai

load_dotenv()

api_key = os.getenv("UPSTAGE_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_story(name, age, country, interests):
    if not api_key:
        raise ValueError("UPSTAGE_API_KEY 환경 변수가 설정되지 않았습니다.")

    prompt_template = PromptTemplate.from_template(
        """
        너는 친근한 반말로 한국에 사는 {age}살 {name}에게 동화를 들려주는 역할이야. 
        {name}(이)가 좋아하는 {interests}를 포함해서 {country}에 대한 동화를 만들어줘.
        먼저 비행기를 타고 한국에서 출발해 {country}에 도착한 후, 우리는 어디를 갈까? 
        그 곳에서 여행 과정과 얻은 교훈을 자세하게 이야기해줘. 
        주인공의 나이에 맞는 수준으로 동화를 만들어줘.

        첫 문장은 "안녕 나는 호솔이야! 우리 {name}와 함께 {country}로 여행을 떠나볼까? \n\n" 로 시작해.
        비행기를 타고 {country}에 도착하면, 그 {country}의 대표 공항도 언급해줘.

        반드시 모든 문장을 반말로 해줘야 해.
        {country}에서 방문하는 도시는 최소 다섯 곳 이상에, 각 도시별로 네 가지 이상의 활동을 해야해.
        {country}의 각 도시에 대한 정보를 상세히 알려주고, 거기서 어떤 음식을 먹는지, 
        그 도시의 구체적인 역사, 유명한 문화, 유명한 스포츠, 필수 관광지에 대해 언급해줘.
        각 도시에 대한 내용은 최소 200글자가 넘도록 상세히 설명해줘.

        단, "카지노, 흡연, 음주, 도박" 등 성인들만 즐길 수 있는 활동은 표시하지 말아줘.
        동화를 끝낼때는 소감, 교훈에 대해 이야기를 나누어줘.
        """
    )

    llm = ChatUpstage(
        model="solar-1-mini-chat",
        temperature=0.95,
        top_p=0.85,
        max_tokens=2048,
        api_key=api_key
    )

    chain = prompt_template | llm | StrOutputParser()

    result = chain.invoke({"name": name, "age": age, "country": country, "interests": interests})
    return result

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_image(prompt):
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        return response.data[0].url
    except Exception as e:
        print(f"이미지 생성 중 오류 발생: {e}")
        return None

def generate_story_with_images(name, age, country, interests):
    story_content = generate_story(name, age, country, interests)
    pages = story_content.split('\n\n')
    images = []

    for page in pages:
        image_prompt = f"{age}살 {name}이(가) {country}에서: {page[:100]}..."
        image_url = generate_image(image_prompt)
        images.append(image_url)

    return story_content, images