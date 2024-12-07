from fastapi import FastAPI
from dotenv import load_dotenv
import os
from typing import List
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.perplexity import ChatPerplexity
from pydantic import BaseModel

from prompts import extract_prompt_generator, identify_common_headers, compare_and_generate_json_prompt, header_parser, specification_parser

app = FastAPI()
load_dotenv()

# Model for JSON
JSON_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    request_timeout=None,
    max_retries=2,
    api_key=os.getenv("OPEN_AI_KEY"),
)

# Perplexity model use to get information
specification_llm_pplx = ChatPerplexity(
    model="llama-3.1-sonar-small-128k-online",
    temperature=0,
    pplx_api_key=os.getenv("PPLX_AI_KEY")
)

class Link(BaseModel):
    link: str
    
class LinkList(BaseModel):
    links: List[str]

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/compare")
async def test_langchain(link_list: LinkList):
    
    res = []
    
    for link in link_list.links:
        custom_prompt_to_extract = extract_prompt_generator(link)
        res.append(specification_llm_pplx.invoke(custom_prompt_to_extract).content)
    
    headers_prompt = identify_common_headers(res)
    headers_result = JSON_llm.invoke(headers_prompt)
    parsed_headers = header_parser.parse(headers_result.content)
    
    compared_specification_json_prompt = compare_and_generate_json_prompt(res, parsed_headers)
    json_result = JSON_llm.invoke(compared_specification_json_prompt)
    parsed_json = specification_parser.parse(json_result.content)
    
    return parsed_json