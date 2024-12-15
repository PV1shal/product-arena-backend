from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from dotenv import load_dotenv
import os
from typing import List
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.perplexity import ChatPerplexity
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from prompts import extract_prompt_generator, identify_common_headers, compare_and_generate_json_prompt, ProductSpecifications, header_parser, specification_parser

app = FastAPI()
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model for JSON
JSON_llm = ChatOpenAI(
    model="gpt-4o-mini",
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

class NewProductRequest(BaseModel):
    prev_products: List[ProductSpecifications]
    new_product: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/compare")
async def newCompare(link_list: LinkList):
    res = []
    product_jsons = []

    # Function to extract information for each link    
    def getItemInformationThread(link):
        custom_prompt_to_extract = extract_prompt_generator(link)
        res.append(specification_llm_pplx.invoke(custom_prompt_to_extract).content)

    # Function to generate JSON for each productInfo        
    def generateJson(productInfo):
        compared_specification_json_prompt = compare_and_generate_json_prompt(productInfo, parsed_headers)
        json_result = JSON_llm.invoke(compared_specification_json_prompt)
        parsed_json = specification_parser.parse(json_result.content)
        product_jsons.append(parsed_json)
    
    # Extract information in parallel
    with ThreadPoolExecutor() as executor:
        executor.map(getItemInformationThread, link_list.links)
    
    headers_prompt = identify_common_headers(res)
    headers_result = JSON_llm.invoke(headers_prompt)
    parsed_headers = header_parser.parse(headers_result.content)
    
    # Generate JSONs in parallel
    with ThreadPoolExecutor() as executor:
        executor.map(generateJson, res)
    
    print(product_jsons)
    
    return product_jsons

@app.post("/compareWithNew")
async def addedCompare(request: NewProductRequest):
    def extract_headers(products):
        headers = []
        seen_headers = set()
        for product in products:
            for spec in product.specifications:
                if spec.name not in seen_headers:
                    headers.append(spec.name)
                    seen_headers.add(spec.name)
        return headers

    def get_product_info(product_link):
        custom_prompt_to_extract = extract_prompt_generator(product_link)
        return specification_llm_pplx.invoke(custom_prompt_to_extract).content

    # Run both operations in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        # Submit both tasks
        future_headers = executor.submit(extract_headers, request.prev_products)
        future_product_info = executor.submit(get_product_info, request.new_product)
        
        # Get results from both threads
        headers = future_headers.result()
        new_product_info = future_product_info.result()

    # Create headers dictionary for JSON generation
    headers_dict = {
        "common_headers": headers,
        "additional_headers": []
    }
    
    # Generate JSON for new product using existing headers
    compared_specification_json_prompt = compare_and_generate_json_prompt(
        new_product_info, 
        headers_dict
    )
    json_result = JSON_llm.invoke(compared_specification_json_prompt)
    new_product_json = specification_parser.parse(json_result.content)
    
    return new_product_json

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Request body: {await request.body()}")
    print(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": await request.body()}
    )