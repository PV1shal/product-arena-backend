
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from typing import List, Optional, Union
from pydantic import BaseModel, Field

# Generates specifications
def extract_prompt_generator(link):
    extract_prompt = [
        {
            "role": "system",
            "content": """
                - Given a link to a product, extract an exhaustive spec sheet and reviews of the product from various sources.
                - Make sure to get specs from AT LEAST 5 sources.
                - For reviews, collect at least 20 reviews in total:
                  - At least 10 positive (good) reviews
                  - At least 10 negative (bad) reviews
                - Reviews should come from different platforms (e.g., Amazon, Reddit, product website).
                - For each review, include:
                  - The platform name
                  - The review text
                  - A classification as either "good" or "bad"
                - Ensure that the reviews are specific and relevant to the product's features and performance.
            """
        },
        {
            "role": "human",
            "content": link
        }
    ]
    return extract_prompt

# Prompt to generate JSON with the generalised specifications
def compare_and_generate_json_prompt(product_info, parsed_headers):
    prompt_template = PromptTemplate(
        template="""
        Using the provided common headers {parsed_headers}, generate a JSON object for the product.

        Product Information:
        {product_info}

        Instructions:
        1. Map attributes matching the common headers to the "specifications" field.
        2. Place any remaining attributes under "additional_specifications".
        3. Include reviews in the "reviews" field (at least 20 total, with 10 good and 10 bad).
        4. The JSON structure should be as follows:
           - product_name: string
           - brand: string
           - category: string
           - specifications: array of objects (attributes matching common headers)
           - additional_specifications: array of objects (attributes not matching common headers)
           - reviews: array of objects (each containing platform, review text, and classification as 'good' or 'bad')
        5. There should be absolutely no line breaks.

        {format_instructions}
        """,
        input_variables=["product_info", "parsed_headers"],
        partial_variables={"format_instructions": specification_parser.get_format_instructions()},
    )
    
    return prompt_template.format(product_info=str(product_info), parsed_headers=str(parsed_headers))

# Generate common headers
def identify_common_headers(product_infos):
    prompt_template = PromptTemplate(
        template="""
        Provided informations of two products, you are tasked to return headers or keys for each specification.

        Product Information:
        {product_infos}

        Instructions:
        1. Identify specification that are common across all products and return common headers for each. These will be considered 'common headers'. The header should be as SPECIFIC as possible.
        2. Any specification that is common for both products SHOULD be categorized in 'specifications'.
        3. Any specification that is not common in all products SHOULD be categorized as 'additional_headers'.
        4. Return the result in JSON format with two keys:
           - "specifications": List of headers shared by all products.
           - "additional_headers": List of headers unique to specific products.

        {format_instructions}
        """,
        input_variables=["product_infos"],
        partial_variables={"format_instructions": header_parser.get_format_instructions()},
    )
    
    return prompt_template.format(product_infos=str(product_infos))

# JSON body class for sepcification
class Specification(BaseModel):
    name: str = Field(description="Name of the specification")
    value: str = Field(description="Value of the specification")
    # unit: Optional[str] = Field(description="Unit of measurement, if applicable", default=None)

class Review(BaseModel):
    platform: str = Field(description="Name of the platform where the review was posted")
    review: str = Field(description="Text of the review")
    classification: str = Field(description="Classification of the review as either 'good' or 'bad'")

# JSON body class for 
class ProductSpecifications(BaseModel):
    product_name: str = Field(description="Name of the product")
    brand: str = Field(description="Brand of the product")
    category: str = Field(description="Category of the product (e.g., tech, audio)")
    # image_link: str = Field(description="Image link of the product")
    specifications: List[Specification] = Field(description="List of product specifications")
    additional_specifications: Optional[List[Specification]] = Field(description="Additional information that doesn't fit into standard specifications", default=[])
    reviews: List[Review] = Field(description="List of product reviews (at least 20, with 10 good and 10 bad)")

# JSON class for common headers
class CommonHeaders(BaseModel):
    specifications: List[str] = Field(description="List of common specification headers")
    additional_headers: List[str] = Field(description="List of specifications that don't fall into common headers")

specification_parser = JsonOutputParser(pydantic_object=ProductSpecifications)
header_parser = JsonOutputParser(pydantic_object=CommonHeaders)