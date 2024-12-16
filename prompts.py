
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
                - Given a link to an item, extract an exhaustive spec sheet of the product all the various sources.
                - Make sure to get specs from ATLEAST 5 sources.
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
        3. The JSON structure should be as follows:
           - product_name: string
           - brand: string
           - category: string
           - specifications: array of objects (attributes matching common headers)
           - additional_specifications: array of objects (attributes not matching common headers)
        4. There should be abolutely no line breaks.

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
        Compare the following product information and identify common specification headers and additional specifications.

        Product Information:
        {product_infos}

        Instructions:
        1. Identify specification headers that are present across all products. These will be considered 'common headers'.
        2. Any specification not present in all products should be categorized as 'additional'.
        3. Return the result in JSON format with two keys:
           - "common_headers": List of headers shared by all products.
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
    value: Union[str, List[str]] = Field(description="Value of the specification")
    # unit: Optional[str] = Field(description="Unit of measurement, if applicable", default=None)

# JSON body class for 
class ProductSpecifications(BaseModel):
    product_name: str = Field(description="Name of the product")
    brand: str = Field(description="Brand of the product")
    category: str = Field(description="Category of the product (e.g., tech, audio)")
    specifications: List[Specification] = Field(description="List of product specifications")
    additional_specifications: Optional[List[Specification]] = Field(description="Additional information that doesn't fit into standard specifications", default=[])

# JSON class for common headers
class CommonHeaders(BaseModel):
    common_headers: List[str] = Field(description="List of common specification headers")
    additional_headers: List[str] = Field(description="List of specifications that don't fall into common headers")

specification_parser = JsonOutputParser(pydantic_object=ProductSpecifications)
header_parser = JsonOutputParser(pydantic_object=CommonHeaders)