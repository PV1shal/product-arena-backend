
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
def compare_and_generate_json_prompt(product_infos, parsed_headers):
    prompt_template = PromptTemplate(
        template="""
        Generate JSON with keys using the headers {parsed_headers}. 
        
        Product Information:
        {product_infos}
        
        Instructions:
        1. Identify specification that fall into each categories.
        2. For each product, create a JSON object with the following structure:
           - product_name: string
           - brand: string
           - category: string (e.g., "audio equipment")
           - specifications: array of objects, each containing:
             - name: string (use common names across products)
             - value: string or number
             - unit: string (if applicable)
           - additional_specifications: array of objects, (for any information that doesn't fit the standard format), each containing:
             - name: string (use common names across products)
             - value: string or number
             - unit: string (if applicable)
        3. Try to get additional_specification values from other sources and move it to specifications.
        4. If value is missing or null get it from other sources.
        4. Ensure all products have the same specification categories, using null for missing values.
        5. Convert units to be consistent across products where possible.
        
        {format_instructions}
        """,
        input_variables=["product_infos", "parsed_headers"],
        partial_variables={"format_instructions": specification_parser.get_format_instructions()},
    )
    
    return prompt_template.format(product_infos=str(product_infos), parsed_headers=str(parsed_headers))

# Generate common headers
def identify_common_headers(product_infos):
    prompt_template = PromptTemplate(
        template="""
        Compare the following product information and identify common specification headers and additional specifications.

        Product Information:
        {product_infos}

        Instructions:
        1. Identify common specification headers across all products.
        2. List specifications that don't fall into the common headers under 'additional'.
        3. Return only the headers/keys, not the values.

        {format_instructions}
        """,
        input_variables=["product_infos"],
        partial_variables={"format_instructions": specification_parser.get_format_instructions()},
    )
    
    return prompt_template.format(product_infos=str(product_infos))

# JSON body class for sepcification
class Specification(BaseModel):
    name: str = Field(description="Name of the specification")
    value: Union[str, int, float] = Field(description="Value of the specification")
    unit: Optional[str] = Field(description="Unit of measurement, if applicable", default=None)

# JSON body class for 
class ProductSpecifications(BaseModel):
    product_name: str = Field(description="Name of the product")
    brand: str = Field(description="Brand of the product")
    category: str = Field(description="Category of the product (e.g., tech, audio)")
    specifications: List[Specification] = Field(description="List of product specifications")
    additional_specifications: Optional[Specification] = Field(description="Additional information that doesn't fit into standard specifications", default=None)

# JSON class for common headers
class CommonHeaders(BaseModel):
    common_headers: List[str] = Field(description="List of common specification headers")
    additional_headers: List[str] = Field(description="List of specifications that don't fall into common headers")

specification_parser = JsonOutputParser(pydantic_object=ProductSpecifications)
header_parser = JsonOutputParser(pydantic_object=CommonHeaders)