import streamlit as st

from os import environ
import dotenv
import base64
import os
import requests
import replicate

env_file = '.env'

dotenv.load_dotenv(env_file, override=True)
OPENAI_API_KEY = environ.get('OPENAI_API_KEY')
DREAMSTUDIO_API_KEY = environ.get('DREAMSTUDIO_API_KEY')
REPLICATE_API_TOKEN = environ.get('REPLICATE_API_TOKEN')

from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

llm_low_creativity = OpenAI(model_name='gpt-3.5-turbo-instruct', temperature='0.1')
llm_medium_creativity = OpenAI(model_name='gpt-3.5-turbo-instruct', temperature='0.5')

product_specialist_role = 'You are a product and marketing psychology specialist and can suggest useful products.'
poster_artist_role = 'You are an ad designer specialised in creating advertisement posters.'
script_writer_role = 'You are a script writer, creative, and marketing expert.'

template = """
    {role}
    {task}
"""

product_prompt = PromptTemplate(
    input_variables = ['role', 'profile'],
    template = """
        {role}
        Choose a suitable product to advertise to the customer with this profile: {profile}, give your answer with the chosen product name contained in a python-dictionary under the key: "product".
    """
)

poster_desc_prompt = PromptTemplate(
    input_variables = ['role', 'product', 'profile'],
    template = """
        {role}
        Design an advertisement for {product}, with a realistic human with the profile {profile} as featured main character, give your answer as a description of the advertisement that is used as a prompt for an ai image generator to create the described poster, make sure that the poster centers the main character front facing, and displays the product clearly with the main character interacting with the product.
    """
)

def get_product(profile):
    product_chain = LLMChain(llm=llm_low_creativity, prompt=product_prompt)
    product = product_chain.run({'role': product_specialist_role, 'profile': profile})
    product_extract = product.split(sep='{')[1]
    product_extract = product_extract.split(sep=':')[1]
    product_extract = product_extract.split(sep='"')
    product_extract = product_extract[1]
    return product_extract

def get_image_prompt(product, profile):
    poster_chain = LLMChain(llm=llm_medium_creativity, prompt=poster_desc_prompt)
    poster_description = poster_chain.run({'role': poster_artist_role, 'product': product_extract, 'profile': profile})
    image_prompt = poster_description.strip()
    image_prompt = image_prompt.replace('\n', '')
    image_prompt += " this is an advertisement poster and the style is photorealistic."
    return image_prompt  

def get_ad(image_prompt):
    ad_poster_path = None

    body = {
        "samples": 1,
        "height": 1024,
        "width": 1024,
        "steps": 40,
        "cfg_scale": 6,
        "text_prompts": [
            {
            "text": image_prompt,
            "weight": 1
            },
            {
            "text": "lowres, blurry, bad, bad anatomy, bad hands, side facing, side profile, cropped, worst quality, unrealistic, cartoon, anime, disfigured, without face",
            "weight": -1
            }
        ],
    }

    response = requests.post(
        "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer {}".format(DREAMSTUDIO_API_KEY),
        },
        json=body,
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()

    # make sure the out directory exists
    if not os.path.exists("./out"):
        os.makedirs("./out")

    for i, image in enumerate(data["artifacts"]):
        ad_poster_path = f'./out/txt2img_{image["seed"]}.jpg' 
        with open(ad_poster_path, "wb") as f:
            f.write(base64.b64decode(image["base64"]))
    
    return ad_poster_path

def swap_faces(input_path, target_path):
    swap_image = input_path
    target_image = open(target_path, 'br')
    output = replicate.run(
        "lucataco/faceswap:9a4298548422074c3f57258c5d544497314ae4112df80d116f0d2109e843d20d",
        input={
            "swap_image": swap_image,
            "target_image": target_image
        }
    )
    return output

def main():
    st.markdown("# ADYOU")
    
    st.markdown("Input information about a user (user profile).")
    input_profile = st.text_area("Enter your text:", height=300)
    
    st.markdown("User image to generate ad for.")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # To read file as bytes:    
        bytes_data = uploaded_file.getvalue()
    
    submit_button = st.button("Submit")
    if submit_button:
        product = get_product(input_profile)
        image_prompt = get_image_prompt(product, input_profile)
        ad_image_path = get_ad(image_prompt)
        if uploaded_file is not None:
            # TODO: This needs a path to upload file not upload file
            link = swap_faces(uploaded_file, ad_image_path)
            st.image(link, caption="Generated Ad", width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        else:
            print("reached end of script")
            st.image(target_image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

if __name__ == "__main__":
    main()