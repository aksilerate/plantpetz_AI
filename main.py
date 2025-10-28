import streamlit as st
import base64
from openai import OpenAI
from PIL import Image
import io


def image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image_with_openai(image_path, prompt_text, api_key):
    """Simple function to analyze image with OpenAI vision"""
    base64_image = image_to_base64(image_path)

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        temperature=0.1,
        top_p=0,
        frequency_penalty=0,
        presence_penalty=0,
        max_tokens=100,
    )

    return response.choices[0].message.content


def main():
    st.title("Plant Detection with OpenAI")

    api_key = "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"

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and api_key:
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getvalue())

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        prompt = "Can you please detect the plant in the provided image? Also how much percent you are sure about it? Dont give me anything else just the information I asked you about"

        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                try:
                    result = analyze_image_with_openai(
                        "temp_image.jpg", prompt, api_key
                    )

                    st.subheader("Analysis Result:")
                    st.write(result)

                except Exception as e:
                    st.error(f"Error analyzing image: {str(e)}")


if __name__ == "__main__":
    main()
