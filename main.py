from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY environment variable not set in .env file.")

# Configure the GenAI API
genai.configure(api_key=api_key)

# Initialize FastAPI app
app = FastAPI()

# Define input and output models
class InputModel(BaseModel):
    input: str

class OutputModel(BaseModel):
    output: str

# Predefined responses
predefined_responses = {
    "hi": "Hello! Here are the tests we currently offer:\nLongevity Tests\nGenetic Tests\nImmunity Store",
    "hello": "Hello! Here are the tests we currently offer:\nLongevity Tests\nGenetic Tests\nImmunity Store",
    "longevity tests": (
        "Okay! Here are the Longevity Tests we currently offer:\n\n"
        "AI Cancer Test: Available\n\nBook AI Cancer Test here: https://longevity.dnaiworld.com/HybridChat.\n\n"
        "Diabetic Testing: Coming Soon\n"
        "Cardiovascular Testing: Coming Soon\n"
        "Personalized Food Test: Coming Soon\n"
        "Healthy Supplement Test: Coming Soon\n\n"
        "Let me know which one you'd like to learn more about!"
    ),
    "genetic tests": (
        "Okay! Here are the Genetic Tests we currently offer:\n\n"
        "NIPT: Noninvasive prenatal screening (NIPS)\n"
        "Clinical Exome Sequencing\n"
        "Microbiome\n"
        "Genetic Disorder Testing\n"
        "Whole Exome Sequencing\n"
        "RNA Sequencing\n\n"
        "Book Genetic Tests here: https://longevity.dnaiworld.com/GetGeneSeq.\n\n"
        "Let me know which one you'd like to learn more about!"
    ),
    "immunity store": (
        "Here’s what we currently have in the Immunity Store:\n\n"
        "Apple: ₹100 for 250 gm\n"
        "Banana: ₹20 for 250 gm\n"
        "Orange: ₹30 for 250 gm\n\n"
        "Let me know if you’d like to add any of these to your selection!\n\n"
        "You can visit the Immunity Store here: https://longevity.dnaiworld.com/immunity-store."
    ),
    "ai cancer test": (
        "The AI Cancer Test provides an AI-generated health report based on your lifestyle, offering insights and guidance to help you live a longer, healthier life. "
        "Would you like to proceed with this test?\n"
        "You can complete your booking here: https://longevity.dnaiworld.com/HybridChat."
    ),
    "microbiome": (
        "The Microbiome test analyzes the composition and diversity of microorganisms in specific environments. Would you like to explore this test further?\n"
        "You can book it here: https://longevity.dnaiworld.com/GetGeneSeq."
    ),
    "genetic disorder testing": (
        "The Genetic Disorder Testing checks for conditions caused by abnormalities in genetic material. Would you like to explore this test further?\n"
        "You can book it here: https://longevity.dnaiworld.com/GetGeneSeq."
    ),
    "whole exome sequencing": (
        "The Whole Exome Sequencing test focuses on exomes, the protein-coding regions of your genome. It captures known disease-causing mutations and is highly comprehensive.\n"
        "Would you like to explore this test further?\n\n"
        "You can book it here: https://longevity.dnaiworld.com/GetGeneSeq."
    ),
    "rna sequencing": (
        "The RNA Sequencing test analyzes the transcriptome to measure RNA levels and gene expression. Would you like to explore this test further?\n"
        "You can book it here: https://longevity.dnaiworld.com/GetGeneSeq."
    ),
    "cancer": (
        "Okay! Here are the Longevity Tests we currently offer:\n\n"
        "AI Cancer Test: Available\n\nBook AI Cancer Test here: https://longevity.dnaiworld.com/HybridChat.\n\n"
        "Diabetic Testing: Coming Soon\n"
        "Cardiovascular Testing: Coming Soon\n"
        "Personalized Food Test: Coming Soon\n"
        "Healthy Supplement Test: Coming Soon\n\n"
        "Let me know which one you'd like to learn more about!"
    ),
}

@app.post("/generate-response", response_model=OutputModel)
async def generate_response(input_data: InputModel):
    try:
        user_input = input_data.input.strip().lower()

        # Check for predefined response
        if user_input in predefined_responses:
            response_text = predefined_responses[user_input]
        else:
            # Generate AI response using GenAI
            response = genai.generate_text(
                prompt=user_input,
                model="gemini-2.0-flash-exp",
                temperature=0.5,
                max_output_tokens=200
            )
            response_text = response.text.strip()

        # Ensure response is not empty
        if not response_text:
            raise ValueError("Generated response is empty.")

        return OutputModel(output=response_text)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Validation Error: {str(e)}")
    except Exception as e:
        # Handle unexpected server errors
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the DNAi World Chatbot Framework! Use /generate-response to interact."}
