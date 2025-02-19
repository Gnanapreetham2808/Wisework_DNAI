from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY not set.")

# Configure GenAI API
genai.configure(api_key=api_key)

# Initialize FastAPI app
app = FastAPI()

# Define input models
class InputModel(BaseModel):
    input: str

class SymptomsModel(BaseModel):
    symptoms: list[str]

class OutputModel(BaseModel):
    output: str

class BookingModel(BaseModel):
    test_name: str
    user_name: str
    contact: str

class HealthQueryModel(BaseModel):
    query: str

# Predefined chatbot responses
predefined_responses = {
    "hi": "Hello! Here are the tests we currently offer:\nLongevity Tests\nGenetic Tests\nImmunity Store",
    "hello": "Hello! Here are the tests we currently offer:\nLongevity Tests\nGenetic Tests\nImmunity Store",
    "how are you": "I'm just a bot, but I'm here to help! How can I assist you?",
    "services": "We offer Longevity Tests, Genetic Tests, and an Immunity Store. How can I help?",
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
}

# Cancer symptom mapping
cancer_symptoms = {
    "lung cancer": ["cough", "chest pain", "shortness of breath", "coughing up blood"],
    "breast cancer": ["lump in breast", "nipple discharge", "breast pain", "skin changes"],
    "colon cancer": ["blood in stool", "abdominal pain", "weight loss", "diarrhea"],
    "leukemia": ["fatigue", "frequent infections", "easy bruising", "pale skin"],
    "skin cancer": ["mole changes", "new skin growth", "itchy lesion", "bleeding spot"]
}

@app.post("/generate-response", response_model=OutputModel)
async def generate_response(input_data: InputModel):
    user_input = input_data.input.strip().lower()
    if user_input in predefined_responses:
        return OutputModel(output=predefined_responses[user_input])
    
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    response = model.generate_content(user_input)
    response_text = response.text.strip().split(".")[0]
    return OutputModel(output=response_text)

@app.post("/detect-cancer", response_model=OutputModel)
async def detect_cancer(symptoms_data: SymptomsModel):
    user_symptoms = set(symptom.lower() for symptom in symptoms_data.symptoms)
    possible_cancers = [cancer for cancer, symptoms in cancer_symptoms.items() if any(symptom in user_symptoms for symptom in symptoms)]
    
    if possible_cancers:
        return OutputModel(output=f"Possible risk: {', '.join(possible_cancers)}. Consult a doctor.")
    
    return OutputModel(output="No strong cancer indications found. Stay healthy and consult a doctor if needed.")

@app.post("/book-test", response_model=OutputModel)
async def book_test(booking_data: BookingModel):
    available_tests = [
        "ai cancer test", "diabetic testing", "cardiovascular testing",
        "personalized food test", "healthy supplement test",
        "nipt", "clinical exome sequencing", "microbiome",
        "genetic disorder testing", "whole exome sequencing", "rna sequencing"
    ]
    if booking_data.test_name.lower() in available_tests:
        return OutputModel(output=f"Thank you {booking_data.user_name}! Your booking for {booking_data.test_name} is confirmed. We will contact you at {booking_data.contact} soon.")
    else:
        return OutputModel(output=f"Sorry, we couldn't find the test '{booking_data.test_name}'. Please check available tests.")

@app.get("/available-tests", response_model=dict)
async def available_tests():
    return {
        "longevity_tests": ["AI Cancer Test", "Diabetic Testing", "Cardiovascular Testing", "Personalized Food Test", "Healthy Supplement Test"],
        "genetic_tests": ["NIPT", "Clinical Exome Sequencing", "Microbiome", "Genetic Disorder Testing", "Whole Exome Sequencing", "RNA Sequencing"]
    }

@app.post("/health-tips", response_model=OutputModel)
async def health_tips(query_data: HealthQueryModel):
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    response = model.generate_content(f"Provide health tips for: {query_data.query}")
    response_text = response.text.strip().split(".")[0]
    return OutputModel(output=f"Health Tip: {response_text}")

@app.get("/immunity-store", response_model=dict)
async def immunity_store():
    return {"items": [{"name": "Apple", "price": "₹100 for 250 gm"}, {"name": "Banana", "price": "₹20 for 250 gm"}, {"name": "Orange", "price": "₹30 for 250 gm"}], "order_link": "https://longevity.dnaiworld.com/immunity-store"}

@app.get("/")
async def root():
    return {"message": "Welcome to DNAi Chatbot. Use available endpoints for assistance."}
