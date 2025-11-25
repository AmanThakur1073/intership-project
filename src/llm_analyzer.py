from groq import Groq
import os

def build_prompt(pred_dict):
    prompt = f"""
You are an expert agricultural biomass analyst.

Here are the biomass values predicted from the crop image:

Dry Clover: {pred_dict['Dry_Clover_g']} g
Dry Dead: {pred_dict['Dry_Dead_g']} g
Dry Green: {pred_dict['Dry_Green_g']} g
Dry Total: {pred_dict['Dry_Total_g']} g
GDM: {pred_dict['GDM_g']} g

Give detailed expert analysis:

1. Overall crop health  
2. Interpretation of each biomass value  
3. Stress level (low/medium/high) and why  
4. Moisture/Irrigation suggestion  
5. Fertilizer & nutrient advice  
6. Disease or stress risk  
7. Expected biomass trend  
8. Final conclusion (2 lines)

Keep the answer simple & clear in  english.
"""
    return prompt


def analyze_with_llm(predictions: dict):

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = build_prompt(predictions)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a crop biomass expert and agriculture advisor."},
            {"role": "user", "content": prompt},
        ]
    )

    return response.choices[0].message.content
