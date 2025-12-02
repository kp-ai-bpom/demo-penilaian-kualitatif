# soal_generator.py
import json
import re
from pathlib import Path
from langchain.chains import LLMChain
import sys
import os

# Add current directory to path to import prompt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prompt.prompt import CREATE_SOAL_SKJ_PROMPT

class SoalGenerator:
    def __init__(self, llm):
        self.llm = llm
        self.generation_chain = LLMChain(llm=llm, prompt=CREATE_SOAL_SKJ_PROMPT)
    
    def generate_soal(self, skj_data):
        """Generate questions based on SKJ data"""
        try:
            print("Generating questions from SKJ...")
            result = self.generation_chain.invoke({"skj_data": json.dumps(skj_data, ensure_ascii=False)})
            
            # Parse JSON output
            soal_data = self.parse_soal_output(result['text'])
            return soal_data
            
        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            return []
    
    def parse_soal_output(self, text):
        """Parse soal JSON output from LLM"""
        try:
            # Extract JSON array from text
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                raise ValueError("No JSON array found in LLM output")
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return []
    
    def save_soal(self, soal_data, filename):
        """Save generated questions to file"""
        output_file = f"data/soal/{Path(filename).stem}_soal.json"
        Path("data/soal").mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(soal_data, f, indent=2, ensure_ascii=False)
        
        return output_file
    
    def load_soal(self, filename):
        """Load generated questions from file"""
        soal_file = f"data/soal/{Path(filename).stem}_soal.json"
        if Path(soal_file).exists():
            with open(soal_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None