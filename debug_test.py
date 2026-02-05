"""Simple debug test"""
import yaml
from pathlib import Path

# First just test YAML loading
persona_path = Path("personas/ux_researcher.yaml")
with open(persona_path, 'r') as f:
    data = yaml.safe_load(f)

print("YAML loaded successfully")
print(f"Keys: {list(data.keys())[:10]}")

# Now try importing
try:
    from persona_engine.schema import Persona
    print("Import successful")
    
    # Try to create persona
    persona = Persona(**data)
    print(f"SUCCESS! Persona created: {persona.label}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
