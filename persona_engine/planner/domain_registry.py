"""
Domain Registry — externalized domain definitions.

Separating domain definitions from detection logic allows:
- Per-deployment customization without touching scoring code
- Easier addition/removal of domains
- Testing with custom registries
"""

from persona_engine.planner.domain_detection import DomainEntry

# Built-in domain registry — can be extended or replaced per deployment.
# Note: Whitelist filtering is NOT applied to these hardcoded keywords.
# Short tokens like "ux", "ai" in the registry are preserved.
DOMAIN_REGISTRY: list[DomainEntry] = [
    DomainEntry(
        domain_id="psychology",
        keywords={
            "psychology": 1.0, "behavior": 0.8, "cognitive": 0.9, "research": 0.6,
            "user": 0.5, "ux": 0.9, "experience": 0.5, "mental": 0.7,
            "emotions": 0.6, "perception": 0.6, "learning": 0.5, "memory": 0.5,
            "motivation": 0.6, "personality": 0.6, "bias": 0.7, "heuristic": 0.7,
            "usability": 0.8, "interview": 0.5, "participant": 0.7, "study": 0.5,
        },
        negative_keywords=["physics", "chemistry", "engineering"],
    ),
    DomainEntry(
        domain_id="technology",
        keywords={
            "technology": 1.0, "software": 0.9, "code": 0.8, "programming": 0.9,
            "computer": 0.7, "data": 0.6, "algorithm": 0.8, "system": 0.5,
            "app": 0.6, "web": 0.7, "design": 0.5, "tool": 0.5, "platform": 0.5,
            "api": 0.8, "database": 0.7, "cloud": 0.6, "ai": 0.7, "ml": 0.7,
            "prototype": 0.6, "figma": 0.7, "sketch": 0.6,
        },
        negative_keywords=["biology", "medicine"],
    ),
    DomainEntry(
        domain_id="business",
        keywords={
            "business": 1.0, "company": 0.7, "market": 0.7, "strategy": 0.8,
            "management": 0.7, "project": 0.6, "stakeholder": 0.8, "agile": 0.7,
            "deadline": 0.5, "budget": 0.6, "roi": 0.8, "kpi": 0.7, "revenue": 0.7,
            "customer": 0.6, "client": 0.6, "meeting": 0.4, "team": 0.5,
            "sprint": 0.6, "roadmap": 0.7,
        },
        negative_keywords=[],
    ),
    DomainEntry(
        domain_id="health",
        keywords={
            "health": 1.0, "medical": 0.9, "doctor": 0.8, "disease": 0.7,
            "symptom": 0.8, "treatment": 0.8, "diagnosis": 0.9, "medicine": 0.8,
            "therapy": 0.7, "wellness": 0.6, "exercise": 0.5, "nutrition": 0.6,
            "diet": 0.5, "mental health": 0.8, "anxiety": 0.7, "depression": 0.7,
        },
        negative_keywords=[],
    ),
    DomainEntry(
        domain_id="personal",
        keywords={
            "personal": 0.8, "family": 0.7, "relationship": 0.8, "friend": 0.6,
            "hobby": 0.6, "travel": 0.5, "home": 0.5, "life": 0.4, "feel": 0.5,
            "opinion": 0.4, "think": 0.3, "believe": 0.4, "prefer": 0.4,
        },
        negative_keywords=[],
    ),
    DomainEntry(
        domain_id="food",
        keywords={
            "food": 1.0, "cook": 0.9, "cooking": 0.9, "chef": 0.9, "kitchen": 0.8,
            "recipe": 0.9, "cuisine": 0.9, "restaurant": 0.8, "meal": 0.7,
            "ingredient": 0.8, "sauce": 0.8, "bake": 0.7, "baking": 0.7,
            "grill": 0.7, "roast": 0.7, "ferment": 0.7, "fermentation": 0.7,
            "pastry": 0.8, "butcher": 0.7, "butchery": 0.7, "flavor": 0.7,
            "flavour": 0.7, "culinary": 0.9, "gastronomy": 0.9, "dish": 0.6,
            "menu": 0.7, "plating": 0.7, "seasoning": 0.7, "spice": 0.6,
            "broth": 0.7, "stock": 0.5, "knife": 0.6, "oven": 0.6,
            "saut\u00e9": 0.7, "simmer": 0.7, "blanch": 0.7, "frozen": 0.4,
            "fresh": 0.4, "organic": 0.5, "farm": 0.5, "table": 0.3,
            # Classical sauces & techniques
            "bechamel": 0.9, "b\u00e9chamel": 0.9, "hollandaise": 0.9,
            "veloute": 0.9, "velout\u00e9": 0.9, "roux": 0.8, "reduction": 0.7,
            "emulsion": 0.7, "deglaze": 0.8, "whisk": 0.6, "fold": 0.5,
            "temper": 0.6, "braise": 0.8, "poach": 0.7, "stew": 0.7,
            # Ingredients
            "butter": 0.5, "cream": 0.5, "egg": 0.4, "flour": 0.5,
            "cheese": 0.5, "meat": 0.6, "steak": 0.7, "chicken": 0.6,
            "fish": 0.5, "vegetable": 0.5, "pasta": 0.7, "bread": 0.6,
            "dough": 0.7, "soup": 0.7, "salad": 0.6,
            # Taste & texture
            "umami": 0.8, "savory": 0.6, "caramelize": 0.8, "sear": 0.8,
        },
        negative_keywords=["computer", "software", "code"],
    ),
    DomainEntry(
        domain_id="science",
        keywords={
            "science": 1.0, "physics": 0.9, "chemistry": 0.9, "biology": 0.9,
            "experiment": 0.8, "hypothesis": 0.9, "theory": 0.7, "quantum": 0.9,
            "molecule": 0.8, "atom": 0.8, "cell": 0.6, "evolution": 0.8,
            "gravity": 0.8, "particle": 0.8, "electron": 0.8, "photon": 0.8,
            "equation": 0.7, "formula": 0.6, "laboratory": 0.7, "research": 0.5,
            "genome": 0.8, "dna": 0.8, "protein": 0.7, "entropy": 0.8,
            "thermodynamics": 0.9, "relativity": 0.9, "astronomy": 0.9,
            "telescope": 0.7, "planet": 0.7, "star": 0.5, "galaxy": 0.8,
            "neuroscience": 0.9, "ecology": 0.8, "species": 0.7,
        },
        negative_keywords=["business", "stakeholder"],
    ),
    DomainEntry(
        domain_id="arts",
        keywords={
            "art": 1.0, "music": 0.9, "painting": 0.9, "sculpture": 0.8,
            "theater": 0.8, "theatre": 0.8, "poetry": 0.8, "novel": 0.7,
            "literature": 0.9, "compose": 0.7, "composition": 0.7, "melody": 0.8,
            "harmony": 0.7, "rhythm": 0.7, "guitar": 0.7, "piano": 0.7,
            "orchestra": 0.8, "symphony": 0.8, "gallery": 0.7, "exhibition": 0.7,
            "canvas": 0.8, "sketch": 0.6, "creative": 0.5, "aesthetic": 0.7,
            "film": 0.7, "cinema": 0.8, "photography": 0.7, "dance": 0.7,
            "opera": 0.8, "ceramic": 0.7, "printmaking": 0.8,
        },
        negative_keywords=["database", "software"],
    ),
    DomainEntry(
        domain_id="education",
        keywords={
            "education": 1.0, "teaching": 0.9, "teacher": 0.8, "student": 0.7,
            "curriculum": 0.9, "pedagogy": 0.9, "classroom": 0.8, "school": 0.7,
            "university": 0.7, "lecture": 0.7, "homework": 0.7, "exam": 0.7,
            "assessment": 0.7, "grading": 0.7, "syllabus": 0.8, "tutoring": 0.8,
            "literacy": 0.8, "lesson": 0.7, "academic": 0.6, "scholarship": 0.7,
            "diploma": 0.7, "degree": 0.5, "mentor": 0.6,
        },
        negative_keywords=[],
    ),
    DomainEntry(
        domain_id="law",
        keywords={
            "law": 1.0, "legal": 0.9, "court": 0.8, "judge": 0.8, "lawyer": 0.9,
            "attorney": 0.9, "statute": 0.9, "regulation": 0.7, "contract": 0.8,
            "plaintiff": 0.9, "defendant": 0.9, "verdict": 0.9, "trial": 0.8,
            "litigation": 0.9, "jurisdiction": 0.8, "constitution": 0.8,
            "precedent": 0.8, "testimony": 0.8, "prosecution": 0.8,
            "compliance": 0.7, "liability": 0.8, "tort": 0.9, "patent": 0.8,
            "copyright": 0.7, "intellectual property": 0.8,
        },
        negative_keywords=[],
    ),
    DomainEntry(
        domain_id="sports",
        keywords={
            "sport": 1.0, "sports": 1.0, "athlete": 0.9, "training": 0.7,
            "coach": 0.8, "fitness": 0.7, "workout": 0.7, "competition": 0.7,
            "championship": 0.8, "tournament": 0.8, "match": 0.5, "game": 0.5,
            "team": 0.4, "score": 0.5, "goal": 0.4, "basketball": 0.8,
            "football": 0.8, "soccer": 0.8, "tennis": 0.8, "swimming": 0.7,
            "marathon": 0.8, "sprint": 0.6, "weightlifting": 0.8, "gym": 0.7,
            "endurance": 0.7, "cardio": 0.7, "muscle": 0.6, "protein": 0.4,
        },
        negative_keywords=["software", "code", "programming"],
    ),
    DomainEntry(
        domain_id="finance",
        keywords={
            "finance": 1.0, "investment": 0.9, "stock": 0.7, "bond": 0.8,
            "portfolio": 0.9, "dividend": 0.9, "interest rate": 0.9,
            "inflation": 0.8, "mortgage": 0.8, "banking": 0.8, "credit": 0.7,
            "debt": 0.7, "loan": 0.7, "tax": 0.7, "accounting": 0.8,
            "audit": 0.7, "hedge fund": 0.9, "mutual fund": 0.8,
            "cryptocurrency": 0.8, "blockchain": 0.7, "equity": 0.8,
            "valuation": 0.8, "asset": 0.7, "liability": 0.6, "capital": 0.6,
        },
        negative_keywords=[],
    ),
]
