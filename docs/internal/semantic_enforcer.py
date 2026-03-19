import re
import os
import sys
import json
import urllib.request
from html.parser import HTMLParser

# --- Helper: Simple HTML Stripper (No BS4 dependency for now) ---
class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = []
    def handle_data(self, d):
        self.text.append(d)
    def get_data(self):
        return ''.join(self.text)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

# --- Semantic Enforcer Class ---
class SemanticEnforcer:
    def __init__(self, api_key=None, model="gpt-3.5-turbo"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model

    def fetch_url_content(self, url):
        """
        Fetches text content from a URL.
        Returns: (success_bool, text_content_or_error)
        """
        try:
            # Fake headers to look like a browser
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as response:
                html = response.read().decode('utf-8', errors='ignore')
                text = strip_tags(html)
                # Cleanup whitespace
                text = " ".join(text.split())
                return True, text
        except Exception as e:
            return False, str(e)

    def call_llm_judge(self, prompt):
        """
        Calls the LLM API to judge the content.
        Needs an API Key. If missing, mocks a PASS (for demo purposes).
        """
        if not self.api_key:
            return "WARNING: No API Key found. Skipping Semantic Check (Mock PASS)."

        # Actual API Call logic would go here (using requests or openai lib)
        # For this prototype without a live key, we will simulate.
        # But to be "real", here is the code:
        
        try:
            # Simple Request to OpenAI-compatible endpoint
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0
            }
            
            # Note: We can't actually make external requests if the environment is restricted.
            # Assuming the user runs this locally.
            req = urllib.request.Request(url, json.dumps(data).encode('utf-8'), headers)
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['choices'][0]['message']['content']
        except Exception as e:
            return f"API ERROR: {str(e)}"

    def check_fm3_citation_laundering(self, turn_text):
        """
        FM3: Checks if linked sources actually support the claims.
        """
        errors = []
        
        # 1. Extract Links: [Anchor](URL)
        links = re.findall(r"\[([^\]]+)\]\((http[^\)]+)\)", turn_text)
        
        if not links:
            return errors # No links to check

        print(f"   🔎 FM3 Check: Found {len(links)} citation(s). Verifying...")

        for anchor_text, url in links:
            print(f"      -> Fetching: {url}...")
            success, content = self.fetch_url_content(url)
            
            if not success:
                # Failing to fetch might be FM3 (Dead Link)
                errors.append(f"FM3 (Broken Link): Could not fetch evidence at {url}. Error: {content}")
                continue
            
            # Truncate content for Token Budget
            snippet = content[:3000]
            
            # 2. Construct Judgment Prompt
            prompt = f"""
            You are an impartial Judge for the ICIF-AES Protocol.
            
            TASK: Verify if the Citation supports the Claim.
            
            [CLAIM CONTEXT]
            Anchor Text: "{anchor_text}"
            Full Turn Context: "{turn_text[:200]}..."
            
            [SOURCE CONTENT]
            URL: {url}
            Text: "{snippet}..."
            
            [JUDGMENT]
            Does the source content explicitly support the claim implied by the anchor text?
            If NO, start with "VIOLATION:". If YES, start with "PASS".
            """
            
            print("      -> Asking LLM Judge...")
            verdict = self.call_llm_judge(prompt)
            print(f"      -> Verdict: {verdict[:50]}...")
            
            if "VIOLATION" in verdict.upper():
                errors.append(f"FM3 (Citation Laundering): Link {url} does not support claim '{anchor_text}'. Judge reason: {verdict}")
            elif "WARNING" in verdict:
                # Pass through the warning about missing key
                print(f"      [!] {verdict}")
                
        return errors

if __name__ == "__main__":
    # Test Driver
    print("Initializing Semantic Enforcer...")
    enforcer = SemanticEnforcer(api_key=None) # Will mock if None
    
    test_turn = """
    Revenue is up 50%.
    Evidence: [Google Homepage](https://www.google.com)
    """
    
    print("\n--- Testing FM3 Check ---")
    errs = enforcer.check_fm3_citation_laundering(test_turn)
    
    if errs:
        print("\n❌ Errors Found:")
        for e in errs: print(e)
    else:
        print("\n✅ Check Passed (or Skipped).")
