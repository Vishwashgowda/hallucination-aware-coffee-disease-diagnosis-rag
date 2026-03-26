"""
Diagnosis Generator Module
Generates final diagnosis and treatment recommendations
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from anthropic import Anthropic


@dataclass
class Diagnosis:
    """Diagnosis result"""
    disease_name: str
    confidence: float
    reason: str
    treatment: str
    prevention: str
    source: str


class DiagnosisGenerator:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Diagnosis Generator

        Args:
            api_key: Anthropic API key (optional, will use env var)
        """
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def generate_diagnosis(
        self,
        query: str,
        user_responses: List[str],
        context: List[Dict],
        confidence: float
    ) -> Diagnosis:
        """
        Generate final diagnosis based on all collected information

        Args:
            query: Original user query
            user_responses: List of user's clarification answers
            context: Retrieved context chunks
            confidence: Current confidence score

        Returns:
            Diagnosis object with disease name, reasons, and treatment
        """
        # Format context and history
        context_text = self._format_context(context)
        history_text = self._format_history(query, user_responses)

        prompt = f"""You are an expert agricultural scientist specializing in coffee disease diagnosis in Karnataka region.

Based ONLY on the following information, provide a diagnosis:

RETRIEVED KNOWLEDGE BASE:
{context_text}

USER INFORMATION:
{history_text}

DIAGNOSTIC RULES:
1. Provide diagnosis ONLY based on the retrieved knowledge base above
2. Do NOT mention diseases not found in the knowledge base
3. Do NOT hallucinate symptoms or treatments
4. If you cannot confidently diagnose, say so explicitly
5. Provide specific treatment recommendations from the knowledge base
6. Format your response as:
   - DISEASE: [name]
   - CONFIDENCE: [0-100]%
   - SYMPTOMS MATCH: [list symptoms that match]
   - REASON: [why this diagnosis]
   - TREATMENT: [specific treatment steps]
   - PREVENTION: [prevention measures]
   - SOURCE PDF: [which PDF this came from]

Provide the diagnosis now:"""

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                stream=False,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # Properly extract text from Anthropic response
            diagnosis_text = ""

            # The response has a .content list
            if hasattr(response, 'content') and isinstance(response.content, list):
                for block in response.content:
                    # Use only text blocks; ignore thinking/tool blocks
                    if getattr(block, 'type', '') == 'text' and hasattr(block, 'text'):
                        diagnosis_text = block.text.strip()
                        break

            # Fallback: try direct attribute access - avoid thinking blocks
            if not diagnosis_text and hasattr(response, 'content'):
                try:
                    for block in response.content:
                        if hasattr(block, 'text') and not hasattr(block, 'thinking'):
                            diagnosis_text = block.text.strip()
                            break
                except:
                    pass

            # Last resort: check if response itself has text
            if not diagnosis_text and hasattr(response, 'text'):
                diagnosis_text = response.text.strip()

            # If still no text, something is wrong
            if not diagnosis_text:
                raise ValueError("Could not extract textual diagnosis from model response")

            # Parse the response
            diagnosis = self._parse_diagnosis(diagnosis_text, confidence)
            return diagnosis

        except Exception as e:
            print(f"[ERROR] Exception in generate_diagnosis: {str(e)}")
            import traceback
            traceback.print_exc()
            return Diagnosis(
                disease_name="Unable to diagnose",
                confidence=0.0,
                reason=f"Error during diagnosis: {str(e)}",
                treatment="Please consult a local agricultural expert",
                prevention="",
                source="Error"
            )

    def _format_context(self, context: List[Dict]) -> str:
        """Format context chunks"""
        formatted = []

        if not context:
            return "No context available"

        for i, chunk in enumerate(context[:5], 1):  # Use top 5 chunks
            try:
                # Handle both dict and Document objects
                if isinstance(chunk, dict):
                    content = chunk.get('content', '')
                    source = chunk.get('source', 'Unknown')
                else:
                    content = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
                    source = chunk.metadata.get('source_file', 'Unknown') if hasattr(chunk, 'metadata') else 'Unknown'

                formatted.append(f"[From {source}]\n{content}")
            except Exception as e:
                print(f"[ERROR] Error formatting context chunk {i}: {e}")
                continue

        return "\n\n---\n\n".join(formatted)

    def _format_history(self, query: str, responses: List[str]) -> str:
        """Format user information"""
        history = f"Initial Symptom Description:\n{query}\n\n"

        if responses:
            history += "Additional Clarifications:\n"
            for i, response in enumerate(responses, 1):
                history += f"{i}. {response}\n"

        return history

    def _parse_diagnosis(self, response_text: str, confidence: float) -> Diagnosis:
        """Parse diagnosis from model response"""
        if not response_text:
            return Diagnosis(
                disease_name="Unable to determine",
                confidence=0.0,
                reason="No response from model",
                treatment="",
                prevention="",
                source=""
            )

        lines = response_text.split('\n')
        disease_name = "Unable to determine"
        reason = ""
        treatment = ""
        prevention = ""
        source = "Knowledge base"

        # Coffee diseases database
        known_diseases = {
            'leaf rust': ['rust', 'orange', 'powder', 'underside'],
            'coffee leaf miner': ['miner', 'tunnel', 'winding', 'leaf'],
            'brown eye spot': ['brown eye', 'eye spot', 'spot disease'],
            'anthracnose': ['anthracnose', 'black rot'],
            'root rot': ['root rot', 'wet soil', 'soggy'],
            'stem canker': ['canker', 'lesion', 'branch', 'gum'],
            'branch canker': ['canker', 'lesion', 'branch', 'oozing'],
            'coffee berry borer': ['borer', 'berry'],
            'scale insect': ['scale'],
            'red spider mite': ['mite', 'red'],
            'nitrogen deficiency': ['nitrogen', 'yellowing', 'deficiency'],
        }

        # Convert text to lowercase for searching
        text_lower = response_text.lower()

        # Look for specific disease patterns in known_diseases
        best_match = None
        best_score = 0

        for disease, keywords in known_diseases.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > best_score:
                best_score = score
                best_match = disease

        if best_match and best_score > 0:
            disease_name = best_match.title()
        else:
            # Fallback: look for disease keywords in order
            disease_patterns = [
                'coffee leaf rust',
                'leaf rust',
                'stem canker',
                'branch canker',
                'root rot',
                'anthracnose',
                'brown eye spot',
                'coffee leaf miner',
                'berry borer',
                'scale insect',
                'red spider mite',
                'nitrogen deficiency',
                'phosphorus deficiency',
                'potassium deficiency',
            ]

            for pattern in disease_patterns:
                if pattern in text_lower:
                    disease_name = pattern.title()
                    break

        # Extract reason, treatment, prevention from specific sections
        full_text = response_text

        # Extract reason
        if 'reason:' in full_text.lower():
            reason_start = full_text.lower().find('reason:')
            reason_section = full_text[reason_start:].split('\n')[1:3]
            reason = ' '.join([line.strip() for line in reason_section if line.strip()])[:200]

        # Extract treatment
        if 'treatment:' in full_text.lower():
            treatment_start = full_text.lower().find('treatment:')
            treatment_section = full_text[treatment_start:].split('\n')[1:3]
            treatment = ' '.join([line.strip() for line in treatment_section if line.strip()])[:200]

        # Extract prevention
        if 'prevention:' in full_text.lower():
            prevention_start = full_text.lower().find('prevention:')
            prevention_section = full_text[prevention_start:].split('\n')[1:3]
            prevention = ' '.join([line.strip() for line in prevention_section if line.strip()])[:200]

        return Diagnosis(
            disease_name=disease_name,
            confidence=confidence,
            reason=reason if reason else "Based on symptoms provided",
            treatment=treatment if treatment else "Consult agricultural expert",
            prevention=prevention if prevention else "Preventive measures recommended",
            source=source
        )

    def format_diagnosis_output(self, diagnosis: Diagnosis) -> str:
        """Format diagnosis for display"""
        output = f"""
===================================================
                DIAGNOSIS RESULT
===================================================

Disease: {diagnosis.disease_name}
Confidence: {diagnosis.confidence * 100:.1f}%

Reason:
{diagnosis.reason}

Treatment:
{diagnosis.treatment}

Prevention:
{diagnosis.prevention}

Source: {diagnosis.source}
===================================================
"""
        return output
