"""
Hallucination Checker Module (SelfCheckGPT)
Verifies consistency of diagnoses to detect hallucinations
"""

from typing import List, Dict, Optional
from coffee_diagnosis.core.llm_client import LLMClient


class HallucinationChecker:
    def __init__(self, llm_client: Optional[LLMClient] = None, num_generations: int = 1):
        """
        Initialize Hallucination Checker

        Args:
            llm_client: Shared LLM client (defaults to local OpenAI-compatible endpoint)
            num_generations: Number of independent diagnosis generations (kept low for rate limits)
        """
        self.llm = llm_client or LLMClient()
        self.num_generations = num_generations

    def check_hallucination(
        self,
        query: str,
        user_responses: List[str],
        context: List[Dict],
        initial_diagnosis: str
    ) -> Dict:
        """
        Check for hallucinations by generating diagnosis multiple times

        Args:
            query: Original user query
            user_responses: List of user responses
            context: Retrieved context
            initial_diagnosis: Initial diagnosis result

        Returns:
            Dictionary with consistency scores and warnings
        """
        print(f"Generating {self.num_generations} independent diagnoses for verification...")

        diagnoses = []

        for i in range(self.num_generations):
            diagnosis = self._generate_diagnosis(query, user_responses, context)
            diagnoses.append(diagnosis)
            print(f"  Generation {i+1} complete")

        # Analyze consistency
        consistency_report = self._analyze_consistency(diagnoses, initial_diagnosis)

        return consistency_report

    def _generate_diagnosis(
        self,
        query: str,
        user_responses: List[str],
        context: List[Dict]
    ) -> str:
        """Generate a single diagnosis"""
        context_text = self._format_context(context)
        history_text = self._format_history(query, user_responses)

        prompt = f"""You are a coffee disease expert. Based ONLY on the information provided, diagnose the disease.

CONTEXT:
{context_text}

USER INFORMATION:
{history_text}

RULES:
1. Use ONLY information from the context
2. Do NOT hallucinate diseases or treatments
3. Be concise
4. State the disease name clearly

Diagnosis (just the disease name and key symptoms)."""

        try:
            diagnosis_text = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
            )
            if not diagnosis_text:
                diagnosis_text = "No diagnosis generated"
            return diagnosis_text

        except Exception as e:
            return f"Error: {str(e)}"

    def _format_context(self, context: List[Dict]) -> str:
        """Format context"""
        formatted = []
        for chunk in context[:3]:
            try:
                if isinstance(chunk, dict):
                    content = chunk.get('content', '')[:200]
                else:
                    content = str(chunk)[:200]
                formatted.append(content)
            except Exception as e:
                formatted.append(f"[Error processing chunk: {str(e)}]")
        return "\n\n".join(formatted)

    def _format_history(self, query: str, responses: List[str]) -> str:
        """Format history"""
        history = query
        for response in responses:
            history += f"\nUser: {response}"
        return history

    def _analyze_consistency(
        self,
        diagnoses: List[str],
        initial_diagnosis: str
    ) -> Dict:
        """
        Analyze consistency across multiple diagnoses

        Args:
            diagnoses: List of generated diagnoses
            initial_diagnosis: Initial diagnosis

        Returns:
            Consistency report
        """
        # Extract disease names
        disease_names = []
        for diagnosis in diagnoses:
            # Simple extraction - get first capitalized word or phrase
            words = diagnosis.split()
            if words:
                disease_names.append(self._extract_disease_name(diagnosis))

        # Check consistency
        consistent_diseases = set(disease_names)
        consistency_score = 1.0 if len(consistent_diseases) == 1 else 0.5

        hallucination_detected = False
        warnings = []

        if len(consistent_diseases) > 1:
            hallucination_detected = True
            warnings.append(
                f"[WARNING] Multiple different diagnoses generated: {', '.join(consistent_diseases)}"
            )

        # Check if treatment recommendations are consistent
        treatment_consistency = self._check_treatment_consistency(diagnoses)
        if treatment_consistency < 0.7:
            hallucination_detected = True
            warnings.append("[WARNING] Treatment recommendations inconsistent across runs")

        # Extract key symptoms and check consistency
        symptom_mentions = self._extract_symptoms(diagnoses)
        if len(symptom_mentions) > 5:
            hallucination_detected = True
            warnings.append("[WARNING] Symptoms vary significantly across generations")

        return {
            'consistent': not hallucination_detected,
            'consistency_score': consistency_score,
            'disease_name': list(consistent_diseases)[0] if consistent_diseases else "Unknown",
            'all_diseases': list(consistent_diseases),
            'hallucination_detected': hallucination_detected,
            'warnings': warnings,
            'num_generations': self.num_generations,
            'agreement': f"{int((1 - len(consistent_diseases) / self.num_generations) * 100)}%"
        }

    def _extract_disease_name(self, diagnosis_text: str) -> str:
        """Extract disease name from diagnosis text"""
        # Simple heuristic: first line or first capitalized phrase
        lines = diagnosis_text.split('\n')
        if lines:
            line = lines[0].strip()
            # Remove common prefixes
            for prefix in ['Disease:', 'Diagnosis:', 'The']:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
            return line
        return "Unknown"

    def _check_treatment_consistency(self, diagnoses: List[str]) -> float:
        """Check if treatment recommendations are similar"""
        # Simple check: count how many diagnoses mention treatment-related keywords
        treatment_keywords = ['treatment', 'spray', 'fungicide', 'manage', 'control', 'apply']
        mention_count = 0

        for diagnosis in diagnoses:
            diagnosis_lower = diagnosis.lower()
            if any(keyword in diagnosis_lower for keyword in treatment_keywords):
                mention_count += 1

        return mention_count / len(diagnoses) if diagnoses else 0.0

    def _extract_symptoms(self, diagnoses: List[str]) -> List[str]:
        """Extract mentioned symptoms across all diagnoses"""
        symptom_keywords = [
            'yellow', 'brown', 'spot', 'leaf', 'stem', 'rust', 'borer', 'scale',
            'rot', 'wilt', 'pattern', 'disease'
        ]

        symptoms = set()
        for diagnosis in diagnoses:
            diagnosis_lower = diagnosis.lower()
            for symptom in symptom_keywords:
                if symptom in diagnosis_lower:
                    symptoms.add(symptom)

        return list(symptoms)

    def format_verification_report(self, report: Dict) -> str:
        """Format verification report for display"""
        status = "[OK] VERIFIED" if report['consistent'] else "[WARNING] ISSUE DETECTED"

        output = f"""
{"="*50}
{status} - Hallucination Check
{"="*50}

Consistency Score: {report['consistency_score'] * 100:.1f}%
Agreement Level: {report['agreement']}

Disease Agreement: {', '.join(report['all_diseases'])}

{f"Warnings:" + chr(10) + chr(10).join(report['warnings']) if report['warnings'] else "No issues detected"}

Generations Checked: {report['num_generations']}
{"="*50}
"""
        return output
