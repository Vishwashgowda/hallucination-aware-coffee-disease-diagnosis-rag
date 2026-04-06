"""
Clarification Generator Module (RAC)
Generates clarification questions grounded in retrieved context
"""

from typing import List, Dict, Optional
from coffee_diagnosis.core.llm_client import LLMClient


class ClarificationGenerator:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize Clarification Generator

        Args:
            llm_client: Shared LLM client (defaults to local OpenAI-compatible endpoint)
        """
        self.llm = llm_client or LLMClient()

    def generate_question(
        self,
        query: str,
        context: List[Dict],
        previous_answers: List[str],
        missing_info: Dict,
        max_questions: int = 3
    ) -> str:
        """
        Generate a clarification question based on retrieved context and what's missing

        Args:
            query: Original user query
            context: Retrieved context chunks
            previous_answers: List of previous user answers
            missing_info: Missing symptom information (keys only, like 'color', 'pattern')
            max_questions: Maximum questions already asked

        Returns:
            Generated clarification question
        """
        # Format context
        context_text = self._format_context(context)
        history_text = self._format_history(query, previous_answers)

        # Create a concise list of what's missing
        missing_keys = list(missing_info.keys()) if missing_info else []

        # Off-topic guard: if the user hasn't mentioned coffee plants or plant symptoms, redirect
        combined_text = f"{query} {' '.join(previous_answers)}".lower()
        plant_terms = ["coffee", "leaf", "leaves", "plant", "tree", "berry", "berries", "stem"]
        if not any(term in combined_text for term in plant_terms):
            return (
                "I can help with coffee plant health. Please describe the coffee plant symptoms "
                "(e.g., leaf color/pattern, spots, wilting, affected parts like leaves, stems, or berries)."
            )

        if not missing_keys:
            return "Do you see any other symptoms or changes on the plant?"

        # Use Claude to generate smart follow-up questions
        prompt = f"""You are helping a farmer diagnose their coffee plant problem. Ask ONE clear, specific question.

User's symptoms so far:
{history_text}

What's missing: {', '.join(missing_keys)}

RULES FOR YOUR QUESTION:
1. Maximum 20-25 words - be clear and specific
2. Ask about ONE thing only (not multiple things)
3. Use simple, practical language - avoid overly technical terms
4. End with exactly ONE question mark (?)
5. NEVER use ?? or multiple question marks
6. Ask for observable details the farmer can see
7. Examples of GOOD questions:
   - "Are the yellow leaves appearing on the older lower branches or on the newer growth at the top?"
   - "Do the affected leaves have distinct spots with defined edges, or is the discoloration more spread out?"
   - "Did this yellowing start suddenly within the last few days, or has it been developing slowly over weeks?"

8. BAD - DO NOT copy these patterns:
   - Using ?? at the end (NEVER DO THIS)
   - Overly technical disease names or scientific terminology
   - Multiple sub-questions bundled together

Generate ONE clear question (20-25 words, ending with one ?):"""

        try:
            question = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
            )

            question = self._clean_question(question)
            return question

        except Exception as e:
            print(f"[ERROR] Failed to generate question: {str(e)}")
            # Fallback to a generic but smart question
            return self._generate_fallback_question(missing_keys)

    def _format_context(self, context: List[Dict]) -> str:
        """Format context chunks for the prompt"""
        formatted = []

        for i, chunk in enumerate(context[:3], 1):  # Use top 3
            source = chunk.get('source', 'Unknown')
            content = chunk['content'][:300]  # Truncate to 300 chars

            formatted.append(f"[{i}] Source: {source}\n{content}...")

        return "\n\n".join(formatted)

    def _format_history(self, query: str, previous_answers: List[str]) -> str:
        """Format conversation history"""
        if not previous_answers:
            return f"Initial observation: {query}"

        history = f"Initial symptom: {query}\n"

        for i, answer in enumerate(previous_answers, 1):
            history += f"Follow-up {i}: {answer}\n"

        return history

    def _generate_fallback_question(self, missing_info) -> str:
        """Generate a smart fallback question when API fails"""
        # missing_info is now a list of keys like ['color', 'pattern']
        if isinstance(missing_info, dict):
            missing_keys = list(missing_info.keys())
        else:
            missing_keys = missing_info if isinstance(missing_info, list) else []

        # Simple, short questions (max 15 words each)
        fallback_questions = {
            'color': "What color is the problem area - yellow, brown, black, or red?",
            'pattern': "Are there distinct spots or is the discoloration spread out?",
            'location': "Which leaves are affected - older ones or newer growth?",
            'spread': "Where on the plant do you see the symptoms?",
            'timing': "Did this appear suddenly or develop slowly over time?",
            'texture': "Do affected areas look dry, wet, or powdery?",
            'berry_fruit': "Are the coffee berries showing any problems?",
        }

        # Ask about the first missing attribute
        if missing_keys:
            return fallback_questions.get(missing_keys[0], "Can you describe what you see in more detail?")

        return "Do you notice any other unusual changes on the plant?"

    def _clean_question(self, question: str) -> str:
        """Keep only the main question line and ensure it ends with a question mark."""
        if not question:
            return "Can you share more details about the symptoms?"

        # Remove prefixed labels and reasoning/explanations after newlines
        if question.lower().startswith("question:"):
            question = question.split(":", 1)[1].strip()

        # Take the first non-empty line as the question
        lines = [l.strip() for l in question.splitlines() if l.strip()]
        question = lines[0] if lines else question.strip()

        # Trim trailing reasoning markers
        for marker in ["reasoning:", "explanation:", "because"]:
            idx = question.lower().find(marker)
            if idx > 0:
                question = question[:idx].strip()

        # Fix malformed endings
        # First, remove any trailing quotes (but preserve question marks inside them)
        # Then ensure proper question mark at the end
        
        # Remove double/triple question marks: text?? or text??? → text?
        while question.endswith('??'):
            question = question[:-1]
        
        # Remove patterns like: text"? → text?
        if question.endswith('"?'):
            question = question[:-2] + '?'
        elif question.endswith("'?"):
            question = question[:-2] + '?'
        # Remove patterns like: text." or text.' → text?
        elif question.endswith('."'):
            question = question[:-2] + '?'
        elif question.endswith(".'"):
            question = question[:-2] + '?'
        # Remove trailing quotes without question mark: text" → text?
        elif question.endswith('"'):
            question = question[:-1] + '?'
        elif question.endswith("'"):
            question = question[:-1] + '?'
        # Ensure question mark at end
        elif not question.endswith("?"):
            question = question.rstrip(".") + "?"
        
        return question

    def validate_question(self, question: str, context: List[Dict]) -> bool:
        """
        Validate that question is grounded in context

        Args:
            question: Question to validate
            context: Retrieved context

        Returns:
            True if question appears grounded, False otherwise
        """
        question_lower = question.lower()
        context_text = " ".join([c['content'].lower() for c in context])

        # Check if key words from question appear in context
        words = [w for w in question_lower.split() if len(w) > 3]
        matches = sum(1 for word in words if word in context_text)

        return matches > 0
