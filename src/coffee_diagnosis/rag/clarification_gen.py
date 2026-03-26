"""
Clarification Generator Module (RAC)
Generates clarification questions grounded in retrieved context
"""

from typing import List, Dict, Optional
from anthropic import Anthropic


class ClarificationGenerator:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Clarification Generator

        Args:
            api_key: Anthropic API key (optional, will use env var)
        """
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def generate_question(
        self,
        query: str,
        context: List[Dict],
        previous_answers: List[str],
        missing_info: Dict,
        max_questions: int = 3
    ) -> str:
        """
        Generate a clarification question based on retrieved context

        Args:
            query: Original user query
            context: Retrieved context chunks
            previous_answers: List of previous user answers
            missing_info: Missing symptom information
            max_questions: Maximum questions already asked

        Returns:
            Generated clarification question
        """
        # Format context
        context_text = self._format_context(context)
        history_text = self._format_history(query, previous_answers)
        missing_text = self._format_missing_info(missing_info)

        prompt = f"""You are an agricultural expert specializing in coffee disease diagnosis.

Retrieved Context from Knowledge Base:
{context_text}

Current Conversation:
{history_text}

Missing Information To Clarify:
{missing_text}

IMPORTANT RULES:
1. Ask ONLY ONE clarification question
2. The question MUST be grounded in the retrieved context above
3. Focus on symptoms that help distinguish between diseases
4. Do NOT ask generic or generic questions
5. Do NOT ask about information not related to coffee diseases
6. Do NOT make up symptoms or treatments
7. Keep the question concise (1-2 sentences)
8. Make the question specific and actionable

Generate a single, focused clarification question that will help narrow down the diagnosis:"""

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=150,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            question = response.content[0].text.strip()

            # Remove common prefixes if present
            if question.startswith("Question:"):
                question = question.replace("Question:", "").strip()

            if question.endswith("?"):
                return question
            else:
                return question + "?"

        except Exception as e:
            # Fallback question if API fails
            return self._generate_fallback_question(missing_info)

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
        history = f"User: {query}\n"

        for i, answer in enumerate(previous_answers, 1):
            history += f"User clarification {i}: {answer}\n"

        return history

    def _format_missing_info(self, missing_info: Dict) -> str:
        """Format missing information"""
        if not missing_info:
            return "All basic information has been provided."

        formatted = []
        for key, value in missing_info.items():
            formatted.append(f"- {key.upper()}: {value}")

        return "\n".join(formatted)

    def _generate_fallback_question(self, missing_info: Dict) -> str:
        """Generate a fallback question when API fails"""
        for key, value in missing_info.items():
            return value

        return "Can you provide more details about the symptoms you're observing?"

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
