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

        if not missing_keys:
            return "Do you see any other symptoms or changes on the plant?"

        # Use Claude to generate smart follow-up questions
        prompt = f"""You are an expert agricultural specialist diagnosing coffee diseases.

User's symptom description so far:
{history_text}

Retrieved information from knowledge base:
{context_text}

        Missing information details needed: {', '.join(missing_keys)}

        IMPORTANT:
1. Ask ONE smart follow-up question to clarify the missing information
2. The question must be grounded in coffee disease symptoms
3. Reference what the user ALREADY told you to show continuity
4. Do NOT assume facts the user hasn't stated; avoid presupposing specifics like "older leaves" unless the user mentioned it. If you need to know, ask neutrally (e.g., "Are the yellow leaves on older or newer growth?")
5. Ask naturally, not like a form
6. Keep it concise (1 sentence)
7. Do NOT ask yes/no questions - ask for details
8. Make the question flow naturally from what they said

Example good questions:
- "You mentioned yellow spots - do you see them mostly on older leaves or newer growth?"
- "Since the browning started recently, has it spread to the stem or is it just on leaves?"
- "Are the spots dry and papery, or do they feel soft and wet?"

Generate ONE natural follow-up question:"""

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

        # More detailed, technical questions based on coffee disease symptoms
        fallback_questions = {
            'color': "Is the discoloration yellow, brown, red, or black? Does it appear between the veins, on the leaf margins, or all over?",
            'pattern': "Does the damage appear as circular spots, streaks, patches, or does the tissue look wilted/twisted? Are the spots round with concentric rings?",
            'location': "Are the symptoms on the oldest leaves first progressing upward, or are they scattered throughout the canopy? Are stems, petioles (leaf stalks), or only leaf blades affected?",
            'spread': "Does the yellowing appear on the oldest leaves (starting from petiole and midrib) or on the youngest leaves? Is it isolated to one side of the plant?",
            'timing': "Did the symptoms appear suddenly or gradually? Are they getting worse daily, weekly, or staying the same?",
            'texture': "Do the affected areas feel dry and papery, soft and waterlogged, or powdery? Is there a visible coating or fungal growth?",
            'berry_fruit': "Are the coffee berries affected? Do they show spots, rot, or borer holes? What is their color - green, red, or black?",
        }

        # Ask about the first missing attribute with more detail
        if missing_keys:
            return fallback_questions.get(missing_keys[0], "Can you describe the symptoms in more detail, including color, texture, and affected plant parts?")

        return "Do you observe any other changes like wilting, leaf curling (whiptail), yellowing between veins, or unusual growth patterns?"

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

        if not question.endswith("?"):
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
