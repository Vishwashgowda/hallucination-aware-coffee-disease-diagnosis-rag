"""
State Manager Module
Maintains conversation state across multi-turn interactions
"""

from typing import List, Dict
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConversationState:
    """Maintains state of the conversation"""
    initial_query: str = ""
    user_responses: List[str] = field(default_factory=list)
    system_questions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    questions_asked: int = 0
    retrieved_context: List[Dict] = field(default_factory=list)
    detected_ambiguities: Dict = field(default_factory=dict)
    possible_diseases: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class StateManager:
    def __init__(self):
        """Initialize State Manager"""
        self.state = ConversationState()

    def initialize(self, query: str) -> None:
        """
        Initialize state with user query

        Args:
            query: Initial user query
        """
        self.state = ConversationState(initial_query=query)
        self.state.confidence = 0.2  # Start with low confidence

    def add_user_response(self, response: str) -> None:
        """Add user response to state"""
        self.state.user_responses.append(response)
        self.increase_confidence(0.15)  # Each answer increases confidence

    def add_system_question(self, question: str) -> None:
        """Add system question to state"""
        self.state.system_questions.append(question)
        self.state.questions_asked += 1

    def increase_confidence(self, amount: float) -> None:
        """Increase confidence score"""
        self.state.confidence = min(self.state.confidence + amount, 1.0)

    def set_confidence(self, value: float) -> None:
        """Set confidence score directly"""
        self.state.confidence = min(max(value, 0.0), 1.0)

    def update_detected_ambiguities(self, ambiguities: Dict) -> None:
        """Update detected ambiguities"""
        self.state.detected_ambiguities = ambiguities

    def update_context(self, context: List[Dict]) -> None:
        """Update retrieved context"""
        self.state.retrieved_context = context

    def set_possible_diseases(self, diseases: List[str]) -> None:
        """Set possible diseases"""
        self.state.possible_diseases = diseases

    def should_stop(self) -> bool:
        """
        Check if conversation should stop

        Conditions:
        - confidence > 0.8 OR
        - only 1 disease remains OR
        - questions_asked >= 3
        """
        if self.state.confidence > 0.8:
            return True

        if len(self.state.possible_diseases) == 1:
            return True

        if self.state.questions_asked >= 3:
            return True

        return False

    def get_conversation_history(self) -> str:
        """Get formatted conversation history"""
        history = f"Initial Query: {self.state.initial_query}\n\n"

        for i, (question, response) in enumerate(
            zip(self.state.system_questions, self.state.user_responses),
            1
        ):
            history += f"Q{i}: {question}\n"
            history += f"A{i}: {response}\n\n"

        return history

    def get_state_summary(self) -> Dict:
        """Get summary of current state"""
        return {
            'initial_query': self.state.initial_query,
            'confidence': self.state.confidence,
            'questions_asked': self.state.questions_asked,
            'ambiguities': self.state.detected_ambiguities,
            'possible_diseases': self.state.possible_diseases,
            'history_length': len(self.state.user_responses)
        }

    def reset(self) -> None:
        """Reset state"""
        self.state = ConversationState()

    @property
    def confidence(self) -> float:
        return self.state.confidence

    @property
    def questions_asked(self) -> int:
        return self.state.questions_asked

    @property
    def responses_received(self) -> int:
        return len(self.state.user_responses)
