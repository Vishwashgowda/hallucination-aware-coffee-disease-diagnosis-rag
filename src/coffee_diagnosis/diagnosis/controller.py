"""
Multi-turn Controller Module
Orchestrates the complete multi-turn coffee disease diagnosis pipeline
"""

from typing import Optional, Dict, List
from coffee_diagnosis.core.pdf_loader import PDFLoader
from coffee_diagnosis.core.vector_store import FAISSVectorStore
from coffee_diagnosis.core.retriever import Retriever
from coffee_diagnosis.rag.ambiguity_detector import AmbiguityDetector
from coffee_diagnosis.rag.retrieval_evaluator import RetrievalEvaluator
from coffee_diagnosis.rag.clarification_gen import ClarificationGenerator
from coffee_diagnosis.rag.state_manager import StateManager
from .diagnosis_generator import DiagnosisGenerator, Diagnosis
from .hallucination_checker import HallucinationChecker


class CoffeeDiagnosisController:
    def __init__(self, data_dir: str = ".", vector_db_path: str = "vector_db"):
        """
        Initialize the complete diagnosis system

        Args:
            data_dir: Directory containing PDF files
            vector_db_path: Path for FAISS vector store
        """
        print("[INIT] Initializing Coffee Disease Diagnosis System...")

        # Initialize components
        self.ambiguity_detector = AmbiguityDetector()
        self.retrieval_evaluator = RetrievalEvaluator(relevance_threshold=0.3)
        self.clarification_gen = ClarificationGenerator()
        self.state_manager = StateManager()
        self.diagnosis_gen = DiagnosisGenerator()
        self.hallucination_checker = HallucinationChecker(num_generations=3)

        # Initialize vector store
        print("[LOAD] Loading PDFs and creating vector store...")
        pdf_loader = PDFLoader(data_dir=data_dir)
        documents = pdf_loader.load_pdfs()

        self.vector_store = FAISSVectorStore(vector_db_path=vector_db_path)
        self.vector_store.create_index(documents)

        self.retriever = Retriever(self.vector_store, top_k=5)

        print("[OK] System initialized successfully!\n")

    def diagnose_web(self, initial_query: str) -> Dict:
        """
        Run diagnosis for web interface (no interactive input)

        Args:
            initial_query: User's initial symptom description

        Returns:
            Dictionary with diagnosis, verification, and state summary
        """
        print(f"\n[QUERY] Initial Query: {initial_query}\n")

        # Initialize state
        self.state_manager.initialize(initial_query)

        # Step 1: Detect ambiguity
        print("[STEP1] Detecting missing information...")
        ambiguity_result = self.ambiguity_detector.detect_ambiguity(initial_query)
        self.state_manager.update_detected_ambiguities(ambiguity_result)

        # Step 2: Retrieve relevant context
        print("[STEP2] Retrieving relevant context...")
        retrieved_docs = self.retriever.retrieve(initial_query, [])

        # Step 3: Evaluate retrieval (CRAG)
        print("[STEP3] Evaluating retrieval relevance...")
        filtered_docs = self.retrieval_evaluator.evaluate_chunks(
            retrieved_docs,
            initial_query
        )
        self.state_manager.update_context(filtered_docs)

        # Step 4: Generate final diagnosis
        print("[STEP4] Generating final diagnosis...")
        final_diagnosis = self.diagnosis_gen.generate_diagnosis(
            initial_query,
            [],  # No user responses yet
            filtered_docs,
            0.7  # Default confidence
        )

        # Step 5: Hallucination check
        print("[STEP5] Verifying diagnosis consistency...")
        verification_report = self.hallucination_checker.check_hallucination(
            initial_query,
            [],
            filtered_docs,
            final_diagnosis.disease_name
        )

        # Compile results
        results = {
            'diagnosis': final_diagnosis,
            'verification': verification_report,
            'state_summary': self.state_manager.get_state_summary(),
            'conversation_history': []
        }

        return results

    def diagnose(self, initial_query: str) -> Dict:
        """
        Run complete multi-turn diagnosis process

        Args:
            initial_query: User's initial symptom description

        Returns:
            Dictionary with diagnosis, conversation history, and verification
        """
        print(f"\n[QUERY] Initial Query: {initial_query}\n")

        # Initialize state
        self.state_manager.initialize(initial_query)

        # Main loop
        while True:
            print(f"{'='*60}")
            print(f"Turn {self.state_manager.questions_asked + 1}")
            print(f"Confidence: {self.state_manager.confidence * 100:.1f}%")
            print(f"{'='*60}\n")

            # Step 1: Detect ambiguity
            print("[STEP1] Detecting missing information...")
            ambiguity_result = self.ambiguity_detector.detect_ambiguity(
                self.state_manager.state.initial_query + " " + " ".join(
                    self.state_manager.state.user_responses
                )
            )
            missing_info = ambiguity_result['missing']
            self.state_manager.update_detected_ambiguities(ambiguity_result)

            if missing_info:
                print(f"Missing: {', '.join(missing_info.keys())}")
            else:
                print("All necessary information present")

            # Step 2: Retrieve relevant context
            print("\n[STEP2] Retrieving relevant context...")
            retrieved_docs = self.retriever.retrieve(
                self.state_manager.state.initial_query,
                self.state_manager.state.user_responses
            )

            # Step 3: Evaluate retrieval (CRAG)
            print("[STEP3] Evaluating retrieval relevance...")
            filtered_docs = self.retrieval_evaluator.evaluate_chunks(
                retrieved_docs,
                self.state_manager.state.initial_query
            )
            self.state_manager.update_context(filtered_docs)

            # Step 4: Check stop condition
            if self.state_manager.should_stop():
                print("\n[OK] Stop condition met - Proceeding to diagnosis")
                break

            # Step 5: Generate clarification question (RAC)
            if missing_info or filtered_docs:
                print("\n[STEP4] Generating clarification question...")
                question = self.clarification_gen.generate_question(
                    self.state_manager.state.initial_query,
                    filtered_docs,
                    self.state_manager.state.user_responses,
                    missing_info
                )
                print(f"\nSystem: {question}\n")
                self.state_manager.add_system_question(question)

                # Get user response
                user_response = input("You: ").strip()

                if user_response.lower() in ['quit', 'exit', 'done', 'stop']:
                    break

                self.state_manager.add_user_response(user_response)
            else:
                print("No additional information needed")
                break

        # Step 6: Generate final diagnosis
        print("\n" + "="*60)
        print("[STEP5] GENERATING FINAL DIAGNOSIS")
        print("="*60)

        final_diagnosis = self.diagnosis_gen.generate_diagnosis(
            self.state_manager.state.initial_query,
            self.state_manager.state.user_responses,
            self.state_manager.state.retrieved_context,
            self.state_manager.confidence
        )

        # Step 7: Hallucination check
        print("\n[STEP6] Verifying diagnosis consistency...")
        verification_report = self.hallucination_checker.check_hallucination(
            self.state_manager.state.initial_query,
            self.state_manager.state.user_responses,
            self.state_manager.state.retrieved_context,
            final_diagnosis.disease_name
        )

        # Compile results
        results = {
            'diagnosis': final_diagnosis,
            'verification': verification_report,
            'state_summary': self.state_manager.get_state_summary(),
            'conversation_history': self.state_manager.get_conversation_history()
        }

        return results

    def print_results(self, results: Dict) -> None:
        """Print formatted results"""
        diagnosis = results['diagnosis']
        verification = results['verification']

        print(self.diagnosis_gen.format_diagnosis_output(diagnosis))
        print(self.hallucination_checker.format_verification_report(verification))

        if not verification['consistent']:
            print("\n[WARNING] Diagnosis may contain hallucinated information. "
                  "Consult local experts for confirmation.")

    def reset(self) -> None:
        """Reset system for new diagnosis"""
        self.state_manager.reset()

    def start_diagnosis(self, initial_query: str) -> Dict:
        """
        Start multi-turn diagnosis for web interface

        Args:
            initial_query: User's initial symptom description

        Returns:
            Dictionary with 'status' ('question' or 'diagnosis') and question/diagnosis data
        """
        print(f"\n[QUERY] Initial Query: {initial_query}\n")

        # Initialize state
        self.state_manager.initialize(initial_query)

        # Step 1: Detect ambiguity
        print("[STEP1] Detecting missing information...")
        ambiguity_result = self.ambiguity_detector.detect_ambiguity(initial_query)
        missing_info = ambiguity_result['missing']
        self.state_manager.update_detected_ambiguities(ambiguity_result)

        # Step 2: Retrieve relevant context
        print("[STEP2] Retrieving relevant context...")
        retrieved_docs = self.retriever.retrieve(initial_query, [])

        # Step 3: Evaluate retrieval (CRAG)
        print("[STEP3] Evaluating retrieval relevance...")
        filtered_docs = self.retrieval_evaluator.evaluate_chunks(
            retrieved_docs,
            initial_query
        )
        self.state_manager.update_context(filtered_docs)

        # Step 4: Check if we need clarifications
        print("[STEP4] Generating clarification question...")
        if missing_info and filtered_docs:
            question = self.clarification_gen.generate_question(
                initial_query,
                filtered_docs,
                [],
                missing_info
            )
            self.state_manager.add_system_question(question)
            return {
                'status': 'question',
                'question': question
            }
        else:
            # No clarifications needed, go to diagnosis
            return self._generate_final_diagnosis(initial_query, [])

    def submit_answer(self, answer: str) -> Dict:
        """
        Submit user answer and continue multi-turn process

        Args:
            answer: User's answer to the clarification question

        Returns:
            Dictionary with 'status' ('question' or 'diagnosis') and next question/diagnosis
        """
        print(f"\n[USER_ANSWER] {answer}\n")

        # Add user response to state
        self.state_manager.add_user_response(answer)

        # Check stop condition
        if self.state_manager.should_stop():
            print("[OK] Stop condition met")
            return self._generate_final_diagnosis(
                self.state_manager.state.initial_query,
                self.state_manager.state.user_responses
            )

        # Detect remaining ambiguities with updated information
        print("[STEP4] Generating next clarification question...")
        full_context = (
            self.state_manager.state.initial_query + " " +
            " ".join(self.state_manager.state.user_responses)
        )
        ambiguity_result = self.ambiguity_detector.detect_ambiguity(full_context)
        missing_info = ambiguity_result['missing']

        # Retrieve updated context
        retrieved_docs = self.retriever.retrieve(
            self.state_manager.state.initial_query,
            self.state_manager.state.user_responses
        )

        filtered_docs = self.retrieval_evaluator.evaluate_chunks(
            retrieved_docs,
            self.state_manager.state.initial_query
        )
        self.state_manager.update_context(filtered_docs)

        # Ask next question if ambiguities remain
        if missing_info and filtered_docs:
            question = self.clarification_gen.generate_question(
                self.state_manager.state.initial_query,
                filtered_docs,
                self.state_manager.state.user_responses,
                missing_info
            )
            self.state_manager.add_system_question(question)
            return {
                'status': 'question',
                'question': question
            }
        else:
            # All info gathered or max questions reached
            return self._generate_final_diagnosis(
                self.state_manager.state.initial_query,
                self.state_manager.state.user_responses
            )

    def _generate_final_diagnosis(self, initial_query: str, user_responses: List[str]) -> Dict:
        """
        Generate final diagnosis with hallucination verification

        Args:
            initial_query: Initial user query
            user_responses: List of user responses to clarifications

        Returns:
            Dictionary with 'status', diagnosis, and verification
        """
        print("\n[STEP5] GENERATING FINAL DIAGNOSIS")

        final_diagnosis = self.diagnosis_gen.generate_diagnosis(
            initial_query,
            user_responses,
            self.state_manager.state.retrieved_context,
            self.state_manager.confidence
        )

        print("\n[STEP6] Verifying diagnosis consistency...")
        verification_report = self.hallucination_checker.check_hallucination(
            initial_query,
            user_responses,
            self.state_manager.state.retrieved_context,
            final_diagnosis.disease_name
        )

        return {
            'status': 'diagnosis',
            'diagnosis': final_diagnosis,
            'verification': verification_report,
            'state_summary': self.state_manager.get_state_summary(),
            'conversation_history': self._get_conversation_history()
        }

    def _get_conversation_history(self) -> str:
        """
        Get formatted conversation history

        Returns:
            Formatted conversation history string
        """
        return self.state_manager.get_conversation_history()
