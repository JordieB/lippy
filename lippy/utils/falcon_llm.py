from typing import Any, List
from pathlib import Path
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from transformers import AutoTokenizer, pipeline
from torch import bfloat16
from pydantic import Field
from langchain.chains import RetrievalQA

from lippy.utils.vector_store import db
from lippy.utils.speaker import Speaker

# Define the project directory
PROJ_DIR = Path(__file__).resolve().parents[2]

class FalconLLM(LLM):
    """
    A class representing a custom Falcon language model.

    Attributes:
        falcon_model_dir (str): The directory of the Falcon model.
        falcon_tokenizer (AutoTokenizer): The tokenizer for the Falcon model.
        falcon_pipeline (Any): The pipeline for the Falcon model.
        db (db): The database for the Falcon model.
        endpoint (Any): The endpoint for the Falcon model.
        voice (Speaker): The speaker for the Falcon model.
    """

    # Define the default directory for the Falcon model
    falcon_model_dir: str = Field(default="tiiuae/falcon-7b-instruct")
    falcon_tokenizer: AutoTokenizer = None
    falcon_pipeline: Any = None
    db: db = None
    endpoint: Any = None
    voice: Speaker = None

    def __init__(self):
        """
        Initializes the FalconLLM class.
        """
        super().__init__()  # Call LLM's __init__ method
        # Load the tokenizer from the Falcon model directory
        self.falcon_tokenizer = AutoTokenizer.from_pretrained(
            self.falcon_model_dir
        )
        # Create a pipeline for text generation using the Falcon model
        self.falcon_pipeline = pipeline(
            "text-generation",
            model=self.falcon_model_dir,
            tokenizer=self.falcon_tokenizer,
            torch_dtype=bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        # Initialize the database
        self.db = db()
        # Create an endpoint for the Falcon model
        self.endpoint = RetrievalQA.from_chain_type(
            llm=self, chain_type="stuff", retriever=self.db.retriever()
        )
        # Initialize the speaker and load the model
        self.voice = Speaker(str(PROJ_DIR / "data/audio"))
        self.voice.load_model()

    @property
    def _llm_type(self) -> str:
        """
        Get the type of the LLM.

        Returns:
            str: "custom" as the type of this LLM.
        """
        return "custom"

    def _call(
        self,
        prompt: str,
        speak: bool = True,
        stop: List[str] = None,
        run_manager: CallbackManagerForLLMRun = None,
    ) -> str:
        """
        Call the LLM with provided parameters.

        Args:
            prompt (str): The input for the LLM.
            speak (bool): Whether to speak the output. Defaults to True.
            stop (List[str]): Not permitted, raises ValueError if provided.
            run_manager (CallbackManagerForLLMRun, optional): A callback
                manager.

        Returns:
            str: The result of the LLM operation.

        Raises:
            ValueError: If the stop argument is provided.
        """
        # Raise an error if the stop argument is provided
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        # Generate sequences using the Falcon pipeline
        sequences = self.falcon_pipeline(
            prompt,
            max_length=8000,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.falcon_tokenizer.eos_token_id,
        )
        # Join the generated sequences into a response
        resp = ", ".join([seq["generated_text"] for seq in sequences])
        # Extract the answer from the response
        answer = resp.split("Helpful Answer:")[1]
        print(answer)
        # Use the speaker to say the answer
        self.voice.say(answer)
        return answer


if __name__ == "__main__":
    # Initialize the FalconLLM
    llm = FalconLLM()
    # Run the endpoint with a prompt and print the result
    print(
        llm.endpoint.run(
            ("What impacts the pH of a sample? Please provide a detailed "
             "answer.")
        )
    )
