from typing import Any, List, Mapping
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from transformers import AutoTokenizer, pipeline
from torch import bfloat16
from pydantic import Field
from langchain.chains import RetrievalQA

from lippy.utils.vector_store import db
from lippy.utils.speaker import Speaker

class FalconLLM(LLM):
    """
    A class representing a custom Falcon language model.

    Attributes:
        n (int): A parameter for the LLM operation.
    """
    falcon_model_dir: str = Field(default="tiiuae/falcon-7b-instruct")
    falcon_tokenizer: AutoTokenizer = None 
    falcon_pipeline: Any = None  # TODO: find the proper type for this
    db: db = None
    endpoint: Any = None
    voice: Speaker = None

    def __init__(self):
        super().__init__()  # Call LLM's __init__ method
        self.falcon_tokenizer = AutoTokenizer.from_pretrained(self.falcon_model_dir)
        self.falcon_pipeline = pipeline(
            "text-generation",
            model=self.falcon_model_dir,
            tokenizer=self.falcon_tokenizer,
            torch_dtype=bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        self.db = db()
        self.endpoint = RetrievalQA.from_chain_type(llm=self, chain_type="stuff", retriever=self.db.retriever())
        self.voice = Speaker("/home/ubuntu/Tehas/lippy/data/audio")
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
            stop: List[str] = None,
            run_manager: CallbackManagerForLLMRun = None,
    ) -> str:
        """
        Call the LLM with provided parameters.

        Args:
            prompt (str): The input for the LLM.
            stop (List[str]): Not permitted, raises ValueError if provided.
            run_manager (CallbackManagerForLLMRun, optional): A callback
                manager.

        Returns:
            str: The result of the LLM operation.

        Raises:
            ValueError: If the stop argument is provided.
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        sequences = self.falcon_pipeline(
            prompt,
            max_length=8000,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.falcon_tokenizer.eos_token_id
        )
        answer = ", ".join([seq['generated_text'] for seq in sequences])
        op = self.voice.say(answer.split('Helpful Answer:')[1])
        print(op)
        self.voice.save_audio(op, "output.wav",overwrite=True)
        return answer


    # @property
    # def _identifying_params(self) -> Mapping[str, Any]:
    #     """
    #     Get the identifying parameters of the LLM.

    #     Returns:
    #         Mapping[str, Any]: Dictionary with identifying parameters.
    #     """
    #     return {"n": self.n}
    

if __name__ == '__main__':
    llm = FalconLLM()
    print(llm.endpoint.run(("What impacts the pH of a sample? Please provide a detailed answer.")))
    