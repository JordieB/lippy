from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from lippy.utils.obsidian_loader import ObsidianLoader
from pathlib import Path

PROJ_DIR = Path(__file__).resolve().parents[2]

class Database:
    """
    A class to represent a database for document retrieval.

    Attributes:
        pathDB (str): Path to the database.
        embeddings (HuggingFaceInstructEmbeddings): Embeddings to use for the
            documents.
        vectordb (Chroma): Vector database for the documents.
    """

    def __init__(self, chunk_size=500):
        """
        Initializes the Database object.

        Args:
            chunk_size (int, optional): Chunk size for the documents. Defaults
                to 500.
        """
        self.pathDB = str(PROJ_DIR / "data/db")
        self.embeddings = HuggingFaceInstructEmbeddings()
        self.vectordb = None

    def injest(self, path=str(PROJ_DIR / "data/vault/2 - Notes")):
        """
        Ingests the documents from the specified path into the vector database.

        Args:
            path (str, optional): Path to the documents. Defaults to
                'data/vault/2 - Notes'.
        """
        self.loader = ObsidianLoader(path)
        docs = self.loader.load()
        self.vectordb = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.pathDB
        )

    def retriever(self):
        """
        Returns a retriever for the vector database.

        Returns:
            Retriever: The retriever for the vector database, or None if the
                database is not initialized.
        """
        self.injest()
        return (
            self.vectordb.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.5, "k": 5},
            )
            if self.vectordb is not None
            else None
        )


if __name__ == "__main__":
    db = Database()
    db.injest()
    docs = db.vectordb.similarity_search_with_score(
        "what are the effects of DO on water?", k=5
    )
    print(db.vectordb._collection)
