from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
# from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.document_loaders import UnstructuredMarkdownLoader, DirectoryLoader
# import langchain
# import inspect
# print(inspect.getfile(MarkdownHeaderTextSplitter))
from lippy.utils.obsidian_loader import ObsidianLoader

# Injest documents
# Provide retriever for use in RetrievalQA chain
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
]

class db:
    def __init__(self,chunkSize=500):
        self.pathDB = "/home/ubuntu/Tehas/lippy/data/db"
        self.embeddings = HuggingFaceInstructEmbeddings()
        self.vectordb = None

    def injest(self, path="/home/ubuntu/Tehas/lippy/data/vault/2 - Notes"):
        self.pathData = path
        self.loader = ObsidianLoader(path)
        docs = self.loader.load()
        self.vectordb = Chroma.from_documents(documents=docs, embedding=self.embeddings, persist_directory=self.pathDB)

    def retriever(self):
        self.injest()
        return self.vectordb.as_retriever() if self.vectordb is not None else None

if __name__ == '__main__':
    # from falcon_llm import FalconLLM
    # llm = FalconLLM()
    # db = db()
    loader = ObsidianLoader("/home/ubuntu/Tehas/lippy/data/vault/2 - Notes")
    docs = loader.load()
    
    # print(inspect.getsource(ObsidianLoader))
    
    # texts = db.splitter.split_texts(docs)
    print(len(docs))
    print(docs[1])