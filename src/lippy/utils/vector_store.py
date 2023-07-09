from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import DirectoryLoader
# from falcon_llm import FalconLLM
# from langchain.chains import RetrievalQA

# Injest documents
# Provide retriever for use in RetrievalQA chain

class db:
    def __init__(self,chunkSize=500):
        self.pathDB = "/home/ubuntu/Tehas/lippy/data/db"
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunkSize, chunk_overlap=0)
        self.embeddings = HuggingFaceInstructEmbeddings()
        self.vectordb = None

    def injest(self, path="/home/ubuntu/Tehas/lippy/data/vault/2 - Notes", pattern='**/*.md'):
        self.pathData = path
        self.loader = DirectoryLoader(path, glob=pattern, loader_cls=UnstructuredMarkdownLoader)
        docs = self.loader.load()
        texts = self.splitter.split_documents(docs)
        self.vectordb = Chroma.from_documents(documents=texts, embedding=self.embeddings, persist_directory=self.pathDB)

    def retriever(self):
        self.injest()
        return self.vectordb.as_retriever() if self.vectordb is not None else None

# if __name__ == '__main__':
#     llm = FalconLLM()
#     db = db()
#     qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.retriever())
#     print(qa.run("what is the procedure for measuring MLVSS"))