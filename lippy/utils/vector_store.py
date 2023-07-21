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
        return self.vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5, "k":5}) if self.vectordb is not None else None

if __name__ == '__main__':
    # from falcon_llm import FalconLLM
    # llm = FalconLLM()
    # loader = ObsidianLoader("/home/theatasigma/lippy/data/vault/2 - Notes")
    # docs = loader.load()
    db = db()
    db.pathDB="/home/theatasigma/lippy/data/db" 
    db.injest("/home/theatasigma/lippy/data/vault/2 - Notes")
    # ret = db.vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5, "k":5})
    # docs = ret.get_relevant_documents("what are the effects of DO on water?")
    docs = db.vectordb.similarity_search_with_score("what are the effects of DO on water?", k=5)
    
    # print(inspect.getsource(ObsidianLoader))
    
    # texts = db.splitter.split_texts(docs)
    print(db.vectordb._collection_)
    # for i in range(len(docs)):
    #     print("-----")
    #     print(docs[i])