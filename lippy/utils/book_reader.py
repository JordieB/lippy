from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from nltk import sent_tokenize
import numpy as numpy
from IPython.display import Audio
from lippy.utils.speaker import Speaker
from pathlib import Path

PROJ_DIR = Path(__file__).resolve().parents[2]
pathLib = PROJ_DIR / Path("data/books/")
bookName = "Vagabonding - Rolf Potts"
pathBook = pathLib / Path(bookName + ".epub")
GEN_TEMP = 0.6
book = epub.read_epub(str(pathBook))
items = list(book.get_items_of_type(ITEM_DOCUMENT))
collection = []
speaker = Speaker()
for i, item in enumerate(items[7:]):
    soup = BeautifulSoup(item.get_body_content(), 'html.parser')
    text = [para.get_text() for para in soup.find_all('p')]
    text = ' '.join(text)
    if len(text) == 0:
        continue
    fn =  f"Vagabond_{i}"
    speaker.say(text, fn)
    collection.append(text)
    print(text)