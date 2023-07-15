from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from lippy.utils.speaker import Speaker

pathLib = "/home/theatasigma/lippy/data/books/"
bookName = "Vagabonding - Rolf Potts"
pathBook = pathLib + bookName + ".epub"
book = epub.read_epub(pathBook)
items = list(book.get_items_of_type(ITEM_DOCUMENT))
collection = []
# print([item.get_name() for item in items])
for i, item in enumerate(items[6:]):
    soup = BeautifulSoup(item.get_body_content(), 'html.parser')
    # print(soup.prettify())
    text = [para.get_text() for para in soup.find_all('p')]
    # print("-----")
    # print(' '.join(text))
    collection.append(' '.join(text))
    if i > 5:
        break

voice = Speaker("/home/theatasigma/lippy/data/audio")
print(collection[1])
voice.say(collection[1])