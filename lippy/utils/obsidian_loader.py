import re
from typing import List, Dict, Tuple, Union
from pathlib import Path, PosixPath
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import LineType
from langchain.docstore.document import Document


class ObsidianLoader(BaseLoader):
    """
    A class to load Obsidian files from disk.

    Attributes:
        FRONT_MATTER_REGEX (re.Pattern): Regular expression pattern to match
            front matter.
        file_path (str): Path to the Obsidian files.
        encoding (str): Encoding of the files.
        collect_metadata (bool): Whether to collect metadata from the files.
        headers_to_split_on (List[Tuple[str, str]]): Headers to split the files
            on.
        return_each_line (bool): Whether to return each line of the files.
    """

    FRONT_MATTER_REGEX = re.compile(
        r"^---\n(.*?)\n---\n", re.MULTILINE | re.DOTALL
    )

    def __init__(
        self,
        path: str,
        headers_to_split_on: List[Tuple[str, str]] = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
        ],
        encoding: str = "UTF-8",
        collect_metadata: bool = True,
        return_each_line: bool = False,
    ):
        """
        Initializes the ObsidianLoader object.

        Args:
            path (str): Path to the Obsidian files.
            headers_to_split_on (List[Tuple[str, str]], optional): Headers to
                split the files on. Defaults to headers 1 to 5.
            encoding (str, optional): Encoding of the files. Defaults to "UTF-8".
            collect_metadata (bool, optional): Whether to collect metadata from
                the files. Defaults to True.
            return_each_line (bool, optional): Whether to return each line of
                the files. Defaults to False.
        """
        self.file_path = path
        self.encoding = encoding
        self.collect_metadata = collect_metadata
        self.headers_to_split_on = sorted(
            headers_to_split_on,
            key=lambda split: len(split[0]),
            reverse=True,
        )
        self.return_each_line = return_each_line

    def _parse_front_matter(self, content: str) -> Dict[str, str]:
        """
        Parses front matter metadata from the content and returns it as a
        dictionary.

        Args:
            content (str): Content to parse front matter from.

        Returns:
            Dict[str, str]: Dictionary containing front matter metadata.
        """
        if not self.collect_metadata:
            return {}
        match = self.FRONT_MATTER_REGEX.search(content)
        front_matter = {}
        if match:
            lines = match.group(1).split("\n")
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    front_matter[key.strip()] = value.strip()
                else:
                    # Skip lines without a colon
                    continue
        return front_matter

    def _remove_front_matter(self, content: str) -> str:
        """
        Removes front matter metadata from the content.

        Args:
            content (str): Content to remove front matter from.

        Returns:
            str: Content without front matter.
        """
        return (
            self.FRONT_MATTER_REGEX.sub("", content)
            if self.collect_metadata
            else content
        )

    def aggregate_lines_to_chunks(
        self, lines: List[LineType]
    ) -> List[Document]:
        """
        Aggregates lines with common metadata into chunks.

        Args:
            lines (List[LineType]): Lines to aggregate.

        Returns:
            List[Document]: Aggregated lines as documents.
        """
        aggregated_chunks: List[LineType] = []
        for line in lines:
            if (
                aggregated_chunks
                and aggregated_chunks[-1]["metadata"]
                == line["metadata"]
            ):
                # If the last line in the aggregated list has the same metadata
                # as the current line, append the current content to the last
                # lines's content
                aggregated_chunks[-1]["content"] += "  \n" + line["content"]
            else:
                # Otherwise, append the current line to the aggregated list
                aggregated_chunks.append(line)
        return [
            Document(
                page_content=chunk["content"], metadata=chunk["metadata"]
            )
            for chunk in aggregated_chunks
        ]

    def split_text(self, text: str, pathObj: PosixPath) -> List[Document]:
        """
        Splits the text of a markdown file.

        Args:
            text (str): Text to split.
            pathObj (PosixPath): Path to the file.

        Returns:
            List[Document]: List of documents resulting from the split.
        """
        # Split the input text by newline character ("\n").
        lines = text.split("\n")
        # Final output
        lines_with_metadata: List[LineType] = []
        # Content and metadata of the chunk currently being processed
        current_content: List[str] = []
        current_metadata: Dict[str, str] = {}
        # Keep track of the nested header structure
        # header_stack: List[Dict[str, Union[int, str]]] = []
        header_stack: List[Dict[str, Union[int, str]]] = []
        front_matter = self._parse_front_matter(text)
        initial_metadata: Dict[str, str] = {
            "source": str(pathObj.name),
            "path": str(pathObj),
            "created": pathObj.stat().st_ctime,
            "last_modified": pathObj.stat().st_mtime,
            "last_accessed": pathObj.stat().st_atime,
            **front_matter,
        }

        for line in lines:
            # Maintain heirarchical structure
            stripped_line = line
            for sep, name in self.headers_to_split_on:
                # Check each line against each of the header types (e.g., #, ##)
                if stripped_line.startswith(sep) and (
                    # Header with no text OR header is followed by space
                    # Both are valid conditions that sep is being used a header
                    len(stripped_line) == len(sep)
                    or stripped_line[len(sep)] == " "
                ):
                    # Ensure we are tracking the header as metadata
                    if name is not None:
                        # Get the current header level
                        current_header_level = sep.count("#")
                        # Pop out headers of lower or same level from the stack
                        while (
                            header_stack
                            and header_stack[-1]["level"]
                            >= current_header_level
                        ):
                            # We have encountered a new header at the same or
                            # higher level
                            popped_header = header_stack.pop()
                            # Clear the metadata for the popped header in
                            # initial_metadata
                            if popped_header["name"] in initial_metadata:
                                initial_metadata.pop(popped_header["name"])
                        # Push the current header to the stack
                        header: Dict[str, Union[int, str]] = {
                            "level": current_header_level,
                            "name": name,
                            "data": stripped_line[len(sep) :].strip(),
                        }
                        header_stack.append(header)
                        # Update initial_metadata with the current header
                        initial_metadata[name] = header["data"]
                    # Add the previous line to the lines_with_metadata only if
                    # current_content is not empty
                    if current_content:
                        lines_with_metadata.append(
                            {
                                "content": " > ".join(
                                    [header["data"] for header in header_stack]
                                )
                                + "\n"
                                + "\n".join(current_content),
                                "metadata": current_metadata.copy(),
                            }
                        )
                        current_content.clear()
                    break
            else:
                if stripped_line:
                    current_content.append(stripped_line)
                elif current_content:
                    lines_with_metadata.append(
                        {
                            "content": " > ".join(
                                [header["data"] for header in header_stack]
                            )
                            + "\n"
                            + "\n".join(current_content),
                            "metadata": current_metadata.copy(),
                        }
                    )
                    current_content.clear()
            current_metadata = initial_metadata.copy()

        if current_content:
            lines_with_metadata.append(
                {
                    "content": " > ".join(
                        [header["data"] for header in header_stack]
                    )
                    + "\n"
                    + "\n".join(current_content),
                    "metadata": current_metadata,
                }
            )

        # lines_with_metadata has each line with associated header metadata
        # aggregate these into chunks based on common metadata
        return (
            self.aggregate_lines_to_chunks(lines_with_metadata)
            if not self.return_each_line
            else [
                Document(
                    page_content=chunk["content"], metadata=chunk["metadata"]
                )
                for chunk in lines_with_metadata
            ]
        )

    def load(self) -> List[Document]:
        """
        Loads documents from the Obsidian files.

        Returns:
            List[Document]: List of documents loaded from the files.
        """
        ps = list(Path(self.file_path).glob("**/*.md"))
        docs = []
        for p in ps:
            with open(p, encoding=self.encoding) as f:
                text = f.read()
            pageDocs = self.split_text(text, p)
            docs.extend(pageDocs)
        return docs


if __name__ == "__main__":
    # Define the project directory
    PROJ_DIR = Path(__file__).resolve().parents[2]
    # Init the Obsidian markdown/doc loader
    loader = ObsidianLoader(str(PROJ_DIR / "data/vault/2 - Notes"))
    # Kick off the loading process
    docs = loader.load()
