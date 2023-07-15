import re
from langchain.document_loaders.base import BaseLoader
from pathlib import Path, PosixPath
from typing import List
from langchain.text_splitter import LineType
from langchain.docstore.document import Document
import pdb

class ObsidianLoader(BaseLoader):
    """Loader that loads Obsidian files from disk."""

    FRONT_MATTER_REGEX = re.compile(r"^---\n(.*?)\n---\n", re.MULTILINE | re.DOTALL)

    def __init__(
        self,
        path: str,
        headers_to_split_on: str = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5")
        ],
        encoding: str = "UTF-8",
        collect_metadata: bool = True,
        return_each_line: bool = False
    ):
        """Initialize with path."""
        self.file_path = path
        self.encoding = encoding
        self.collect_metadata = collect_metadata
        self.headers_to_split_on = sorted(
            headers_to_split_on, key=lambda split: len(split[0]), reverse=True
        )
        self.return_each_line = return_each_line

    def _parse_front_matter(self, content: str) -> dict:
        """Parse front matter metadata from the content and return it as a dict."""
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
        """Remove front matter metadata from the given content."""
        if not self.collect_metadata:
            return content
        return self.FRONT_MATTER_REGEX.sub("", content)

    def aggregate_lines_to_chunks(self, lines: List[LineType]) -> List[Document]:
        """Combine lines with common metadata into chunks
        Args:
            lines: Line of text / associated header metadata
        """
        aggregated_chunks: List[LineType] = []

        for line in lines:
            if (
                aggregated_chunks
                and aggregated_chunks[-1]["metadata"] == line["metadata"]
            ):
                # If the last line in the aggregated list
                # has the same metadata as the current line,
                # append the current content to the last lines's content
                aggregated_chunks[-1]["content"] += "  \n" + line["content"]
            else:
                # Otherwise, append the current line to the aggregated list
                aggregated_chunks.append(line)

        return [
            Document(page_content=chunk["content"], metadata=chunk["metadata"])
            for chunk in aggregated_chunks
        ]

    def split_text(self, text: str, pathObj: PosixPath) -> List[Document]:
        """Split markdown file
        Args:
            text: Markdown file"""

        # Split the input text by newline character ("\n").
        lines = text.split("\n")
        # Final output
        lines_with_metadata: List[LineType] = []
        # Content and metadata of the chunk currently being processed
        current_content: List[str] = []
        current_metadata: Dict[str, str] = {}
        # Keep track of the nested header structure
        # header_stack: List[Dict[str, Union[int, str]]] = []
        header_stack: List[HeaderType] = []
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
            # stripped_line = line.strip()
            stripped_line = line            # Maintain heirarchical structure
            # Check each line against each of the header types (e.g., #, ##)
            for sep, name in self.headers_to_split_on:
                # Check if line starts with a header that we intend to split on
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
                            and header_stack[-1]["level"] >= current_header_level
                        ):
                            # We have encountered a new header
                            # at the same or higher level
                            popped_header = header_stack.pop()
                            # Clear the metadata for the
                            # popped header in initial_metadata
                            if popped_header["name"] in initial_metadata:
                                initial_metadata.pop(popped_header["name"])

                        # Push the current header to the stack
                        header: HeaderType = {
                            "level": current_header_level,
                            "name": name,
                            "data": stripped_line[len(sep) :].strip(),
                        }
                        header_stack.append(header)
                        # Update initial_metadata with the current header
                        initial_metadata[name] = header["data"]

                    # Add the previous line to the lines_with_metadata
                    # only if current_content is not empty
                    if current_content:
                        lines_with_metadata.append(
                            {
                                "content": ' > '.join([header["data"] for header in header_stack]) + '\n' +  "\n".join(current_content),
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
                            "content": ' > '.join([header["data"] for header in header_stack]) + '\n' + "\n".join(current_content),
                            "metadata": current_metadata.copy(),
                        }
                    )
                    current_content.clear()

            current_metadata = initial_metadata.copy()

        if current_content:
            lines_with_metadata.append(
                {"content": ' > '.join([header["data"] for header in header_stack]) + '\n' + "\n".join(current_content), "metadata": current_metadata}
            )

        # lines_with_metadata has each line with associated header metadata
        # aggregate these into chunks based on common metadata
        if not self.return_each_line:
            return self.aggregate_lines_to_chunks(lines_with_metadata)
        else:
            return [
                Document(page_content=chunk["content"], metadata=chunk["metadata"])
                for chunk in lines_with_metadata
            ]

    def load(self) -> List[Document]:
        """Load documents."""
        ps = list(Path(self.file_path).glob("**/*.md"))
        docs = []
        for p in ps:
            with open(p, encoding=self.encoding) as f:
                text = f.read()

            pageDocs = self.split_text(text, p)
            docs.append(pageDocs)
        docs = [ele for inner_list in docs for ele in inner_list]
        return docs



# class MarkdownHeaderTextSplitter:
#     """Implementation of splitting markdown files based on specified headers."""

#     def __init__(
#         self, headers_to_split_on: List[Tuple[str, str]], return_each_line: bool = False
#     ):
#         """Create a new MarkdownHeaderTextSplitter.

#         Args:
#             headers_to_split_on: Headers we want to track
#             return_each_line: Return each line w/ associated headers
#         """
#         # Output line-by-line or aggregated into chunks w/ common headers
#         self.return_each_line = return_each_line
#         # Given the headers we want to split on,
#         # (e.g., "#, ##, etc") order by length
#         self.headers_to_split_on = sorted(
#             headers_to_split_on, key=lambda split: len(split[0]), reverse=True
#         )

#     def aggregate_lines_to_chunks(self, lines: List[LineType]) -> List[Document]:
#         """Combine lines with common metadata into chunks
#         Args:
#             lines: Line of text / associated header metadata
#         """
#         aggregated_chunks: List[LineType] = []

#         for line in lines:
#             if (
#                 aggregated_chunks
#                 and aggregated_chunks[-1]["metadata"] == line["metadata"]
#             ):
#                 # If the last line in the aggregated list
#                 # has the same metadata as the current line,
#                 # append the current content to the last lines's content
#                 aggregated_chunks[-1]["content"] += "  \n" + line["content"]
#             else:
#                 # Otherwise, append the current line to the aggregated list
#                 aggregated_chunks.append(line)

#         return [
#             Document(page_content=chunk["content"], metadata=chunk["metadata"])
#             for chunk in aggregated_chunks
#         ]

#     def split_text(self, text: str) -> List[Document]:
#         """Split markdown file
#         Args:
#             text: Markdown file"""

#         # Split the input text by newline character ("\n").
#         lines = text.split("\n")
#         # Final output
#         lines_with_metadata: List[LineType] = []
#         # Content and metadata of the chunk currently being processed
#         current_content: List[str] = []
#         current_metadata: Dict[str, str] = {}
#         # Keep track of the nested header structure
#         # header_stack: List[Dict[str, Union[int, str]]] = []
#         header_stack: List[HeaderType] = []
#         initial_metadata: Dict[str, str] = {}

#         for line in lines:
#             stripped_line = line.strip()
#             # Check each line against each of the header types (e.g., #, ##)
#             for sep, name in self.headers_to_split_on:
#                 # Check if line starts with a header that we intend to split on
#                 if stripped_line.startswith(sep) and (
#                     # Header with no text OR header is followed by space
#                     # Both are valid conditions that sep is being used a header
#                     len(stripped_line) == len(sep)
#                     or stripped_line[len(sep)] == " "
#                 ):
#                     # Ensure we are tracking the header as metadata
#                     if name is not None:
#                         # Get the current header level
#                         current_header_level = sepathObj.count("#")

#                         # Pop out headers of lower or same level from the stack
#                         while (
#                             header_stack
#                             and header_stack[-1]["level"] >= current_header_level
#                         ):
#                             # We have encountered a new header
#                             # at the same or higher level
#                             popped_header = header_stack.pop()
#                             # Clear the metadata for the
#                             # popped header in initial_metadata
#                             if popped_header["name"] in initial_metadata:
#                                 initial_metadata.pop(popped_header["name"])

#                         # Push the current header to the stack
#                         header: HeaderType = {
#                             "level": current_header_level,
#                             "name": name,
#                             "data": stripped_line[len(sep) :].strip(),
#                         }
#                         header_stack.append(header)
#                         # Update initial_metadata with the current header
#                         initial_metadata[name] = header["data"]

#                     # Add the previous line to the lines_with_metadata
#                     # only if current_content is not empty
#                     if current_content:
#                         lines_with_metadata.append(
#                             {
#                                 "content": "\n".join(current_content),
#                                 "metadata": current_metadata.copy(),
#                             }
#                         )
#                         current_content.clear()

#                     break
#             else:
#                 if stripped_line:
#                     current_content.append(stripped_line)
#                 elif current_content:
#                     lines_with_metadata.append(
#                         {
#                             "content": "\n".join(current_content),
#                             "metadata": current_metadata.copy(),
#                         }
#                     )
#                     current_content.clear()

#             current_metadata = initial_metadata.copy()

#         if current_content:
#             lines_with_metadata.append(
#                 {"content": "\n".join(current_content), "metadata": current_metadata}
#             )

#         # lines_with_metadata has each line with associated header metadata
#         # aggregate these into chunks based on common metadata
#         if not self.return_each_line:
#             return self.aggregate_lines_to_chunks(lines_with_metadata)
#         else:
#             return [
#                 Document(page_content=chunk["content"], metadata=chunk["metadata"])
#                 for chunk in lines_with_metadata
#             ]

if __name__ == "__main__":
    loader = ObsidianLoader("/home/theatasigma/lippy/data/vault/2 - Notes")
    docs = loader.load()