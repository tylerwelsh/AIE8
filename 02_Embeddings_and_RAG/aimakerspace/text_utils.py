import os
from typing import List
try:
    import pypdf
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt file."
            )

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    with open(
                        os.path.join(root, file), "r", encoding=self.encoding
                    ) as f:
                        self.documents.append(f.read())

    def load_documents(self):
        self.load()
        return self.documents


class PDFFileLoader:
    def __init__(self, path: str):
        self.documents = []
        self.path = path
        
    def load(self):
        if not PDF_AVAILABLE:
            raise ImportError("pypdf is required to read PDF files. Install it with: pip install pypdf")
            
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".pdf"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .pdf file."
            )

    def load_file(self):
        try:
            with open(self.path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                
                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    raise ValueError(f"PDF file {self.path} is password-protected and cannot be read.")
                
                # Extract text from all pages
                full_text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        full_text += page_text + "\n\n"
                
                if full_text.strip():
                    self.documents.append(full_text.strip())
                else:
                    raise ValueError(f"No readable text found in PDF file {self.path}")
                    
        except Exception as e:
            raise ValueError(f"Error reading PDF file {self.path}: {str(e)}")

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "rb") as f:
                            pdf_reader = pypdf.PdfReader(f)
                            
                            # Skip encrypted PDFs
                            if pdf_reader.is_encrypted:
                                print(f"Skipping password-protected PDF: {file_path}")
                                continue
                            
                            # Extract text from all pages
                            full_text = ""
                            for page_num in range(len(pdf_reader.pages)):
                                page = pdf_reader.pages[page_num]
                                page_text = page.extract_text()
                                if page_text.strip():
                                    full_text += page_text + "\n\n"
                            
                            if full_text.strip():
                                self.documents.append(full_text.strip())
                                
                    except Exception as e:
                        print(f"Error reading PDF file {file_path}: {str(e)}")
                        continue

    def load_documents(self):
        self.load()
        return self.documents


class DocumentLoader:
    """Unified loader that can handle both .txt and .pdf files"""
    
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.path = path
        self.encoding = encoding
        self.documents = []
    
    def load_documents(self):
        if os.path.isfile(self.path):
            if self.path.endswith(".txt"):
                loader = TextFileLoader(self.path, self.encoding)
            elif self.path.endswith(".pdf"):
                loader = PDFFileLoader(self.path)
            else:
                raise ValueError(f"Unsupported file type: {self.path}. Only .txt and .pdf files are supported.")
        elif os.path.isdir(self.path):
            # Load both txt and pdf files from directory
            txt_loader = TextFileLoader(self.path, self.encoding)
            pdf_loader = PDFFileLoader(self.path)
            
            documents = []
            try:
                documents.extend(txt_loader.load_documents())
            except ValueError:
                pass  # No txt files found
                
            try:
                documents.extend(pdf_loader.load_documents())
            except ValueError:
                pass  # No pdf files found
                
            if not documents:
                raise ValueError(f"No readable .txt or .pdf files found in directory: {self.path}")
            
            return documents
        else:
            raise ValueError(f"Path does not exist: {self.path}")
            
        return loader.load_documents()


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


if __name__ == "__main__":
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])
