import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import shutil
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import threading
import queue
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import logging
import traceback
import asyncio
import aiofiles
from tqdm import tqdm

# Utility Classes

# Wrapper for embedding models
class EmbeddingsWrapper:
    """Wrapper class to ensure compatibility between different embedding models."""
    
    def __init__(self, embedding_model):
        """Initialize with an embedding model."""
        self.model = embedding_model
        
        # Determine the type of embeddings model
        self.is_huggingface = hasattr(embedding_model, 'embed_query')
        self.is_sentence_transformer = hasattr(embedding_model, 'encode')
        
        if not (self.is_huggingface or self.is_sentence_transformer):
            raise ValueError("Unsupported embedding model type")
    
    def encode(self, texts):
        """Encode texts using the appropriate method."""
        if self.is_huggingface:
            # For LangChain HuggingFaceEmbeddings
            if isinstance(texts, str):
                return self.model.embed_query(texts)
            return [self.model.embed_query(text) for text in texts]
        elif self.is_sentence_transformer:
            # For SentenceTransformer
            return self.model.encode(texts)
    
    def embed_query(self, text):
        """Embed a single query text."""
        if self.is_huggingface:
            return self.model.embed_query(text)
        elif self.is_sentence_transformer:
            return self.model.encode(text if isinstance(text, list) else [text])[0]
        
 # Logger - Centralized logging

# Configuration settings
class PDFProcessorConfig:
    def __init__(self, 
                 chunk_size=1000,
                 chunk_overlap=200,
                 embedding_model=None,
                 library_dir="./ebooks_library",
                 processed_dir="./processed_data",
                 similarity_threshold=0.7,
                 use_semantic_chunking=True,
                 max_workers=4,
                 chunking_strategy="semantic",
                 use_llm=True):
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = EmbeddingsWrapper(embedding_model) if embedding_model else None
        self.library_dir = library_dir
        self.processed_dir = processed_dir
        self.similarity_threshold = similarity_threshold
        self.use_semantic_chunking = use_semantic_chunking
        self.max_workers = max_workers
        self.chunking_strategy = chunking_strategy  # "semantic", "fixed", "hybrid"
        self.use_llm = use_llm  # Whether to use LLM for structure extraction
        
        # Validate config
        self._validate()
        
    def _validate(self):
        """Validate configuration parameters."""
        if self.chunk_size < 100:
            raise ValueError("chunk_size must be at least 100")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.chunking_strategy not in ["semantic", "fixed", "hybrid"]:
            raise ValueError("chunking_strategy must be one of: semantic, fixed, hybrid")

# Text Splitting

class SemanticTextSplitter:
    """Advanced text splitter that considers semantic breaks in addition to character-based splitting."""
    
    def __init__(self, embedding_model, chunk_size=1000, chunk_overlap=200):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        # Check if we're using HuggingFaceEmbeddings or direct SentenceTransformer
        self.use_embed_query = hasattr(embedding_model, 'embed_query')
        self.similarity_threshold = 0.7  # Default threshold for detecting natural breaks
    
    def split_text(self, text: str) -> List[str]:
        """Split text using a combination of character-based and semantic approaches."""
        # First do standard splitting to get initial chunks
        initial_chunks = self.base_splitter.split_text(text)
        
        if len(initial_chunks) <= 1:
            return initial_chunks
            
        # For very long texts, process in batches to avoid memory issues
        if len(initial_chunks) > 50:
            return self._process_large_text(initial_chunks)
            
        return self._optimize_chunks(initial_chunks)
        
    def _process_large_text(self, initial_chunks: List[str]) -> List[str]:
        """Process very large texts by splitting into manageable batches."""
        batch_size = 40  # Process chunks in batches of 40
        optimized_chunks = []
        
        for i in range(0, len(initial_chunks), batch_size):
            batch = initial_chunks[i:min(i+batch_size, len(initial_chunks))]
            optimized_batch = self._optimize_chunks(batch)
            optimized_chunks.extend(optimized_batch)
            
        return optimized_chunks
    
    def _optimize_chunks(self, chunks: List[str]) -> List[str]:
        """Optimize chunk boundaries based on semantic similarity."""
        if len(chunks) <= 1:
            return chunks
            
        try:
            # Get embeddings for each chunk
            if self.use_embed_query:
                # Using LangChain's HuggingFaceEmbeddings
                embeddings = [self.embedding_model.embed_query(chunk) for chunk in chunks]
            else:
                # Using SentenceTransformer directly
                embeddings = self.embedding_model.encode(chunks)
                
            # Calculate similarity between adjacent chunks
            similarities = []
            for i in range(len(chunks)-1):
                # Handle different array shapes based on embedding type
                if isinstance(embeddings[0], list):
                    # For list-based embeddings
                    import numpy as np
                    vec1 = np.array(embeddings[i]).reshape(1, -1)
                    vec2 = np.array(embeddings[i+1]).reshape(1, -1)
                    sim = cosine_similarity(vec1, vec2)[0][0]
                else:
                    # For numpy array embeddings
                    sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                similarities.append(sim)
            
            # Find natural breakpoints (where similarity is lowest)
            breakpoints = []
            for i, sim in enumerate(similarities):
                if sim < self.similarity_threshold:  # Threshold for detecting natural breaks
                    breakpoints.append(i+1)
            
            # If no good breakpoints found, return original chunks
            if not breakpoints:
                return chunks
                
            # Create new chunks based on discovered breakpoints
            optimized_chunks = []
            start_idx = 0
            
            for bp in breakpoints:
                # Join chunks from start_idx to bp
                combined = " ".join(chunks[start_idx:bp])
                # Re-split to maintain roughly consistent chunk size
                if len(combined) > self.chunk_size * 1.5:
                    # If too large, use basic splitter to break it down
                    sub_chunks = self.base_splitter.split_text(combined)
                    optimized_chunks.extend(sub_chunks)
                else:
                    optimized_chunks.append(combined)
                start_idx = bp
                
            # Don't forget the last segment
            if start_idx < len(chunks):
                combined = " ".join(chunks[start_idx:])
                if len(combined) > self.chunk_size * 1.5:
                    sub_chunks = self.base_splitter.split_text(combined)
                    optimized_chunks.extend(sub_chunks)
                else:
                    optimized_chunks.append(combined)
                    
            return optimized_chunks
            
        except Exception as e:
            # If semantic chunking fails, fall back to basic chunking
            print(f"Semantic chunking failed, falling back to basic chunking: {str(e)}")
            return chunks
    
    def create_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[Document]:
        """Create Document objects from texts."""
        if not metadatas:
            metadatas = [{} for _ in texts]
            
        documents = []
        for i, text in enumerate(texts):
            chunks = self.split_text(text)
            
            for j, chunk in enumerate(chunks):
                meta = metadatas[i].copy()
                meta["chunk_index"] = j
                meta["total_chunks"] = len(chunks)
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=meta
                ))
                
        return documents

#  Main Processing Class: EnhancedPDFProcessor

class EnhancedPDFProcessor:
    def __init__(self, config=None):
        """Initialize with configuration."""
        if config is None:
            config = PDFProcessorConfig()
        
        self.config = config
        
        # Setup directories from config
        os.makedirs(self.config.library_dir, exist_ok=True)
        os.makedirs(self.config.processed_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('pdf_processor')
        
        # Initialize splitter based on config
        if config.use_semantic_chunking and config.embedding_model:
            self.semantic_splitter = SemanticTextSplitter(
                config.embedding_model,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
        else:
            self.semantic_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                separators=["\n\n", "\n", ". ", ", ", " ", ""]
            )

# Core Methods
    
    # Main entry points
    def process_pdf(self, pdf_path: str, vector_db=None, force_reprocess: bool = False) -> Tuple[Dict[str, Any], List[Document]]:
            """Process a PDF file with improved structure extraction for all types of educational books."""
            try:
                # Validate PDF before processing
                if not os.path.exists(pdf_path):
                    raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                    
                # Check if file is actually a PDF
                with open(pdf_path, 'rb') as f:
                    header = f.read(4)
                    if header != b'%PDF':
                        raise ValueError(f"File is not a valid PDF: {pdf_path}")
                
                # Check if already processed
                file_hash = self.get_pdf_hash(pdf_path)
                processed_path = os.path.join(self.config.processed_dir, file_hash)
                
                if not force_reprocess and os.path.exists(processed_path):
                    return self.load_processed_pdf(file_hash)
                
                # Open PDF and extract basic metadata
                doc = fitz.open(pdf_path)
                if doc.is_encrypted:
                    raise ValueError(f"PDF is encrypted and cannot be processed: {pdf_path}")
                
                # Try to get title from metadata, fallback to filename
                title = doc.metadata.get("title", "")
                if not title or title.strip() == "":
                    title = os.path.basename(pdf_path).replace('.pdf', '')
                    
                author = doc.metadata.get("author", "Unknown")
                
                # Extract structure with enhanced methods
                headers = self.extract_headers_and_structure(doc)
                
                self.logger.info(f"Detected {len(headers)} headers in {title}")
                for i, header in enumerate(headers[:5]):  # Log first 5 headers
                    self.logger.info(f"  - Page {header['page']}: {header['title']}")
                if len(headers) > 5:
                    self.logger.info(f"  - ... and {len(headers) - 5} more")
                
                # Extract tables with improved classification for educational content
                tables, content_blocks = self._detect_tables_vs_content_blocks(doc)
                self.logger.info(f"Detected {len(tables)} tables and {len(content_blocks)} content blocks in {title}")
                
                # Create a mapping of page numbers to headers
                page_to_headers = {}
                for header in headers:
                    page = header["page"]
                    if page not in page_to_headers:
                        page_to_headers[page] = []
                    page_to_headers[page].append(header)
                
                # Extract text with structural context
                pages_content = []
                for page_num, page in enumerate(doc):
                    page_idx = page_num + 1
                    text = page.get_text()
                    
                    current_headers = page_to_headers.get(page_idx, [])
                    
                    # Sort headers by position
                    current_headers.sort(key=lambda h: h.get("position", 0))
                    
                    # Get tables on this page
                    page_tables = [t for t in tables if t["page_num"] == page_idx]
                    
                    # Get content blocks on this page
                    page_content_blocks = [b for b in content_blocks if b["page_num"] == page_idx]
                    
                    pages_content.append({
                        "page_num": page_idx,
                        "text": text,
                        "headers": current_headers,
                        "tables": page_tables,
                        "content_blocks": page_content_blocks
                    })
                
                # Create PDF data object
                pdf_data = {
                    "book_title": title,
                    "author": author,
                    "file_hash": file_hash,
                    "filename": os.path.basename(pdf_path),
                    "total_pages": len(doc),
                    "headers": headers,
                    "tables": tables,
                    "content_blocks": content_blocks,
                    "pages": pages_content
                }
                
                # Generate chunks with improved contextual metadata
                chunks = self.create_semantic_chunks(pdf_data)
                
                # Save processed data
                self.save_processed_pdf(pdf_data, chunks)
                
                # Add directly to vector database if provided
                if vector_db is not None:
                    self.add_chunks_to_vector_db(chunks, vector_db)
                
                return pdf_data, chunks
            
            except Exception as e:
                self.logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
                self.logger.debug(traceback.format_exc())
                raise
    
    # Process multiple PDFs
    def process_library(self, vector_db=None, force_reprocess=False, parallel=True, max_workers=4):
        """Process all PDFs in the library directory with parallelism option."""
        pdf_files = [os.path.join(self.config.library_dir, f) for f in os.listdir(self.config.library_dir) 
                    if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {self.config.library_dir}")
            return {}
        
        print(f"Found {len(pdf_files)} PDF files in {self.config.library_dir}")
        
        processed_books = {}
        
        if parallel and len(pdf_files) > 1:
            # Use parallel processing
            self.parallel_process_pdfs(pdf_files, vector_db, max_workers)
            
            # Load results from processed directory
            for filename in tqdm(os.listdir(self.config.library_dir), desc="Processing PDFs"):
                if filename.endswith('.pdf'):
                    pdf_path = os.path.join(self.config.library_dir, filename)
                    file_hash = self.get_pdf_hash(pdf_path)
                    pass

                    try:
                        pdf_data, chunks = self.load_processed_pdf(file_hash)
                        processed_books[pdf_data["book_title"]] = {
                            "pdf_data": pdf_data,
                            "chunks": chunks
                        }
                    except Exception as e:
                        print(f"Error loading processed data for {filename}: {str(e)}")
        else:
            # Sequential processing
            for filename in os.listdir(self.config.library_dir):
                if filename.endswith('.pdf'):
                    pdf_path = os.path.join(self.config.library_dir, filename)
                    print(f"Processing {filename}...")
                    
                    try:
                        pdf_data, chunks = self.process_pdf(pdf_path, vector_db, force_reprocess)
                        processed_books[pdf_data["book_title"]] = {
                            "pdf_data": pdf_data,
                            "chunks": chunks
                        }
                        print(f"✓ Processed {len(chunks)} chunks from {filename}")
                    except Exception as e:
                        print(f"✗ Error processing {filename}: {str(e)}")
                    
        return processed_books
    
    async def process_library_async(self, vector_db=None, force_reprocess=False, max_workers=4):
        """Process all PDFs in the library directory asynchronously."""
        pdf_files = [os.path.join(self.config.library_dir, f) for f in os.listdir(self.config.library_dir) 
                    if f.endswith('.pdf')]
        
        if not pdf_files:
            self.logger.info(f"No PDF files found in {self.config.library_dir}")
            return {}
        
        self.logger.info(f"Found {len(pdf_files)} PDF files in {self.config.library_dir}")
        
        # Create tasks for each PDF
        tasks = []
        for pdf_path in pdf_files:
            tasks.append(self.process_pdf_async(pdf_path, vector_db, force_reprocess))
        
        # Process PDFs concurrently with limit on max concurrent tasks
        processed_books = {}
        semaphore = asyncio.Semaphore(max_workers)
        
        async def process_with_semaphore(pdf_path):
            async with semaphore:
                return await self.process_pdf_async(pdf_path, vector_db, force_reprocess)
        
        results = await asyncio.gather(*[process_with_semaphore(pdf) for pdf in pdf_files], 
                                    return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Error processing {pdf_files[i]}: {str(result)}")
            else:
                pdf_data, chunks = result
                processed_books[pdf_data["book_title"]] = {
                    "pdf_data": pdf_data,
                    "chunks": chunks
                }
        
        return processed_books
    
    # Multi-threaded processing
    def parallel_process_pdfs(self, pdf_files: List[str], vector_db=None, max_workers: int = 4):
        """Process multiple PDFs in parallel for better performance."""
        # Create a thread-safe queue for results
        result_queue = queue.Queue()
        
        # Function to process one PDF and put result in queue
        def process_one_pdf(pdf_path):
            try:
                pdf_data, chunks = self.process_pdf(pdf_path, vector_db)
                book_title = pdf_data["book_title"]
                chunks_count = len(chunks)
                result_queue.put((True, book_title, chunks_count))
            except Exception as e:
                result_queue.put((False, os.path.basename(pdf_path), str(e)))
        
        # Create and start worker threads
        threads = []
        active_workers = min(max_workers, len(pdf_files))
        
        print(f"Processing {len(pdf_files)} PDFs using {active_workers} parallel workers...")
        
        for pdf_path in pdf_files:
            thread = threading.Thread(target=process_one_pdf, args=(pdf_path,))
            threads.append(thread)
            thread.start()
            
            # Limit concurrent threads
            while sum(t.is_alive() for t in threads) >= max_workers:
                time.sleep(0.1)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # Print summary
        success_count = sum(1 for r in results if r[0])
        print(f"✓ Successfully processed {success_count}/{len(pdf_files)} PDFs")
        
        # Print any errors
        for result in results:
            if not result[0]:  # If not successful
                print(f"  ✗ Error processing {result[1]}: {result[2]}")
        
        return results
    
# Methods for document structure extraction
    
    # Main structure extraction method
    def extract_headers_and_structure(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """Extract document structure with optimized approach prioritizing LLM when available."""
        headers = []
        
        # Always prioritize LLM-based extraction when enabled
        use_llm = hasattr(self.config, 'use_llm') and self.config.use_llm
        
        if use_llm:
            self.logger.info("Attempting LLM-based structure extraction...")
            
            # First try processing TOC with LLM (faster, more targeted approach)
            llm_toc_headers = self.process_toc_with_llm(doc)
            if llm_toc_headers and len(llm_toc_headers) >= 3:
                self.logger.info(f"Successfully extracted {len(llm_toc_headers)} TOC entries with LLM")
                return llm_toc_headers
                
            # If TOC processing failed, try more comprehensive structure extraction
            llm_structure_headers = self.extract_structure_with_llm(doc)
            if llm_structure_headers and len(llm_structure_headers) >= 3:
                self.logger.info(f"Successfully extracted {len(llm_structure_headers)} structure entries with LLM")
                return llm_structure_headers
                
            self.logger.info("LLM-based extraction did not produce sufficient results, falling back to traditional methods")
        
        # Try built-in TOC first (simple and fast)
        built_in_toc = doc.get_toc()
        if built_in_toc and len(built_in_toc) >= 3:
            # Convert built-in TOC to our format
            for i, (level, title, page) in enumerate(built_in_toc):
                headers.append({
                    "title": title,
                    "page": page,
                    "level": level,
                    "position": 0,
                    "id": f"toc_{i}",
                    "type": "toc_entry"
                })
            self.logger.info(f"Using {len(headers)} entries from built-in TOC")
            return headers
        
        # If built-in TOC failed, try visual TOC detection
        toc_headers = self._extract_from_toc_pages(doc)
        if toc_headers and len(toc_headers) >= 3:
            self.logger.info(f"Using {len(toc_headers)} entries from visual TOC")
            headers.extend(toc_headers)
            return self._organize_header_hierarchy(headers)
        
        # For educational books, use specialized extraction
        if self._is_educational_book(doc):
            edu_headers = self._extract_educational_book_structure(doc)
            if edu_headers and len(edu_headers) >= 3:
                self.logger.info(f"Using {len(edu_headers)} entries from educational book structure analysis")
                return edu_headers
        
        # If all TOC methods failed, use typography analysis
        font_stats = self._analyze_document_typography(doc)
        typography_headers = self._detect_headers_by_typography(doc, font_stats)
        
        if typography_headers:
            self.logger.info(f"Using {len(typography_headers)} headers from typography")
            headers.extend(typography_headers)
        
        # Final fallback: Try content pattern detection
        if len(headers) < 3:
            pattern_headers = self._detect_headers_by_content_patterns(doc)
            if pattern_headers:
                self.logger.info(f"Using {len(pattern_headers)} headers from content patterns")
                headers.extend(pattern_headers)
        
        # Sort, deduplicate and organize
        headers.sort(key=lambda h: (h["page"], h.get("position", 0)))
        headers = self._deduplicate_headers(headers)
        return self._organize_header_hierarchy(headers)
    
    # Extract from TOC pages
    def _extract_from_toc_pages(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """Extract structure from table of contents/index pages with improved detection for various books."""
        toc_headers = []
        
        # First try to use the document's built-in TOC if available
        built_in_toc = doc.get_toc()
        
        # Check if built-in TOC has actual content information or just metadata
        if built_in_toc:
            # Filter out entries that are likely metadata and not real content
            filtered_toc = []
            metadata_patterns = ['cover', 'prilim', 'text', 'pages', 'preliminary']
            
            for level, title, page in built_in_toc:
                # Check if this looks like a metadata entry
                is_metadata = False
                title_lower = title.lower()
                
                for pattern in metadata_patterns:
                    if pattern in title_lower:
                        is_metadata = True
                        break
                        
                # Also check if it's just a page number
                if re.match(r'^page\s+\d+$', title_lower):
                    is_metadata = True
                    
                if not is_metadata:
                    filtered_toc.append((level, title, page))
            
            # Only use built-in TOC if we have actual content entries after filtering
            if len(filtered_toc) >= 3:
                self.logger.info(f"Found usable built-in TOC with {len(filtered_toc)} entries")
                for i, (level, title, page) in enumerate(filtered_toc):
                    toc_headers.append({
                        "title": title,
                        "page": page,
                        "level": level,
                        "position": 0,
                        "id": f"toc_{i}",
                        "type": "toc_entry"
                    })
                return toc_headers
            else:
                self.logger.info("Built-in TOC exists but appears to be metadata, not content information")
        
        # Look for visual TOC/index pages
        max_pages_to_check = min(30, len(doc))  # Check more pages for TOC
        
        # Keywords for various types of TOC/index pages
        toc_keywords = [
            'contents', 'content', 'index', 'chapters', 'topics', 
            'lessons', 'syllabus', 'curriculum', 'units', 'sections',
            'table of contents', 'toc', 'subjects', 'parts'
        ]
        
        toc_page_candidates = []
        
        # Step 1: Identify potential TOC pages
        for page_num in range(max_pages_to_check):
            page = doc[page_num]
            page_text = page.get_text().lower()
            
            # Check for TOC header markers
            for keyword in toc_keywords:
                # Check if keyword appears as a header
                if re.search(r'(^|\n|\s)' + keyword + r'(\s|$|\n)', page_text):
                    toc_page_candidates.append(page_num)
                    break
            
            # Look for patterns that suggest a TOC page
            # Pattern 1: Number + text + page number (with dots or spaces between)
            if re.search(r'\d+\s*\.?[^\n\d]+\d+\s*$', page_text, re.MULTILINE):
                toc_page_candidates.append(page_num)
                
            # Pattern 2: Multiple page number references at line ends
            page_ref_count = len(re.findall(r'\.{2,}\s*\d+\s*$', page_text, re.MULTILINE))
            if page_ref_count >= 3:
                toc_page_candidates.append(page_num)
        
        # Remove duplicates and sort
        toc_page_candidates = sorted(set(toc_page_candidates))
        
        # Step 2: Process candidate TOC pages with multiple pattern matching strategies
        for page_num in toc_page_candidates:
            page = doc[page_num]
            page_text = page.get_text()
            
            # Multiple pattern matching for different TOC formats
            
            # Pattern 1: Chapter/section number + title + page number (common in textbooks)
            # Example: "1. Introduction to Science..............15"
            entries = re.findall(
                r'(?:^|\n)\s*(\d+)\.?\s+([^\n\d][^\n]{3,}?)(?:\.{2,}|\s{3,}|\t+)\s*(\d+)\s*(?:\n|$)', 
                page_text
            )
            
            for num, title, page in entries:
                title = title.strip()
                if len(title) < 3:  # Skip very short titles
                    continue
                    
                toc_headers.append({
                    "title": f"{num}. {title}",
                    "page": int(page),
                    "level": 1,
                    "position": page_text.find(f"{num}"),
                    "id": f"ch_{num}",
                    "type": "toc_entry"
                })
            
            # Pattern 2: Subsection numbering (e.g., "1.2 Advanced Topics....25")
            subsections = re.findall(
                r'(?:^|\n)\s*(\d+\.\d+)\.?\s+([^\n\d][^\n]{3,}?)(?:\.{2,}|\s{3,}|\t+)\s*(\d+)\s*(?:\n|$)', 
                page_text
            )
            
            for num, title, page in subsections:
                title = title.strip()
                if len(title) < 3:
                    continue
                    
                level = num.count('.') + 1
                toc_headers.append({
                    "title": f"{num} {title}",
                    "page": int(page),
                    "level": level,
                    "position": page_text.find(f"{num}"),
                    "id": f"sec_{num.replace('.', '_')}",
                    "type": "toc_entry"
                })
            
            # Pattern 3: Named sections without numbers (common in notes/guides)
            # Example: "Introduction................5"
            named_sections = re.findall(
                r'(?:^|\n)\s*([A-Z][^\n\d][^\n]{3,}?)(?:\.{2,}|\s{3,}|\t+)\s*(\d+)\s*(?:\n|$)', 
                page_text
            )
            
            for title, page in named_sections:
                title = title.strip()
                # Skip if already captured or too short
                if len(title) < 3 or any(title.lower() == h["title"].lower() for h in toc_headers):
                    continue
                    
                # Try to detect level from indentation or formatting
                level = 1
                if self._is_indented(page, title):
                    level = 2
                    
                toc_headers.append({
                    "title": title,
                    "page": int(page),
                    "level": level,
                    "position": page_text.find(title),
                    "id": f"sec_{len(toc_headers)}",
                    "type": "toc_entry"
                })
            
            # Pattern 4: Examine for tabular structure (common in educational books)
            # Look for repeating patterns of text + page number
            if not toc_headers and "contents" in page_text.lower():
                # Try to detect table-like structure through pattern analysis
                lines = page_text.strip().split('\n')
                potential_toc_lines = []
                
                for line in lines:
                    # Check if line ends with a page number
                    if re.search(r'\d+\s*$', line) and len(line.strip()) > 3:
                        potential_toc_lines.append(line)
                
                # If we have multiple potential TOC lines
                if len(potential_toc_lines) >= 3:
                    for i, line in enumerate(potential_toc_lines):
                        # Extract title and page number
                        match = re.search(r'(.+?)(?:\.{2,}|\s{3,}|\t+)\s*(\d+)\s*$', line)
                        if match:
                            title = match.group(1).strip()
                            page_num = int(match.group(2))
                            
                            # Skip very short or empty titles
                            if len(title) < 3:
                                continue
                                
                            # Try to detect numbering pattern
                            num_match = re.match(r'^\s*(\d+\.?)\s+(.+)$', title)
                            if num_match:
                                num = num_match.group(1)
                                title = num_match.group(2)
                                toc_headers.append({
                                    "title": f"{num} {title}",
                                    "page": page_num,
                                    "level": 1,
                                    "position": page_text.find(line),
                                    "id": f"ch_{num.replace('.', '')}",
                                    "type": "toc_entry"
                                })
                            else:
                                toc_headers.append({
                                    "title": title,
                                    "page": page_num,
                                    "level": 1,
                                    "position": page_text.find(line),
                                    "id": f"sec_{i}",
                                    "type": "toc_entry"
                                })
        
        # If we found enough TOC entries, deduplicate and return
        if len(toc_headers) >= 3:
            self.logger.info(f"Successfully extracted {len(toc_headers)} entries from TOC/index pages")
            return self._deduplicate_headers(toc_headers)
        
        # If we still don't have enough entries, try one more specialized approach for educational books
        for page_num in range(min(5, len(doc))):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            # Look for a potential TOC based on visual layout and formatting
            for block in blocks:
                if "lines" not in block:
                    continue
                    
                # Check if this block looks like a TOC header
                block_text = "".join([span["text"] for span in block["lines"][0]["spans"]]) if block["lines"] else ""
                
                if any(keyword in block_text.lower() for keyword in ['contents', 'content', 'index']):
                    # Process subsequent blocks as potential TOC entries
                    for next_idx in range(blocks.index(block) + 1, len(blocks)):
                        next_block = blocks[next_idx]
                        if "lines" not in next_block:
                            continue
                            
                        for line in next_block["lines"]:
                            line_text = "".join([span["text"] for span in line["spans"]])
                            
                            # Check for possible TOC entry patterns
                            entry_match = re.search(r'(.+?)(?:\.{2,}|\s{3,})\s*(\d+)\s*$', line_text)
                            if entry_match:
                                title = entry_match.group(1).strip()
                                page_num = int(entry_match.group(2))
                                
                                if len(title) < 3:
                                    continue
                                    
                                toc_headers.append({
                                    "title": title,
                                    "page": page_num,
                                    "level": 1, 
                                    "position": line["bbox"][1],  # Y position
                                    "id": f"toc_{len(toc_headers)}",
                                    "type": "toc_entry"
                                })
        
        # If we now have enough entries, return them
        if len(toc_headers) >= 3:
            self.logger.info(f"Extracted {len(toc_headers)} entries from TOC using advanced detection")
            return self._deduplicate_headers(toc_headers)
        
        # If all else fails, return what we found (even if insufficient)
        self.logger.info(f"Could only find {len(toc_headers)} TOC entries, will try additional methods")
        return toc_headers

    def _is_indented(self, page, text):
        """Check if text appears to be indented (useful for detecting subsections)."""
        blocks = page.get_text("dict")["blocks"]
        page_width = page.rect.width
        left_margin_threshold = page_width * 0.15  # 15% of page width
        
        for block in blocks:
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                line_text = "".join([span["text"] for span in line["spans"]])
                if text in line_text:
                    # Check if this line is indented
                    if line["spans"] and "origin" in line["spans"][0]:
                        x_position = line["spans"][0]["origin"][0]
                        return x_position > left_margin_threshold
        
        return False
    
    def _extract_and_parse_json(self, response_text):
        """Safely extract and parse JSON from LLM response text.
        
        Args:
            response_text: The text response from the LLM
            
        Returns:
            Parsed JSON data or None if parsing fails
        """
        import re
        import json
        
        try:
            # First try direct JSON parsing
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from text
                pass
                
            # Look for JSON array pattern
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
                    
            # Try finding JSON with more aggressive pattern
            json_match = re.search(r'\[[\s\S]*\]', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
                    
            # Try to fix common JSON errors in LLM responses
            # 1. Replace single quotes with double quotes
            fixed_text = re.sub(r"'([^']*)':", r'"\1":', response_text)
            # 2. Fix missing commas between objects
            fixed_text = re.sub(r'}\s*{', '},{', fixed_text)
            # 3. Try to extract a valid JSON array
            json_match = re.search(r'\[\s*\{.*\}\s*\]', fixed_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
                    
            # As a last resort, try to build JSON manually by parsing the response
            if "title" in response_text and "level" in response_text and "page" in response_text:
                manual_json = []
                # Find patterns like "title": "something", "level": number, "page": number
                pattern = r'"title":\s*"([^"]+)"[,\s]+"level":\s*(\d+)[,\s]+"page":\s*(\d+)'
                matches = re.findall(pattern, response_text)
                for title, level, page in matches:
                    manual_json.append({
                        "title": title,
                        "level": int(level),
                        "page": int(page)
                    })
                if manual_json:
                    return manual_json
                    
            # If all else fails
            return None
            
        except Exception as e:
            self.logger.warning(f"Error parsing JSON from LLM response: {str(e)}")
            return None
    
    # Use LLM for TOC extraction
    def extract_structure_with_llm(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """Extract document structure using LLM for comprehensive analysis.
        
        This function uses LLM to analyze document content and extract hierarchical structure,
        even when a formal TOC isn't available or is incomplete. It's designed specifically
        for educational materials including textbooks, guides, and notes.
        """
        # Import necessary modules
        import re
        import json
        from llm_interface import setup_educational_llm
        
        try:
            # Get LLM instance
            llm = setup_educational_llm()
            
            # Get document metadata
            metadata = doc.metadata
            title = metadata.get("title", "")
            if not title or title.strip() == "":
                title = os.path.basename(doc.name).replace('.pdf', '') if hasattr(doc, 'name') else "Unknown"
            
            # Analyze document structure by sampling pages strategically
            content_samples = []
            total_pages = len(doc)
            
            # Get key pages to analyze - beginning, key intervals, and TOC candidates
            # Most educational books have structure established in first 20% of pages
            sample_pages = [0, 1, 2, 3, 4, 5, 10, 15, 20, total_pages//10, total_pages//5]
            sample_pages = sorted(set([p for p in sample_pages if p < total_pages]))
            
            # Identify TOC pages
            toc_pages = self._identify_toc_pages(doc)
            if toc_pages:
                # Add TOC pages to our samples with high priority
                sample_pages = sorted(set(sample_pages + toc_pages))
            
            # Extract header candidates from sample pages
            header_candidates = []
            
            for page_num in sample_pages:
                page = doc[page_num]
                page_text = page.get_text()
                
                # Add sample content (first 500 chars from each sampled page)
                content_samples.append("Page {}:\n{}...".format(page_num+1, page_text[:500]))
                
                # Extract potential headers using multiple heuristics
                lines = page_text.split('\n')
                for line_idx, line in enumerate(lines):
                    line = line.strip()
                    # Skip empty or very short lines
                    if not line or len(line) < 3 or len(line) > 150:
                        continue
                    
                    # Check for various header patterns common in educational materials
                    is_header_candidate = (
                        # Numbered headers: "1. Title", "1.1 Title", etc.
                        re.match(r'^\d+(\.\d+)*\s+\w', line) or
                        # Chapter/Unit style: "Chapter 1: Title", "Unit 2 - Title"
                        re.match(r'^(chapter|unit|section|lesson|module|part|topic)\s+\d+[:\-\s]', line, re.IGNORECASE) or
                        # ALL CAPS headers
                        (line.isupper() and 5 <= len(line) <= 50) or
                        # Headers on their own line with proper capitalization
                        (3 <= len(line.split()) <= 8 and line[0].isupper() and 
                        (line_idx == 0 or not lines[line_idx-1].strip()) and
                        (line_idx == len(lines)-1 or not lines[line_idx+1].strip()))
                    )
                    
                    if is_header_candidate:
                        header_candidates.append((page_num+1, line))
            
            # Format header candidates for LLM
            candidate_content = "\nPOTENTIAL HEADERS:\n"
            if header_candidates:
                candidate_content += "\n".join(["Page {}: {}".format(page, text) for page, text in header_candidates[:50]])
            else:
                candidate_content += "[No clear header candidates found]"
            
            # Add table of contents data if available
            toc_content = ""
            built_in_toc = doc.get_toc()
            if built_in_toc:
                toc_content += "\nBUILT-IN TOC:\n"
                for level, title, page in built_in_toc:
                    indent = "  " * (level-1)
                    toc_content += "{}- {} (page {})\n".format(indent, title, page)
            
            # Create a comprehensive, education-focused prompt
            prompt = """You are analyzing an educational document titled "{0}" with {1} pages.
    As an expert in extracting structure from educational materials (textbooks, guides, study notes, etc.), 
    create a comprehensive table of contents by identifying chapters, sections, and subsections.

    Here are content samples from key pages:
    {2}
    {3}
    {4}

    Based on this information, create a structured table of contents for this educational material with:
    - "title": The exact chapter/section title text as it appears in the document
    - "level": Hierarchy level (1 for chapters/units, 2 for sections, 3 for subsections)
    - "page": Page number where the section begins

    Pay special attention to:
    1. Chapter/unit numbering patterns (both numeric and text-based)
    2. Educational section markers (Lessons, Topics, Modules, etc.)
    3. Hierarchical relationships between sections
    4. Page number references

    Return only a JSON array with this structure:
    [
    {{"title": "Chapter 1: Introduction", "level": 1, "page": 1}},
    {{"title": "1.1 Key Concepts", "level": 2, "page": 3}}
    ]
    """.format(
                title, 
                total_pages, 
                "\n".join(content_samples[:5]),
                candidate_content,
                toc_content
            )
            
            # Get LLM response
            from langchain_core.output_parsers import StrOutputParser
            output_parser = StrOutputParser()
            response = output_parser.invoke(llm.invoke(prompt))
            
            # Parse JSON response
            import json
            import re
            
            # Extract JSON if embedded in text
            json_match = re.search(r'\[\s*{.*}\s*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
            
            # Convert to proper headers format
            try:
                structure_data = json.loads(json_str)
                headers = []
                
                for i, entry in enumerate(structure_data):
                    if "title" in entry and "level" in entry and "page" in entry:
                        # Clean and validate data
                        title = entry["title"].strip()
                        try:
                            level = int(entry["level"])
                            page = int(entry["page"])
                        except (ValueError, TypeError):
                            level = 1
                            page = 1
                        
                        # Skip non-content entries
                        if any(term in title.lower() for term in ["cover", "blank page"]):
                            continue
                        
                        headers.append({
                            "title": title,
                            "page": page,
                            "level": level,
                            "position": i,  # Preserve detected order
                            "id": f"llm_structure_{i}",
                            "type": "llm_content"
                        })
                
                self.logger.info(f"LLM extracted {len(headers)} structure entries from content")
                return headers
                
            except Exception as e:
                self.logger.warning(f"Failed to parse LLM structure extraction: {str(e)}")
                return []
                
        except Exception as e:
            self.logger.warning(f"Error in LLM structure extraction: {str(e)}")
            return []
    
    def _get_toc_content_for_llm(self, doc: fitz.Document) -> str:
        """Extract TOC content for LLM processing.
        
        This method collects relevant TOC information from both the built-in TOC
        and potential TOC pages in the document.
        """
        content = ""
        
        # 1. Try built-in TOC first (faster and usually accurate when available)
        built_in_toc = doc.get_toc()
        if built_in_toc:
            content += "BUILT-IN TABLE OF CONTENTS:\n"
            for level, title, page in built_in_toc:
                content += f"{'  ' * (level-1)}- {title} (Page {page})\n"
            
            # If built-in TOC seems substantial, just return it
            if len(built_in_toc) >= 5:
                return content
        
        # 2. Look for visual TOC/index pages
        max_pages_to_check = min(30, len(doc))
        toc_page_candidates = []
        
        # Check for TOC indicator pages
        for page_num in range(max_pages_to_check):
            page = doc[page_num]
            page_text = page.get_text()
            
            # Check for TOC indicators
            lower_text = page_text.lower()
            if any(marker in lower_text for marker in ['contents', 'index', 'chapters', 'toc', 'table of contents']):
                toc_page_candidates.append(page_num)
                
                # Also check next page (TOCs often span multiple pages)
                if page_num < len(doc) - 1:
                    toc_page_candidates.append(page_num + 1)
                    
                    # For longer TOCs, check one more page
                    if page_num < len(doc) - 2:
                        toc_page_candidates.append(page_num + 2)
        
        # Add content from all candidate TOC pages
        for page_num in sorted(set(toc_page_candidates)):
            page = doc[page_num]
            page_text = page.get_text()
            
            # Clean up the text to remove unnecessary whitespace
            lines = [line.strip() for line in page_text.split('\n') if line.strip()]
            clean_text = '\n'.join(lines)
            
            content += f"\n--- TOC PAGE {page_num+1} ---\n{clean_text}\n"
        
        return content

    def process_toc_with_llm(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """Process table of contents with LLM to extract structure efficiently.
        
        This is an optimized function that uses LLM to extract structure from TOC content,
        replacing multiple pattern-matching functions with a single powerful LLM call.
        """
        # Import necessary modules
        import re
        import json
        from llm_interface import setup_educational_llm
        
        try:
            # Get LLM instance
            llm = setup_educational_llm()
            
            # Extract TOC content
            toc_content = self._get_toc_content_for_llm(doc)
            
            if not toc_content:
                self.logger.info("No TOC content found for LLM processing")
                return []
            
            # Get book metadata for better context
            metadata = doc.metadata
            title = metadata.get("title", "")
            if not title or title.strip() == "":
                title = os.path.basename(doc.name).replace('.pdf', '') if hasattr(doc, 'name') else "Unknown"
            
            # Create a more detailed prompt for the LLM with specific document context
            prompt = """You are analyzing an educational document titled "{0}".
    Your task is to extract its table of contents structure accurately from the content below.

    Return a JSON array with objects containing:
    - "title": The exact chapter/section title text
    - "level": Hierarchy level (1 for chapters, 2 for sections, 3 for subsections, etc.)
    - "page": Page number as an integer

    CONTENT:
    {1}

    Analyze specifically:
    1. Chapter numbers and titles (e.g., "Chapter 1: Introduction")
    2. Section and subsection numbering (e.g., "1.1 Background", "1.1.2 Details")
    3. Named sections without numbers (e.g., "Appendix", "Bibliography")
    4. Page number references
    5. Educational indicators (Units, Lessons, Topics, etc.)

    Return only valid JSON with no extra text. Example format:
    [
    {{"title": "Chapter 1: Introduction", "level": 1, "page": 1}},
    {{"title": "1.1 Background", "level": 2, "page": 2}}
    ]
    """.format(title, toc_content)
            
            # Get LLM response
            from langchain_core.output_parsers import StrOutputParser
            output_parser = StrOutputParser()
            response = output_parser.invoke(llm.invoke(prompt))
            
            # Parse JSON response
            import json
            import re
            
            # Extract JSON if embedded in text
            json_match = re.search(r'\[\s*{.*}\s*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
            
            # Convert to proper headers format
            try:
                toc_data = json.loads(json_str)
                headers = []
                
                for i, entry in enumerate(toc_data):
                    if "title" in entry and "level" in entry and "page" in entry:
                        # Clean and validate the data
                        title = entry["title"].strip()
                        
                        try:
                            level = int(entry["level"])
                            page = int(entry["page"])
                        except (ValueError, TypeError):
                            level = 1
                            page = 1
                        
                        # Skip entries that don't look like actual content
                        if any(term in title.lower() for term in ["cover", "blank page", "index of"]):
                            continue
                        
                        headers.append({
                            "title": title,
                            "page": page,
                            "level": level,
                            "position": i,
                            "id": f"llm_toc_{i}",
                            "type": "llm_toc"
                        })
                
                self.logger.info(f"LLM successfully extracted {len(headers)} TOC entries")
                return headers
                
            except Exception as e:
                self.logger.warning(f"Failed to parse LLM TOC extraction: {str(e)}")
                return []
                
        except Exception as e:
            self.logger.warning(f"Error in LLM TOC extraction: {str(e)}")
            return []

    def _get_toc_content_for_llm(self, doc: fitz.Document) -> str:
        """Extract TOC content for LLM processing.
        
        This method collects relevant TOC information from both the built-in TOC
        and potential TOC pages in the document.
        """
        content = ""
        
        # 1. Try built-in TOC first (faster and usually accurate when available)
        built_in_toc = doc.get_toc()
        if built_in_toc:
            content += "BUILT-IN TABLE OF CONTENTS:\n"
            for level, title, page in built_in_toc:
                content += f"{'  ' * (level-1)}- {title} (Page {page})\n"
        
        # 2. Look for visual TOC/index pages in the first 30 pages
        max_pages_to_check = min(30, len(doc))
        toc_page_candidates = []
        
        # Keywords for various types of TOC/index pages
        toc_keywords = [
            'contents', 'table of contents', 'index', 'chapters', 'topics', 
            'lessons', 'syllabus', 'curriculum', 'units', 'sections'
        ]
        
        # Identify potential TOC pages
        for page_num in range(max_pages_to_check):
            page = doc[page_num]
            page_text = page.get_text().lower()
            
            # Check for TOC indicators
            if any(keyword in page_text for keyword in toc_keywords):
                toc_page_candidates.append(page_num)
                
                # Also check next page (TOCs often span multiple pages)
                if page_num < len(doc) - 1:
                    toc_page_candidates.append(page_num + 1)
                    
                    # For longer TOCs, check one more page
                    if page_num < len(doc) - 2 and any(
                        re.search(r'\d+\s*\.\s*\w+.*\d+\s*$', line) 
                        for line in page.get_text().split('\n')):
                        toc_page_candidates.append(page_num + 2)
        
        # Extract content from all candidate TOC pages
        for page_num in sorted(set(toc_page_candidates)):
            page = doc[page_num]
            page_text = page.get_text()
            
            # Clean up the text to remove unnecessary whitespace
            lines = [line.strip() for line in page_text.split('\n') if line.strip()]
            clean_text = '\n'.join(lines)
            
            content += f"\n--- TOC PAGE {page_num+1} ---\n{clean_text}\n"
        
        return content

    
    def _extract_from_builtin_toc(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """Extract structure from built-in TOC with simple filtering."""
        toc_headers = []
        built_in_toc = doc.get_toc()
        
        if not built_in_toc:
            return []
        
        # Filter out metadata entries
        metadata_patterns = ['cover', 'prilim', 'text', 'pages']
        filtered_toc = []
        
        for level, title, page in built_in_toc:
            # Skip metadata entries
            if any(pattern in title.lower() for pattern in metadata_patterns):
                continue
            # Skip page number entries
            if re.match(r'^page\s+\d+$', title.lower()):
                continue
                
            filtered_toc.append((level, title, page))
        
        # Only use if we have meaningful entries
        if len(filtered_toc) >= 3:
            self.logger.info(f"Found usable built-in TOC with {len(filtered_toc)} entries")
            for i, (level, title, page) in enumerate(filtered_toc):
                toc_headers.append({
                    "title": title,
                    "page": page,
                    "level": level,
                    "position": 0,
                    "id": f"toc_{i}",
                    "type": "toc_entry"
                })
                
        return toc_headers
    
    # Typography-based detection
    def _detect_headers_by_typography(self, doc: fitz.Document, font_stats: Dict) -> List[Dict[str, Any]]:
        """Detect headers based on typography with optimized processing."""
        headers = []
        base_font_size = font_stats["most_common_size"]
        title_font_sizes = font_stats["larger_sizes"]
        
        # Process a sample of pages to identify headers
        # For large documents, we'll process beginning, middle and end
        pages_to_process = []
        total_pages = len(doc)
        
        # For smaller documents (< 100 pages), process all pages
        if total_pages <= 100:
            pages_to_process = list(range(total_pages))
        else:
            # First 50 pages
            pages_to_process.extend(range(min(50, total_pages)))
            
            # Middle section (sample)
            if total_pages > 100:
                middle_start = max(50, total_pages // 2 - 15)
                middle_end = min(total_pages, total_pages // 2 + 15)
                pages_to_process.extend(range(middle_start, middle_end))
                
            # Last 30 pages
            if total_pages > 30:
                pages_to_process.extend(range(max(0, total_pages - 30), total_pages))
                
            # Remove duplicates
            pages_to_process = sorted(set(pages_to_process))
        
        # Process selected pages
        for page_num in pages_to_process:
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block_idx, block in enumerate(blocks):
                if "lines" not in block:
                    continue
                    
                # Skip blocks that appear to be tables or figures
                if len(block["lines"]) > 0 and len(block["lines"][0]["spans"]) > 3:
                    # Many spans in the first line suggests a table
                    continue
                    
                # Single-line blocks are more likely to be headers
                is_potential_header_block = len(block["lines"]) <= 2
                
                for line_idx, line in enumerate(block["lines"]):
                    # Skip if this is unlikely to be a header block and not the first line
                    if not is_potential_header_block and line_idx > 0:
                        continue
                        
                    line_text = "".join([span["text"] for span in line["spans"]]).strip()
                    
                    # Skip empty or very short lines
                    if not line_text or len(line_text) < 3:
                        continue
                        
                    # Skip lines that start with bullet points or list markers
                    if line_text.startswith(('•', '-', '*', '>', '→', '▪', '◦')):
                        continue
                    
                    # Check each span in the line
                    for span in line["spans"]:
                        span_text = span["text"].strip()
                        
                        # Skip empty spans
                        if not span_text:
                            continue
                            
                        font_size = span["size"]
                        is_bold = span["flags"] & 2**4 > 0
                        
                        # Check if this span has header characteristics
                        is_larger_font = font_size > base_font_size * 1.2
                        is_header_font_size = font_size in title_font_sizes
                        
                        # Calculate header confidence score
                        header_score = 0
                        
                        if is_larger_font:
                            header_score += 2
                        elif is_header_font_size:
                            header_score += 1
                            
                        if is_bold:
                            header_score += 1
                            
                        if is_potential_header_block:
                            header_score += 1
                            
                        # Check if text matches known header patterns
                        if re.match(r'^(chapter|section|unit|part|module)\s+\d+', span_text.lower()):
                            header_score += 3
                        elif re.match(r'^\d+\.\d+(\.\d+)*\s+[A-Z]', span_text):
                            header_score += 2
                            
                        # If this is likely a header, add it
                        if header_score >= 2 and 3 <= len(span_text) <= 100:
                            headers.append({
                                "title": span_text,
                                "page": page_num + 1,
                                "position": span["origin"][1],  # y position for vertical ordering
                                "font_size": font_size,
                                "is_bold": is_bold,
                                "id": f"h_{page_num}_{len(headers)}",
                                "type": "typography_detected"
                            })
        
        return headers
    
# Table Processing Methods

    #
    def _detect_tables_vs_content_blocks(self, doc: fitz.Document) -> Tuple[List[Dict], List[Dict]]:
        """Differentiate between tables and educational content blocks (exercises, examples, etc.)."""
        tables = []
        content_blocks = []
        
        # Educational content markers (expand as needed for your specific books)
        content_block_markers = [
            'try this', 'activity', 'remember', 'note:', 'example', 'exercise',
            'practice', 'important', 'caution', 'warning', 'do you know',
            'think about it', 'brain power', 'observe and discuss'
        ]
        
        for page_num, page in enumerate(doc):
            # Get tables using PyMuPDF's built-in detection
            tab = page.find_tables()
            
            if tab.tables:
                for i, table in enumerate(tab.tables):
                    try:
                        # Extract table content
                        rows = table.extract()
                        table_data = [[cell.strip() if isinstance(cell, str) else "" for cell in row] for row in rows]
                        
                        # Analyze table characteristics
                        row_count = len(rows)
                        col_count = max(len(row) for row in rows) if rows else 0
                        
                        # Check if this is likely a real table or a content block
                        is_likely_table = True
                        is_content_block = False
                        
                        # Criteria 1: Check for educational content markers
                        flat_text = " ".join(" ".join(row) for row in table_data).lower()
                        for marker in content_block_markers:
                            if marker in flat_text:
                                is_content_block = True
                                is_likely_table = False
                                break
                        
                        # Criteria 2: Single-column tables are often content blocks
                        if col_count <= 1 and row_count >= 2:
                            is_likely_table = False
                            is_content_block = True
                        
                        # Criteria 3: Very small tables might be formatting artifacts
                        if row_count < 2 or col_count < 2:
                            is_likely_table = False
                        
                        # Criteria 4: Check for table header characteristics
                        has_header_row = False
                        if rows and len(rows[0]) > 1:
                            header_candidates = [cell for cell in rows[0] if isinstance(cell, str) and cell.strip()]
                            if header_candidates and all(len(cell) < 30 for cell in header_candidates):
                                has_header_row = True
                        
                        if has_header_row:
                            is_likely_table = True
                        
                        # Create table dictionary
                        table_dict = {
                            "page_num": page_num + 1,
                            "table_id": f"table_{page_num+1}_{i+1}",
                            "data": table_data,
                            "position": {}
                        }
                        
                        # Add position data if available
                        if hasattr(table, 'rect'):
                            table_dict["position"] = {
                                "x0": table.rect.x0,
                                "y0": table.rect.y0,
                                "x1": table.rect.x1,
                                "y1": table.rect.y1
                            }
                        elif hasattr(table, 'bbox'):
                            table_dict["position"] = {
                                "x0": table.bbox[0],
                                "y0": table.bbox[1],
                                "x1": table.bbox[2],
                                "y1": table.bbox[3]
                            }
                        
                        # Classify based on our analysis
                        if is_content_block:
                            table_dict["type"] = "content_block"
                            content_blocks.append(table_dict)
                        elif is_likely_table:
                            table_dict["type"] = "data_table"
                            tables.append(table_dict)
                        else:
                            # Uncertain cases - check additional criteria
                            has_grid_lines = self._check_for_table_lines(page, table_dict["position"])
                            if has_grid_lines:
                                table_dict["type"] = "data_table"
                                tables.append(table_dict)
                            else:
                                table_dict["type"] = "unknown_block"
                                content_blocks.append(table_dict)
                            
                    except Exception as e:
                        self.logger.warning(f"Could not process table {i+1} on page {page_num+1}: {str(e)}")
        
        self.logger.info(f"Classified {len(tables)} data tables and {len(content_blocks)} content/educational blocks")
        return tables, content_blocks

    def _check_for_table_lines(self, page, position):
        """Check if the area contains horizontal or vertical lines indicative of a table."""
        if not position:
            return False
            
        # Extract lines from the page
        lines = page.get_drawings()
        if not lines:
            return False
        
        # Define the table region
        table_region = (position.get("x0", 0), position.get("y0", 0), 
                        position.get("x1", 0), position.get("y1", 0))
        
        # Count horizontal and vertical lines in the region
        h_lines = 0
        v_lines = 0
        
        for line in lines:
            for item in line["items"]:
                if item[0] != "l":  # Not a line
                    continue
                    
                # Extract line coordinates
                x0, y0, x1, y1 = item[1]
                
                # Check if line is in the table region
                if (x0 >= table_region[0] and x1 <= table_region[2] and
                    y0 >= table_region[1] and y1 <= table_region[3]):
                    
                    # Determine if horizontal or vertical
                    if abs(y1 - y0) < 2:  # Horizontal line
                        h_lines += 1
                    elif abs(x1 - x0) < 2:  # Vertical line
                        v_lines += 1
        
        # If we have multiple horizontal or vertical lines, likely a table
        return h_lines >= 2 or v_lines >= 2
    
    # # Original table extraction 
    # def extract_tables_from_pdf(self, doc: fitz.Document) -> List[Dict[str, Any]]:
    #     """Extract tables from PDF using PyMuPDF's table detection capabilities."""
    #     tables = []
        
    #     for page_num, page in enumerate(doc):
    #         try:
    #             # Try to find tables on the page using PyMuPDF
    #             tab = page.find_tables()
    #             if tab.tables:
    #                 for i, table in enumerate(tab.tables):
    #                     try:
    #                         # Extract table content
    #                         table_data = []
    #                         rows = table.extract()
    #                         for row in rows:
    #                             table_data.append([cell.strip() if isinstance(cell, str) else "" for cell in row])
                            
    #                         # Create table object with safer rect access
    #                         table_dict = {
    #                             "page_num": page_num + 1,
    #                             "table_id": f"table_{page_num+1}_{i+1}",
    #                             "data": table_data,
    #                             "position": {}
    #                         }
                            
    #                         # Safely add position data if rect is available
    #                         if hasattr(table, 'rect'):
    #                             table_dict["position"] = {
    #                                 "x0": table.rect.x0,
    #                                 "y0": table.rect.y0,
    #                                 "x1": table.rect.x1,
    #                                 "y1": table.rect.y1
    #                             }
    #                         else:
    #                             # Fall back to bbox if available
    #                             if hasattr(table, 'bbox'):
    #                                 table_dict["position"] = {
    #                                     "x0": table.bbox[0],
    #                                     "y0": table.bbox[1],
    #                                     "x1": table.bbox[2],
    #                                     "y1": table.bbox[3]
    #                                 }
    #                             # Otherwise use placeholder position
    #                             else:
    #                                 table_dict["position"] = {
    #                                     "x0": 0,
    #                                     "y0": 0,
    #                                     "x1": 0,
    #                                     "y1": 0
    #                                 }
                            
    #                         tables.append(table_dict)
    #                     except Exception as e:
    #                         # Skip this table if extraction fails
    #                         print(f"Warning: Could not extract table {i+1} on page {page_num+1}: {str(e)}")
    #                         continue
    #         except Exception as e:
    #             # Skip table detection for this page if it fails
    #             print(f"Warning: Table detection failed for page {page_num+1}: {str(e)}")
    #             continue
        
    #     return tables
    
# Text Processing Methods

    # Create chunks with context
    def create_semantic_chunks(self, pdf_data: Dict[str, Any]) -> List[Document]:
        """Create semantically meaningful chunks with enhanced contextual awareness."""
        chunks = []
        all_headers = pdf_data["headers"]
        
        # If we have no headers, create fallback headers at regular intervals
        if not all_headers:
            # Create synthetic section markers every ~10 pages
            total_pages = pdf_data["total_pages"]
            pages_per_section = min(10, max(1, total_pages // 10))
            
            for i in range(0, total_pages, pages_per_section):
                section_num = i // pages_per_section + 1
                all_headers.append({
                    "title": f"Section {section_num}",
                    "page": i + 1,
                    "level": 1,
                    "position": 0,
                    "id": f"synthetic_{section_num}",
                    "type": "synthetic"
                })
        
        # Sort headers by page and position
        all_headers.sort(key=lambda h: (h["page"], h.get("position", 0)))
        
        # Process each page
        for page in pdf_data["pages"]:
            page_num = page["page_num"]
            page_text = page["text"]
            
            # Find headers for this page
            page_headers = page["headers"]
            page_tables = page["tables"]
            
            if not page_headers:
                # No headers on this page, use the most recent header before this page
                current_header = None
                for header in all_headers:
                    if header["page"] < page_num:
                        current_header = header
                    else:
                        break
                
                # Create semantic chunks for page with inherited header
                metadata = {
                    "book_title": pdf_data["book_title"],
                    "author": pdf_data.get("author", "Unknown"),
                    "filename": pdf_data["filename"],
                    "file_hash": pdf_data["file_hash"],
                    "page_num": page_num,
                    "current_header": current_header["title"] if current_header else "Unknown",
                    "header_type": current_header["type"] if current_header else None,
                    "chapter_num": current_header.get("id") if current_header else None,
                    "level": current_header.get("level", 1) if current_header else None
                }
                
                # Check if page has tables
                if page_tables:
                    # Special handling for pages with tables
                    # First chunk the text
                    if isinstance(self.semantic_splitter, SemanticTextSplitter):
                        page_chunks = self.semantic_splitter.create_documents([page_text], [metadata])
                    else:
                        page_chunks = self.semantic_splitter.create_documents([page_text], [metadata])
                    
                    # Add table data as separate chunks
                    for table in page_tables:
                        table_metadata = metadata.copy()
                        table_metadata["content_type"] = "table"
                        table_metadata["table_id"] = table["table_id"]
                        
                        # Format table as string
                        table_content = self._format_table_as_text(table["data"])
                        
                        table_chunk = Document(
                            page_content=table_content,
                            metadata=table_metadata
                        )
                        
                        chunks.append(table_chunk)
                    
                    # Add regular text chunks
                    chunks.extend(page_chunks)
                else:
                    # No tables, just chunk the text
                    if isinstance(self.semantic_splitter, SemanticTextSplitter):
                        page_chunks = self.semantic_splitter.create_documents([page_text], [metadata])
                    else:
                        page_chunks = self.semantic_splitter.create_documents([page_text], [metadata])
                    chunks.extend(page_chunks)
            else:
                # Page has headers, split text by header positions
                page_headers.sort(key=lambda h: h.get("position", 0))
                
                # Initial text before first header
                first_header_pos = page_headers[0].get("position", 0)
                if first_header_pos > 0:
                    # Ensure first_header_pos is an integer to avoid slice index errors
                    try:
                        if not isinstance(first_header_pos, int):
                            first_header_pos = int(first_header_pos)
                    except (ValueError, TypeError):
                        # If conversion fails, use a safe default
                        first_header_pos = 0
                    
                    # Safely extract pre-header text
                    if first_header_pos > 0 and first_header_pos < len(page_text):
                        pre_header_text = page_text[:first_header_pos]
                        if len(pre_header_text.strip()) > 100:  # Only process if meaningful content
                            # Find most recent header from previous pages
                            prev_header = None
                            for header in all_headers:
                                if header["page"] < page_num:
                                    prev_header = header
                                elif header["page"] > page_num:
                                    break
                                elif header.get("position", 0) >= first_header_pos:
                                    break
                            
                            pre_metadata = {
                                "book_title": pdf_data["book_title"],
                                "author": pdf_data.get("author", "Unknown"),
                                "filename": pdf_data["filename"],
                                "file_hash": pdf_data["file_hash"],
                                "page_num": page_num,
                                "chunk_id": f"{page_num}-pre",
                                "current_header": prev_header["title"] if prev_header else "Unknown",
                                "header_type": prev_header["type"] if prev_header else None,
                                "chapter_num": prev_header.get("id") if prev_header else None,
                                "level": prev_header.get("level", 1) if prev_header else None
                            }
                            
                            if isinstance(self.semantic_splitter, SemanticTextSplitter):
                                pre_chunks = self.semantic_splitter.create_documents([pre_header_text], [pre_metadata])
                            else:
                                pre_chunks = self.semantic_splitter.create_documents([pre_header_text], [pre_metadata])
                            chunks.extend(pre_chunks)
                
                # Process text for each header section
                for i, header in enumerate(page_headers):
                    # Ensure position is an integer
                    try:
                        start_pos = header.get("position", 0)
                        if not isinstance(start_pos, int):
                            start_pos = int(start_pos)
                    except (ValueError, TypeError):
                        # If conversion fails, use a safe default
                        start_pos = 0
                    
                    # Determine end position (next header or end of page)
                    if i < len(page_headers) - 1:
                        try:
                            end_pos = page_headers[i+1].get("position", len(page_text))
                            if not isinstance(end_pos, int):
                                end_pos = int(end_pos)
                        except (ValueError, TypeError):
                            # If conversion fails, use end of text
                            end_pos = len(page_text)
                    else:
                        end_pos = len(page_text)
                    
                    # Safely extract section text
                    if 0 <= start_pos < len(page_text) and start_pos < end_pos <= len(page_text):
                        section_text = page_text[start_pos:end_pos]
                    else:
                        # If indices are invalid, use a safe range
                        safe_start = max(0, min(start_pos, len(page_text)-1))
                        safe_end = max(safe_start+1, min(end_pos, len(page_text)))
                        section_text = page_text[safe_start:safe_end]
                    
                    # Create metadata for this section
                    section_metadata = {
                        "book_title": pdf_data["book_title"],
                        "author": pdf_data.get("author", "Unknown"),
                        "filename": pdf_data["filename"],
                        "file_hash": pdf_data["file_hash"],
                        "page_num": page_num,
                        "chunk_id": f"{page_num}-{i}",
                        "current_header": header["title"],
                        "header_type": header["type"],
                        "chapter_num": header.get("id"),
                        "level": header.get("level", 1)
                    }
                    
                    # Check for tables in this section
                    section_tables = []
                    for table in page_tables:
                        table_y = table["position"].get("y0", 0)
                        # Try to convert table_y to float if it's not already a number
                        if not isinstance(table_y, (int, float)):
                            try:
                                table_y = float(table_y)
                            except (ValueError, TypeError):
                                table_y = 0
                                
                        # Estimate if table is in this section
                        # This is approximate since we don't have y-coordinates for header positions
                        if i < len(page_headers) - 1:
                            if i == 0 or table_y >= start_pos:
                                section_tables.append(table)
                        else:
                            section_tables.append(table)
                    
                    if section_tables:
                        # Create chunks from text
                        if isinstance(self.semantic_splitter, SemanticTextSplitter):
                            section_chunks = self.semantic_splitter.create_documents([section_text], [section_metadata])
                        else:
                            section_chunks = self.semantic_splitter.create_documents([section_text], [section_metadata])
                        
                        # Add tables as separate chunks
                        for table in section_tables:
                            table_metadata = section_metadata.copy()
                            table_metadata["content_type"] = "table"
                            table_metadata["table_id"] = table["table_id"]
                            
                            # Format table as string
                            table_content = self._format_table_as_text(table["data"])
                            
                            table_chunk = Document(
                                page_content=table_content,
                                metadata=table_metadata
                            )
                            
                            chunks.append(table_chunk)
                        
                        # Add regular text chunks
                        chunks.extend(section_chunks)
                    else:
                        # No tables, just chunk the text
                        if isinstance(self.semantic_splitter, SemanticTextSplitter):
                            section_chunks = self.semantic_splitter.create_documents([section_text], [section_metadata])
                        else:
                            section_chunks = self.semantic_splitter.create_documents([section_text], [section_metadata])
                        chunks.extend(section_chunks)
        
        return chunks
    
    # Format tables as text
    def _format_table_as_text(self, table_data: List[List[str]]) -> str:
        """Format table data as readable text."""
        if not table_data:
            return "Empty table"
            
        result = "TABLE:\n"
        
        # Calculate column widths
        col_widths = [0] * len(table_data[0])
        for row in table_data:
            for i, cell in enumerate(row):
                if i < len(col_widths):  # Ensure index is in range
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Format as plain text table with separators
        for i, row in enumerate(table_data):
            row_str = "| "
            for j, cell in enumerate(row):
                if j < len(col_widths):  # Ensure index is in range
                    cell_text = str(cell).ljust(col_widths[j])
                    row_str += cell_text + " | "
            result += row_str + "\n"
            
            # Add separator after header row
            if i == 0:
                sep_row = "| "
                for j, width in enumerate(col_widths):
                    sep_row += "-" * width + " | "
                result += sep_row + "\n"
        
        return result
    
# Utility Methods

    #

    
    def extract_structure_with_unstructured(self, pdf_path: str) -> Tuple[List[Dict[str, Any]], List[Document]]:
        """Use the 'unstructured' library to extract document structure and content."""
        try:
            # This requires installing the unstructured library
            from unstructured.partition.pdf import partition_pdf
            
            # Process the PDF
            elements = partition_pdf(
                pdf_path,
                extract_images_in_pdf=False,
                infer_table_structure=True,
                chunking_strategy="by_title"
            )
            
            # Convert to our structure format
            headers = []
            chunks = []
            
            current_page = 1
            for element in elements:
                # Extract headers
                if element.category == "Title" or element.category == "Header":
                    # Determine level based on font size or metadata
                    level = 1
                    if hasattr(element, "metadata") and "font_size" in element.metadata:
                        # Higher font size often means higher level header
                        font_size = element.metadata["font_size"]
                        if font_size > 16:
                            level = 1
                        elif font_size > 13:
                            level = 2
                        else:
                            level = 3
                    
                    # Get page number if available
                    page = current_page
                    if hasattr(element, "metadata") and "page_number" in element.metadata:
                        page = element.metadata["page_number"]
                        current_page = page
                    
                    headers.append({
                        "title": str(element),
                        "page": page,
                        "level": level,
                        "position": 0,
                        "id": f"header_{len(headers)}",
                        "type": "unstructured_header"
                    })
                
                # Create chunks with metadata
                chunks.append(Document(
                    page_content=str(element),
                    metadata={
                        "page": current_page,
                        "category": element.category,
                        "element_id": f"elem_{len(chunks)}"
                    }
                ))
            
            return headers, chunks
            
        except ImportError:
            self.logger.warning("'unstructured' library not available. Using fallback methods.")
            return [], []

    def extract_basic_structure_with_pypdf2(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract basic document structure using PyPDF2."""
        import PyPDF2
        
        headers = []
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Get outline/bookmarks
            outlines = reader.outline
            
            if outlines:
                # Process outline entries
                def process_outline_item(item, level=1):
                    item_headers = []
                    
                    if isinstance(item, list):
                        # It's a list of items
                        for subitem in item:
                            item_headers.extend(process_outline_item(subitem, level))
                    elif isinstance(item, PyPDF2.generic.Destination):
                        # It's an actual outline item
                        title = item['/Title']
                        page_num = reader.get_destination_page_number(item)
                        
                        item_headers.append({
                            "title": title,
                            "page": page_num + 1,  # 1-indexed page numbers
                            "level": level,
                            "position": 0,
                            "id": f"outline_{len(headers)}",
                            "type": "pdf_outline"
                        })
                    elif isinstance(item, dict) and '/Title' in item:
                        # Dictionary with a title and potentially children
                        title = item['/Title']
                        try:
                            page_num = reader.get_destination_page_number(item)
                        except:
                            page_num = 0
                        
                        item_headers.append({
                            "title": title,
                            "page": page_num + 1,  # 1-indexed page numbers
                            "level": level,
                            "position": 0,
                            "id": f"outline_{len(headers)}",
                            "type": "pdf_outline"
                        })
                        
                        # Process any children with increased level
                        if '/Kids' in item and item['/Kids']:
                            for kid in item['/Kids']:
                                item_headers.extend(process_outline_item(kid, level+1))
                    
                    return item_headers
                
                headers = process_outline_item(outlines)
            
            return headers
        
    def detect_structure_with_vector_db(self, doc: fitz.Document, embedding_model) -> List[Dict[str, Any]]:
        """Use vector similarity to detect document structure."""
        # Extract text from each page
        page_texts = [page.get_text() for page in doc]
        
        # Create embeddings for each page
        page_embeddings = embedding_model.encode(page_texts)
        
        # Find potential section boundaries based on semantic shifts
        section_boundaries = []
        
        for i in range(1, len(page_embeddings)):
            # Calculate similarity between consecutive pages
            similarity = np.dot(page_embeddings[i-1], page_embeddings[i]) / (
                np.linalg.norm(page_embeddings[i-1]) * np.linalg.norm(page_embeddings[i])
            )
            
            # If similarity drops, it might indicate a section boundary
            if similarity < 0.7:  # Threshold for significant change
                # Look for potential header text at the top of the page
                page_text = page_texts[i]
                lines = page_text.split('\n')
                
                for line_idx, line in enumerate(lines[:5]):  # Check first 5 lines
                    if 3 < len(line.strip()) < 100:  # Reasonable header length
                        # Verify it looks like a header (e.g., starts with number or capital letter)
                        if re.match(r'^[A-Z0-9]', line.strip()):
                            section_boundaries.append({
                                "title": line.strip(),
                                "page": i + 1,
                                "level": 1,
                                "position": line_idx,
                                "id": f"semantic_boundary_{i}",
                                "type": "semantic_section"
                            })
                            break
        
        return section_boundaries


    def _extract_educational_book_structure(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """Extract structure from various types of educational books (textbooks, exam guides, notes, etc.)."""
        headers = []
        
        # First attempt: Enhanced TOC extraction (works across different educational formats)
        toc_headers = self._extract_from_toc_pages(doc)
        if toc_headers and len(toc_headers) >= 3:
            self.logger.info(f"Using {len(toc_headers)} entries from book TOC")
            return toc_headers
        
        # Second attempt: Look for chapter/section headers with broader patterns
        # Common in various educational books
        header_patterns = [
            # Traditional chapter/section numbering
            r'^\s*Chapter\s+(\d+)\s*[:\-]?\s*(.*?)$',
            r'^\s*(\d+)\.\s+(.*?)$',
            r'^\s*(\d+\.\d+)\s+(.*?)$',
            # Unit/module/lesson patterns (textbooks)
            r'^\s*Unit\s+(\d+)\s*[:\-]?\s*(.*?)$',
            r'^\s*Lesson\s+(\d+)\s*[:\-]?\s*(.*?)$',
            r'^\s*Module\s+(\d+)\s*[:\-]?\s*(.*?)$',
            # Topic patterns (common in notes and exam guides)
            r'^\s*Topic\s+(\d+)\s*[:\-]?\s*(.*?)$',
            r'^\s*Section\s+(\d+)\s*[:\-]?\s*(.*?)$',
            # Exam-specific patterns
            r'^\s*Paper\s+(\d+)\s*[:\-]?\s*(.*?)$',
            r'^\s*Test\s+(\d+)\s*[:\-]?\s*(.*?)$',
            # Custom format detection (covers more cases)
            r'^\s*([A-Z][A-Z0-9\.\-]+)\s+(.*?)$',  # Alphanumeric codes like "A1.2 Topic"
        ]
        
        # Analyze a sampling of pages for header detection
        sample_step = max(1, len(doc) // 40)  # Sample about 40 pages throughout document
        pages_to_check = list(range(0, len(doc), sample_step))
        # Always check first 10 pages as they often contain important structure
        for i in range(min(10, len(doc))):
            if i not in pages_to_check:
                pages_to_check.append(i)
        
        for page_num in sorted(pages_to_check):
            page = doc[page_num]
            page_text = page.get_text()
            lines = page_text.split('\n')
            
            for line_idx, line in enumerate(lines):
                line = line.strip()
                # Skip empty or very short lines
                if not line or len(line) < 3:
                    continue
                
                # Check for header patterns
                for pattern in header_patterns:
                    match = re.match(pattern, line)
                    if match:
                        identifier = match.group(1)
                        title = match.group(2).strip() if len(match.groups()) > 1 else ""
                        
                        # Validate the title
                        if not title or len(title) > 150:
                            continue
                        
                        # Check for formatting indicators (headers often have special formatting)
                        is_likely_header = (
                            self._is_text_centered(page, line) or
                            self._is_larger_font(page, line) or
                            self._is_bold_text(page, line) or
                            line.isupper() or  # ALL CAPS
                            len(lines) > line_idx + 1 and not lines[line_idx + 1].strip()  # Followed by blank line
                        )
                        
                        if is_likely_header:
                            # Determine hierarchy level based on pattern
                            level = 1
                            if "." in identifier:
                                level = identifier.count(".") + 1
                            
                            headers.append({
                                "title": f"{identifier} {title}".strip(),
                                "page": page_num + 1,
                                "level": level,
                                "position": page_text.find(line),
                                "id": f"h_{identifier.replace('.', '_')}",
                                "type": "educational_header"
                            })
        
        # If we found enough structural headers, return them
        if len(headers) >= 3:
            self.logger.info(f"Extracted {len(headers)} structural headers from educational content")
            return self._deduplicate_headers(headers)
        
        # Fallback: Look for emphasized text that might be headers
        # This works well for notes and guides that don't follow strict numbering
        headers_from_emphasis = self._extract_headers_from_emphasis(doc)
        if headers_from_emphasis and len(headers_from_emphasis) >= 3:
            self.logger.info(f"Using {len(headers_from_emphasis)} headers detected from text emphasis")
            return headers_from_emphasis
        
        # Final fallback: Create synthetic structure based on content
        self.logger.info("Using synthetic structure markers")
        return self._create_synthetic_structure(doc)

    def _extract_headers_from_emphasis(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """Extract headers based on text emphasis (bold, large font, etc.) - useful for notes and guides."""
        headers = []
        sample_step = max(1, len(doc) // 30)
        
        for page_num in range(0, len(doc), sample_step):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block_idx, block in enumerate(blocks):
                if "lines" not in block:
                    continue
                    
                for line_idx, line in enumerate(block["lines"]):
                    if not line.get("spans"):
                        continue
                        
                    line_text = "".join([span["text"] for span in line["spans"]])
                    line_text = line_text.strip()
                    
                    # Skip very short or empty lines
                    if not line_text or len(line_text) < 3 or len(line_text) > 100:
                        continue
                    
                    # Check for emphasis indicators
                    is_emphasized = False
                    for span in line["spans"]:
                        # Check for emphasized text formatting
                        is_bold = span.get("flags", 0) & 2**4 > 0
                        is_large = span.get("size", 0) > 12  # Larger than typical body text
                        
                        if is_bold or is_large:
                            is_emphasized = True
                            break
                    
                    if is_emphasized:
                        # Check if this looks like a header (not just random emphasized text)
                        if not re.search(r'^(note|example|warning|caution|remember):', line_text.lower()):
                            headers.append({
                                "title": line_text,
                                "page": page_num + 1,
                                "level": 1,  # Default level
                                "position": block["bbox"][1],  # Y position
                                "id": f"emph_{page_num}_{block_idx}_{line_idx}",
                                "type": "emphasis_header"
                            })
        
        return self._deduplicate_headers(headers)

    def _create_synthetic_structure(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """Create synthetic structure when no clear headers are detected - fallback method."""
        headers = []
        total_pages = len(doc)
        
        # For educational materials, approximate section length
        avg_section_length = min(10, max(5, total_pages // 15))
        
        for i in range(0, total_pages, avg_section_length):
            page_num = min(i, total_pages - 1)
            page = doc[page_num]
            page_text = page.get_text()
            
            # Try to find a meaningful title on this page
            title = f"Section {i // avg_section_length + 1}"
            
            # Look for potential title text in first few lines
            lines = page_text.split('\n')
            for line in lines[:10]:
                line = line.strip()
                if line and 5 <= len(line) <= 80:
                    # Avoid page numbers, headers/footers
                    if not re.match(r'^\d+$|page|^\s*section\s*\d+\s*$', line.lower()):
                        title = line
                        break
            
            headers.append({
                "title": title,
                "page": page_num + 1,
                "level": 1,
                "position": 0,
                "id": f"synth_{i // avg_section_length + 1}",
                "type": "synthetic"
            })
        
        return headers

    def _is_bold_text(self, page, text):
        """Check if text appears to be bold."""
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                line_text = "".join([span["text"] for span in line["spans"]])
                if text in line_text:
                    for span in line["spans"]:
                        if span.get("flags", 0) & 2**4 > 0:  # Bold flag
                            return True
        return False
    
    def _is_educational_book(self, doc: fitz.Document) -> bool:
        """Detect if document appears to be an educational book (textbook, guide, notes, etc.)."""
        # Check a sample of pages for educational indicators
        education_indicators = 0
        max_pages_to_check = min(15, len(doc))
        
        # Educational content markers (broader set)
        educational_terms = [
            'chapter', 'section', 'unit', 'module', 'lesson', 'topic',
            'exercise', 'practice', 'example', 'question', 'answer', 'solution',
            'test', 'exam', 'quiz', 'assessment', 'problem', 'study',
            'lecture', 'course', 'curriculum', 'syllabus', 'reference'
        ]
        
        # Structure indicators
        structure_terms = [
            'contents', 'index', 'table of contents', 'toc', 'chapters',
            'sections', 'glossary', 'appendix', 'bibliography'
        ]
        
        for page_num in range(max_pages_to_check):
            page = doc[page_num]
            page_text = page.get_text().lower()
            
            # Check for educational content
            for term in educational_terms:
                if re.search(r'\b' + term + r'\b', page_text):
                    education_indicators += 1
                    break
                    
            # Check for structural elements
            for term in structure_terms:
                if re.search(r'\b' + term + r'\b', page_text):
                    education_indicators += 2
                    break
                    
            # Check for numbered patterns (common in educational materials)
            if re.search(r'\b\d+\.\d*\s+[A-Z]', page_text):
                education_indicators += 1
                
            # Check for instructional patterns
            if any(term in page_text for term in ['try this', 'remember', 'note:', 'important']):
                education_indicators += 1
        
        # Check for TOC/index page
        has_index_page = False
        for page_num in range(min(10, len(doc))):
            page = doc[page_num]
            page_text = page.get_text()
            
            # Look for TOC header and structured content
            if (re.search(r'^\s*CONTENTS?\s*$', page_text, re.MULTILINE | re.IGNORECASE) or
                re.search(r'^\s*INDEX\s*$', page_text, re.MULTILINE | re.IGNORECASE)):
                
                # Check for structured entries 
                entry_count = len(re.findall(r'^\s*(?:\d+\.|\w+\s+\d+)[^\n]+\d+\s*$', page_text, re.MULTILINE))
                if entry_count >= 3:
                    has_index_page = True
                    education_indicators += 5
                    break
        
        # If score is high enough or has a clear index page, it's an educational book
        return education_indicators >= 4 or has_index_page


    

    def _analyze_document_typography(self, doc: fitz.Document) -> Dict:
        """Analyze document typography to identify text hierarchy patterns with early stopping."""
        font_sizes = {}
        font_styles = {}
        
        # Only sample from representative pages to improve performance
        # Check first few pages, some middle pages, and last few pages
        sample_pages = []
        total_pages = len(doc)
        
        # First few pages
        for i in range(min(5, total_pages)):
            sample_pages.append(i)
        
        # Some middle pages
        if total_pages > 10:
            middle_start = total_pages // 2 - 2
            for i in range(middle_start, min(middle_start + 4, total_pages)):
                sample_pages.append(i)
        
        # Last few pages
        if total_pages > 5:
            for i in range(max(0, total_pages - 5), total_pages):
                sample_pages.append(i)
        
        # Remove duplicates
        sample_pages = sorted(set(sample_pages))
        
        # Analyze font statistics on sample pages
        for page_num in sample_pages:
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    # Skip if this looks like an image caption or footnote
                    is_small_text = all(span["size"] < 8 for span in line["spans"] if "size" in span)
                    if is_small_text:
                        continue
                        
                    for span in line["spans"]:
                        font_size = span["size"]
                        is_bold = span["flags"] & 2**4 > 0
                        text = span["text"].strip()
                        
                        # Only consider spans with meaningful text
                        if len(text) > 1:
                            key = (font_size, is_bold)
                            font_sizes[font_size] = font_sizes.get(font_size, 0) + len(text)
                            font_styles[key] = font_styles.get(key, 0) + len(text)
        
        # Find the most common font size (likely the main text)
        most_common_size = 12  # Default
        if font_sizes:
            most_common_size = max(font_sizes.items(), key=lambda x: x[1])[0]
        
        # Find font sizes larger than the most common (potential headers)
        larger_sizes = [size for size in font_sizes.keys() if size > most_common_size * 1.1]
        larger_sizes.sort(reverse=True)  # Sort from largest to smallest
        
        # Analyze potential header styles
        header_styles = []
        for (size, is_bold), count in font_styles.items():
            if size > most_common_size * 1.1 or (is_bold and size >= most_common_size):
                header_styles.append({
                    "size": size,
                    "is_bold": is_bold,
                    "count": count
                })
        
        # Sort by importance (size, then boldness)
        header_styles.sort(key=lambda x: (x["size"], x["is_bold"]), reverse=True)
        
        return {
            "most_common_size": most_common_size,
            "larger_sizes": larger_sizes,
            "header_styles": header_styles
        }
    
    

    def _identify_toc_pages(self, doc: fitz.Document) -> List[int]:
        """Identify pages that are likely to be table of contents."""
        potential_toc_pages = []
        
        # Check early pages in the document (TOC is usually at the beginning)
        max_pages_to_check = min(20, len(doc))
        
        for page_num in range(max_pages_to_check):
            page = doc[page_num]
            page_text = page.get_text()
            
            # Look for TOC indicators
            toc_indicators = [
                "contents", "table of contents", "index", "chapters", 
                "sections", "toc", "page", "chapter", "section"
            ]
            
            # Score this page for TOC likelihood
            toc_score = 0
            
            # Check for TOC title
            for indicator in toc_indicators[:5]:  # First 5 are stronger indicators
                if re.search(f"\\b{indicator}\\b", page_text, re.IGNORECASE):
                    toc_score += 3
            
            # Check for remaining indicators
            for indicator in toc_indicators[5:]:
                if re.search(f"\\b{indicator}\\b", page_text, re.IGNORECASE):
                    toc_score += 1
                    
            # Check for ellipsis patterns common in TOCs
            ellipsis_count = len(re.findall(r'\.{3,}', page_text))
            toc_score += min(ellipsis_count, 5)  # Cap at 5 points
            
            # Check for page number patterns at line ends
            page_num_patterns = len(re.findall(r'\.+\s*\d+\s*$', page_text, re.MULTILINE))
            toc_score += min(page_num_patterns, 5)  # Cap at 5 points
            
            # If score is high enough, consider it a TOC page
            if toc_score >= 5:
                potential_toc_pages.append(page_num)
                
                # Check next page too (TOCs often span multiple pages)
                if page_num < len(doc) - 1:
                    potential_toc_pages.append(page_num + 1)
                    
                    # For longer TOCs, check a few more pages
                    if toc_score > 10 and page_num < len(doc) - 2:
                        potential_toc_pages.append(page_num + 2)
        
        return sorted(set(potential_toc_pages))  # Remove duplicates and sort

    def _get_indentation(self, page, text_to_find):
        """Get the indentation level of text on the page in points."""
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                line_text = "".join([span["text"] for span in line["spans"]])
                
                # If the text is in this line, get the x-position (indentation)
                if text_to_find in line_text:
                    return line["spans"][0]["origin"][0]  # x-coordinate of first span
        
        return 0  # Default if not found


    def _build_header_hierarchy(self, headers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build a hierarchical structure from detected headers."""
        if not headers:
            return []
            
        # If these are already from TOC with levels, we can use them directly
        if all("level" in h for h in headers):
            return headers
            
        # Try to determine hierarchy signals
        has_numbered_sections = any(re.match(r'^\d+\.\d+', h["title"]) for h in headers)
        has_chapter_markers = any(re.match(r'^(chapter|section)\s+\d+', h["title"].lower()) for h in headers)
        
        # Get font size statistics if available
        font_sizes = sorted(set(h["font_size"] for h in headers if "font_size" in h), reverse=True)
        
        for header in headers:
            # Default level
            level = 1
            
            # If header already has level from TOC, keep it
            if "level" in header:
                continue
                
            # Try to determine level
            if "type" in header and header["type"] == "toc_entry":
                # Already processed in TOC extraction
                continue
                
            if has_numbered_sections:
                # Check for numbering pattern (1.2.3)
                match = re.match(r'^(\d+(\.\d+)*)', header["title"])
                if match:
                    section_number = match.group(1)
                    level = section_number.count('.') + 1
                    header["section_number"] = section_number
            
            elif has_chapter_markers:
                # Check for chapter/section markers
                if re.match(r'^chapter\s+\d+', header["title"].lower()):
                    level = 1
                elif re.match(r'^section\s+\d+', header["title"].lower()):
                    level = 2
                    
            elif "font_size" in header and font_sizes:
                # Use font size to determine level
                try:
                    level = font_sizes.index(header["font_size"]) + 1
                except ValueError:
                    level = len(font_sizes)  # If not found, assume lowest level
                    
                # Adjust level based on bold formatting
                if header.get("is_bold", False) and level > 1:
                    level -= 0.5  # Bold text of the same size often indicates higher importance
            
            # Set the level
            header["level"] = level
        
        return headers
   
    
    def _deduplicate_headers(self, headers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate headers that may be detected by multiple methods."""
        # Sort headers by page and position
        headers.sort(key=lambda h: (h["page"], h.get("position", 0)))
        
        unique_headers = []
        seen_on_page = {}  # Track headers by page and content
        
        for header in headers:
            page = header["page"]
            title = header["title"].lower()
            position = header.get("position", 0)
            
            # Create a unique key for this header
            header_key = f"{page}_{title}"
            
            # Check if we've seen this header on this page already
            if header_key in seen_on_page:
                existing_pos = seen_on_page[header_key]["position"]
                # If positions are close, consider it a duplicate
                if abs(existing_pos - position) < 100:
                    continue
            
            # Add to unique list
            unique_headers.append(header)
            seen_on_page[header_key] = header
            
        return unique_headers
    
    def _organize_header_hierarchy(self, headers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Organize headers into a hierarchical structure if possible."""
        # Sort by page and position
        headers.sort(key=lambda h: (h["page"], h.get("position", 0)))
        
        # Try to detect header levels (chapter, section, subsection)
        # Group headers by font size
        size_groups = {}
        for header in headers:
            if "font_size" in header:
                size = header["font_size"]
                if size not in size_groups:
                    size_groups[size] = []
                size_groups[size].append(header)
        
        # Sort sizes from largest to smallest
        sorted_sizes = sorted(size_groups.keys(), reverse=True)
        
        # Assign hierarchy levels based on font size
        for i, header in enumerate(headers):
            if "font_size" in header:
                # Find its position in the size hierarchy
                level = sorted_sizes.index(header["font_size"]) + 1 if header["font_size"] in sorted_sizes else None
                header["level"] = level
            
            # Also try to infer from the id pattern (1, 1.1, 1.1.1)
            if header["type"] == "pattern_detected" and isinstance(header["id"], str):
                dots = header["id"].count(".")
                if dots >= 0:
                    # Override with pattern-based level if available
                    header["level"] = dots + 1
                    
        return headers
    
    
    
    def _detect_headers_by_content_patterns(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """Detect headers based on content patterns when typography and TOC methods fail."""
        headers = []
        
        # Common header patterns
        header_patterns = [
            (r'^chapter\s+(\d+)[:\s]+(.+)$', 1),  # Chapter headings
            (r'^(\d+(?:\.\d+)*)\s+([A-Z].+)$', None),  # Numbered sections (level from numbering)
            (r'^([A-Z][^.!?]{10,60})$', 2)  # Capitalized phrases of reasonable length
        ]
        
        # Sample pages throughout the document
        total_pages = len(doc)
        sample_step = max(1, total_pages // 20)  # Sample about 20 pages
        
        for page_num in range(0, total_pages, sample_step):
            page = doc[page_num]
            text_lines = page.get_text("text").split('\n')
            
            for i, line in enumerate(text_lines):
                line = line.strip()
                if not line or len(line) < 4:
                    continue
                    
                # Check each pattern
                for pattern, fixed_level in header_patterns:
                    match = re.match(pattern, line, re.IGNORECASE)
                    if match:
                        if fixed_level is not None:
                            level = fixed_level
                        else:
                            # Determine level from numbering
                            if '.' in match.group(1):
                                level = match.group(1).count('.') + 1
                            else:
                                level = 1
                        
                        headers.append({
                            "title": line,
                            "page": page_num + 1,
                            "position": page.get_text().find(line),
                            "level": level,
                            "id": f"content_{page_num}_{i}",
                            "type": "content_pattern"
                        })
                        break
        
        return headers

    def get_pdf_hash(self, pdf_path: str) -> str:
        """Generate a hash for a PDF file to identify it uniquely."""
        with open(pdf_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    
    
    
    
    
    def evaluate_structure_extraction(self, pdf_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the quality of structure extraction for a document."""
        headers = pdf_data.get("headers", [])
        
        # Calculate basic metrics
        metrics = {
            "title": pdf_data.get("book_title", "Unknown"),
            "total_pages": pdf_data.get("total_pages", 0),
            "header_count": len(headers),
            "headers_per_page": len(headers) / max(1, pdf_data.get("total_pages", 1)),
            "has_hierarchy": any(h.get("level", 1) > 1 for h in headers),
            "extraction_methods": list(set(h.get("type", "unknown") for h in headers)),
            "pages_with_headers": len(set(h.get("page", 0) for h in headers)),
            "table_count": len(pdf_data.get("tables", [])),
            "content_block_count": len(pdf_data.get("content_blocks", []))
        }
        
        # Calculate coverage (percentage of pages with headers)
        if metrics["total_pages"] > 0:
            metrics["page_coverage"] = metrics["pages_with_headers"] / metrics["total_pages"]
        else:
            metrics["page_coverage"] = 0
        
        # Estimate quality score (higher is better)
        quality_score = 0
        
        # Factors to consider in quality score
        if metrics["header_count"] > 0:
            quality_score += min(5, metrics["header_count"] / 10)  # Up to 5 points for number of headers
        
        if metrics["has_hierarchy"]:
            quality_score += 2  # 2 points for having hierarchy
        
        if "toc_entry" in metrics["extraction_methods"] or "textbook_toc" in metrics["extraction_methods"]:
            quality_score += 3  # 3 points for having TOC entries
        
        quality_score += min(5, metrics["page_coverage"] * 10)  # Up to 5 points for page coverage
        
        # Check for real tables (not just formatting elements)
        if metrics["table_count"] > 0:
            quality_score += min(2, metrics["table_count"] / 5)  # Up to 2 points for tables
        
        # Check if we have proper content blocks
        if metrics["content_block_count"] > 0:
            quality_score += 1  # 1 point for detecting content blocks
        
        metrics["quality_score"] = round(quality_score, 1)
        metrics["extraction_quality"] = "Good" if quality_score >= 8 else "Medium" if quality_score >= 5 else "Poor"
        
        return metrics
    
    
    
    def add_chunks_to_vector_db(self, chunks: List[Document], vector_db):
        """Add chunks directly to vector database with efficient batching."""
        if not chunks:
            return
            
        batch_size = 100  # Process in manageable batches
        total_chunks = len(chunks)
        
        print(f"Adding {total_chunks} chunks to vector database...")
        
        for i in range(0, total_chunks, batch_size):
            end_idx = min(i + batch_size, total_chunks)
            batch = chunks[i:end_idx]
            
            # Extract text and metadata
            texts = [c.page_content for c in batch]
            metadatas = [c.metadata for c in batch]
            
            # Add to vector store
            vector_db.add_texts(texts=texts, metadatas=metadatas)
            
            print(f"Indexed batch: {i//batch_size + 1} ({end_idx}/{total_chunks})")
            
        print(f"✓ Successfully indexed all {total_chunks} chunks")

    
    
    
    def save_processed_pdf(self, pdf_data: Dict[str, Any], chunks: List[Document]) -> None:
        """Save processed PDF data and chunks to disk with improved metadata."""
        # Create directory structure
        book_dir = os.path.join(self.config.processed_dir, pdf_data["file_hash"])
        os.makedirs(book_dir, exist_ok=True)
        
        # Save PDF metadata (without the large text content)
        metadata_path = os.path.join(book_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            # Remove the large 'pages' entry before saving
            metadata = pdf_data.copy()
            metadata.pop("pages", None)
            json.dump(metadata, f, indent=2)
        
        # Save chunks with enhanced metadata
        chunks_data = []
        for chunk in chunks:
            chunks_data.append({
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            })
        
        chunks_path = os.path.join(book_dir, "chunks.json")
        with open(chunks_path, "w") as f:
            json.dump(chunks_data, f, indent=2)
            
        # Save a structured TOC based on headers
        toc_path = os.path.join(book_dir, "toc.json")
        toc_data = self._generate_toc_from_headers(pdf_data["headers"])
        with open(toc_path, "w") as f:
            json.dump(toc_data, f, indent=2)

    def _generate_toc_from_headers(self, headers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate a structured table of contents from detected headers."""
        # Sort headers by page number
        headers = sorted(headers, key=lambda h: h["page"])
        
        # Convert to TOC format
        toc = []
        for header in headers:
            toc_entry = {
                "title": header["title"],
                "page": header["page"],
                "level": header.get("level", 1),  # Default to level 1 if not specified
                "id": header.get("id", "")
            }
            toc.append(toc_entry)
        
        return toc
            
    def load_processed_pdf(self, file_hash: str) -> Tuple[Dict[str, Any], List[Document]]:
        """Load previously processed PDF data with enhanced metadata."""
        book_dir = os.path.join(self.config.processed_dir, file_hash)
        
        # Load metadata
        metadata_path = os.path.join(book_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            pdf_data = json.load(f)
        
        # Load chunks
        chunks_path = os.path.join(book_dir, "chunks.json")
        with open(chunks_path, "r") as f:
            chunks_data = json.load(f)
        
        # Convert back to Document objects
        chunks = []
        for chunk_data in chunks_data:
            chunk = Document(
                page_content=chunk_data["page_content"],
                metadata=chunk_data["metadata"]
            )
            chunks.append(chunk)
        
        return pdf_data, chunks

    
    
    

    

    def create_intelligent_search_index(self, processed_books: Dict[str, Dict], output_dir: str = None):
        """Create an intelligent search index from processed books data."""
        if output_dir is None:
            output_dir = os.path.join(self.config.processed_dir, "search_index")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect all chunks across books
        all_chunks = []
        chunk_to_book = {}
        
        for book_title, book_data in processed_books.items():
            chunks = book_data["chunks"]
            all_chunks.extend(chunks)
            
            # Track which book each chunk belongs to
            for chunk in chunks:
                chunk_id = f"{chunk.metadata.get('file_hash', '')}_{chunk.metadata.get('chunk_id', '')}"
                chunk_to_book[chunk_id] = book_title
        
        # Save mapping information
        mapping_path = os.path.join(output_dir, "chunk_mapping.json")
        with open(mapping_path, "w") as f:
            json.dump(chunk_to_book, f, indent=2)
        
        print(f"Created intelligent search index with {len(all_chunks)} total chunks")
        return all_chunks

    def find_similar_chunks(self, query_text: str, top_k: int = 5, embedding_model=None):
        """Find chunks similar to a query text using vector similarity."""
        if not embedding_model and self.embedding_model is None:
            # Initialize a model if none provided
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        elif not embedding_model:
            embedding_model = self.embedding_model
        
        # Check if we're using HuggingFaceEmbeddings or SentenceTransformer
        if hasattr(embedding_model, 'embed_query'):
            # Using LangChain's HuggingFaceEmbeddings
            query_embedding = embedding_model.embed_query(query_text)
        elif hasattr(embedding_model, 'encode'):
            # Using SentenceTransformer directly
            query_embedding = embedding_model.encode(query_text)
        else:
            raise ValueError("Unsupported embedding model type")
        
        # Search in vector database
        search_results = self.vector_db.similarity_search_by_vector(
            query_embedding, 
            k=top_k
        )
        # Return results
        return search_results
    
    def find_similar_chunks_with_reranking(self, query_text: str, chunks: List[Document], 
                                      top_k: int = 5, reranker=None, embedding_model=None):
        """Find chunks similar to a query text with optional reranking."""
        # First get top-N*2 results (get more than needed for reranking)
        initial_results = self.find_similar_chunks(query_text, chunks, top_k=top_k*2, embedding_model=embedding_model)
        
        # If no reranker, return initial results
        if reranker is None:
            return initial_results[:top_k]
        
        # Apply reranking
        reranked_results = reranker(query_text, initial_results)
        
        # Return top-k after reranking
        return reranked_results[:top_k]
    

    # This function can be called instead of the current find_similar_chunks
    # when you have data in Qdrant and want to perform efficient similarity search.
    def find_similar_chunks_with_qdrant(self, query_text: str, vector_db, top_k: int = 5):
        """Find chunks similar to a query text using Qdrant vector database."""
        if not vector_db:
            raise ValueError("Vector database is required for this search method")
        
        # Use LangChain's similarity_search which will use Qdrant under the hood
        results = vector_db.similarity_search(query_text, k=top_k)
        
        # Or you can use Qdrant client directly for more control
        # embedding = self.config.embedding_model.encode([query_text])[0]
        # results = vector_db.client.search(
        #     collection_name=vector_db.collection_name,
        #     query_vector=embedding,
        #     limit=top_k
        # )
        
        return results

    def cluster_chunks(self, chunks: List[Document], min_clusters=2, max_clusters=20, embedding_model=None):
        """Cluster chunks with dynamic cluster selection using the Elbow Method."""
        from sklearn.metrics import silhouette_score
        import matplotlib.pyplot as plt
        
        if embedding_model is None:
            if self.config.embedding_model is None:
                embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            else:
                embedding_model = self.config.embedding_model
        
        # Get embeddings for chunks
        chunk_texts = [chunk.page_content for chunk in chunks]
        chunk_embeddings = embedding_model.encode(chunk_texts)
        
        # Find optimal number of clusters
        silhouette_scores = []
        k_values = range(min_clusters, min(max_clusters, len(chunks) - 1))
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(chunk_embeddings)
            
            # Skip if only one sample in a cluster
            if len(set(cluster_labels)) < k:
                silhouette_scores.append(0)
                continue
                
            score = silhouette_score(chunk_embeddings, cluster_labels)
            silhouette_scores.append(score)
        
        # Find optimal k (highest silhouette score)
        optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
        
        # Apply K-means with optimal clusters
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(chunk_embeddings)
        
        # Group chunks by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(chunks[i])
        
        return clusters, optimal_k

    def extract_key_topics(self, chunk_clusters: Dict[int, List[Document]], embedding_model=None):
        """Extract key topics or themes from clustered chunks."""
        topics = {}
        
        for cluster_id, cluster_chunks in chunk_clusters.items():
            # Get most common headers in this cluster
            headers = {}
            for chunk in cluster_chunks:
                header = chunk.metadata.get("current_header", "")
                if header:
                    headers[header] = headers.get(header, 0) + 1
            
            # Find most common header(s)
            if headers:
                common_headers = sorted(headers.items(), key=lambda x: x[1], reverse=True)
                top_header = common_headers[0][0]
                
                # Use as topic name
                topic_name = top_header
            else:
                # Fallback: use first few words of first chunk
                first_text = cluster_chunks[0].page_content
                words = first_text.split()[:5]
                topic_name = " ".join(words) + "..."
            
            topics[cluster_id] = {
                "name": topic_name,
                "chunk_count": len(cluster_chunks),
                "sample_chunks": [c.page_content[:100] + "..." for c in cluster_chunks[:3]]
            }
        
        return topics

# Embedding
def setup_embeddings(model_name="all-MiniLM-L6-v2"):
    """Set up open-source embeddings model.
    
    Args:
        model_name: Name of the HuggingFace model to use
        
    Returns:
        A HuggingFaceEmbeddings instance
    """
    # Check if models directory exists, if not create it
    cache_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Loading embeddings model {model_name}...")
    
    # Use HuggingFaceEmbeddings as a free alternative to OpenAI embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=cache_dir,
        model_kwargs={'device': 'cpu'}
    )
    
    return embeddings

def get_embedding_dimension(model_name="all-MiniLM-L6-v2"):
    """Get the dimension of embeddings for a model.
    
    Args:
        model_name: Name of the HuggingFace model
        
    Returns:
        Integer representing embedding dimension
    """
    # Check if models directory exists, if not create it
    cache_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load the model to check its dimension
    model = SentenceTransformer(model_name, cache_folder=cache_dir)
    return model.get_sentence_embedding_dimension()