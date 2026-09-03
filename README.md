Here's the complete README.md file you can copy directly:

```markdown
# Search Engine

A command-line search engine built with Python.

Currently, the project supports keyword-based search using an **inverted index**. Documents are processed and indexed for fast keyword lookup.

## Current Features

- Inverted index for keyword search
- Text preprocessing and tokenization
- Stop-word removal
- Word stemming with NLTK `PorterStemmer`
- Term frequency tracking
- Cached index using `pickle`
- Command-line search interface
- TF (Term Frequency) calculation
- IDF (Inverse Document Frequency) calculation
- TF-IDF scoring

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- [uv](https://github.com/astral-sh/uv) package manager

### Install Dependencies
```bash
uv sync
```

### Download NLTK Data
```bash
python -c "import nltk; nltk.download('punkt')"
```

---

## CLI Commands

### Build the Index
Build and save the inverted index from your document collection:
```bash
uv run cli/keyword_search_cli.py build
```
This will process all documents, create the inverted index, and cache it using pickle.

**Output:**
```
Building inverted index...
Inverted index built successfully.
```

---

### Search for Keywords
Search for documents containing specific keywords:
```bash
uv run cli/keyword_search_cli.py search "your search query"
```

**Example:**
```bash
uv run cli/keyword_search_cli.py search "brave new world"
```

**Output:**
```
Searching for: brave new world
1. (doc_123) Brave New World - Aldous Huxley
2. (doc_456) World War Z - Max Brooks
```

---

### Term Frequency (TF)
Get the frequency of a specific term in a specific document:
```bash
uv run cli/keyword_search_cli.py tf <doc_id> <term>
```

**Example:**
```bash
uv run cli/keyword_search_cli.py tf 42 "brave"
```

**Output:**
```
Term frequency of 'brave' in document 42: 3
```

---

### Inverse Document Frequency (IDF)
Get the inverse document frequency for a specific term across all documents:
```bash
uv run cli/keyword_search_cli.py idf <term>
```

**Example:**
```bash
uv run cli/keyword_search_cli.py idf "brave"
```

**Output:**
```
Inverse document frequency of 'brave': 2.30
```

---

### TF-IDF Score
Get the TF-IDF score for a specific term in a specific document:
```bash
uv run cli/keyword_search_cli.py tfidf <doc_id> <term>
```

**Example:**
```bash
uv run cli/keyword_search_cli.py tfidf 42 "brave"
```

**Output:**
```
TF-IDF score of 'brave' in document '42': 6.90
```

---

## Command Overview

| Command | Description | Example |
|---------|-------------|---------|
| `build` | Build and save the inverted index | `uv run cli/keyword_search_cli.py build` |
| `search` | Search for documents containing keywords | `uv run cli/keyword_search_cli.py search "query"` |
| `tf` | Get term frequency for a document-term pair | `uv run cli/keyword_search_cli.py tf 1 "word"` |
| `idf` | Get inverse document frequency for a term | `uv run cli/keyword_search_cli.py idf "word"` |
| `tfidf` | Get TF-IDF score for a document-term pair | `uv run cli/keyword_search_cli.py tfidf 1 "word"` |

---

## How It Works

1. **Text Preprocessing**: Documents are tokenized, stop-words removed, and words stemmed using NLTK's PorterStemmer
2. **Inverted Index**: Maps terms to documents containing them, with term frequency tracking
3. **Caching**: Index is saved using pickle for faster subsequent loads
4. **Search**: Query terms are processed the same way as documents, then matched against the index
5. **TF-IDF**: Combines term frequency (how often a term appears in a document) with inverse document frequency (how rare the term is across all documents) to rank relevance

---

## Project Structure

```
search-engine/
├── cli/
│   └── keyword_search_cli.py    # Command-line interface
├── keyword_search/              # Core search engine module
│   ├── __init__.py
│   ├── inverted_index.py        # Inverted index implementation
│   └── tokenizer.py             # Text preprocessing utilities
├── data/                        # Document collection
│   └── documents.json           # Source documents
├── index/                       # Cached index files
│   └── inverted_index.pkl       # Pickled index
└── README.md
```

---

## Tech Stack

- **Python** - Core programming language
- **NLTK** - Natural Language Toolkit for stemming and tokenization
- **uv** - Fast Python package installer and resolver

---

## Troubleshooting

### NLTK Errors
If you encounter NLTK-related errors, make sure you've downloaded the required data:
```bash
python -c "import nltk; nltk.download('punkt')"
```

### No Search Results
- Make sure you've built the index first with `build` command
- Try using more general keywords
- Check if your documents are in the correct format

### Build Fails
- Ensure your document collection is in the expected location
- Check that `data/documents.json` exists and is properly formatted
- Verify you have write permissions for the index directory

### Module Import Errors
If you see import errors, ensure you're running commands from the project root directory:
```bash
cd /path/to/search-engine
uv run cli/keyword_search_cli.py build
```

---

## Status

Work in progress. More search and retrieval techniques will be added as the project develops.

## Future Improvements

- [ ] BM25 ranking algorithm
- [ ] Vector space model with cosine similarity
- [ ] Query expansion and spelling correction
- [ ] Boolean operators (AND, OR, NOT)
- [ ] Phrase search
- [ ] Web interface
- [ ] Support for more document formats (PDF, DOCX, etc.)

---

## License

MIT License - feel free to use and modify as needed.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
