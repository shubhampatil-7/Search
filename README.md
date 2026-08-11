# Search Engine

A command-line search engine built with Python.

Currently, the project supports keyword-based search using an **inverted index**. Documents are processed and indexed for fast keyword lookup.

## Current Features

* Inverted index for keyword search
* Text preprocessing and tokenization
* Stop-word removal
* Word stemming with NLTK `PorterStemmer`
* Term frequency tracking
* Cached index using `pickle`
* Command-line search interface

## Usage

### Build the index

```bash id="kg2l3w"
uv run cli/keyword_search_cli.py build
```

### Search

```bash id="dfc9hz"
uv run cli/keyword_search_cli.py search "brave"
```

## Tech

* Python
* NLTK
* uv

## Status

Work in progress. More search and retrieval techniques will be added as the project develops.
