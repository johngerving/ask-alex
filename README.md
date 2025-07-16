# Ask Alex

Ask Alex is an AI-powered chat assistant designed to provide an intuitive interface for searching and interacting with documents from Cal Poly Humboldt's institutional repositories. It leverages a Retrieval Augmented Generation (RAG) architecture to deliver relevant answers based on the university's Digital Commons repository.

## Features

- **AI Chat:** Utilizes an agentic system with distinct retrieval and writer agents to handle user queries effectively.
- **Retrieval Augmented Generation (RAG):** Answers are grounded in the content of Cal Poly Humboldt's Digital Commons, providing citations and links to the source documents.
- **Hybrid Search:** Combines semantic (vector) search with full-text keyword search for comprehensive and relevant results.
- **Metadata Filtering:** Allows users to refine searches by metadata such as author, department, collection, and publication date.
- **Authentication:** User authentication is handled through Google OAuth, ensuring secure access.
- **Scalable Data Pipeline:** A data processing pipeline built with Ray handles the ingestion, conversion, and indexing of documents.

## Architecture

The Ask Alex application is composed of several key components:

1.  **Frontend:** A web interface built with SvelteKit that provides the user-facing chat application.
2.  **Backend (`app`)** A FastAPI application that serves the main API. It handles user authentication, manages chat sessions, and orchestrates the agentic workflow for processing user queries.
3.  **Data Processing Pipeline (`parse`):** A separate, Ray-based application responsible for periodically fetching documents from the Digital Commons API, converting them from PDF to text, and indexing them into the database and vector store.
4.  **Database (ParadeDB):** The primary data store for documents, metadata, user information, and chat history. It is enhanced with `pg_vector` for semantic search and `pg_trgm` for efficient full-text search.

## Technology Stack

| Component           | Technologies                                              |
| ------------------- | --------------------------------------------------------- |
| **Frontend**        | SvelteKit, TypeScript, Tailwind CSS, Vite                 |
| **Backend**         | Python, FastAPI, LlamaIndex, OpenRouter (LLMs), `psycopg` |
| **Data Processing** | Ray, `docling`                                            |
| **Database**        | PostgreSQL, `pg_vector`, `pg_trgm`                        |
| **Testing**         | Cypress (E2E), Vitest (Unit)                              |
| **Deployment**      | Docker, Kubernetes, Helm (WIP)                            |

## Repository Structure

```
.
├── backend/        # FastAPI application and data processing pipeline
│   ├── app/        # Core backend application code (agents, auth, chat)
│   ├── db/         # SQL database migrations (using tern)
│   ├── eval/       # Scripts for evaluating the RAG pipeline
│   └── parse/      # Ray-based data ingestion and indexing pipeline
├── cypress/        # Cypress end-to-end tests
├── frontend/       # SvelteKit frontend application
├── helm/           # Helm chart for Kubernetes deployment (WIP)
├── build.sh        # Script for building and pushing Docker images
└── README.md       # This file
```
