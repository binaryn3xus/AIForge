# AI.SQLVector

This project is a C# console application demonstrating a complete Retrieval-Augmented Generation (RAG) pipeline. It uses SQL Server 2025 as a vector database and a locally hosted Ollama instance for AI embeddings and text generation.

The application allows a user to ask natural language questions about the products in the AdventureWorks database. It then finds the most relevant product descriptions using vector similarity search and uses an AI model to generate a conversational answer based on that context.

---

## Features

-   **Vector Search:** Leverages the new `VECTOR` data type and `VECTOR_DISTANCE` function in SQL Server 2025.
-   **Local AI:** Integrates with a local Ollama instance for all AI operations, keeping data processing private.
-   **RAG Pipeline:** Implements a full Retrieval-Augmented Generation workflow to provide context-aware answers.
-   **Configuration Management:** Uses .NET User Secrets to securely store the database connection string.

---

## Prerequisites

-   .NET 8 SDK
-   SQL Server 2025 Developer Edition (on Windows)
-   Ollama installed and running with the `nomic-embed-text` and `llama3` models.
-   The AdventureWorks2022 sample database, restored on your SQL Server instance.

---

## Setup & Configuration

### 1. Database Preparation

Ensure you have run the necessary SQL scripts to:
-   Add the `embeddings` and `chunk` columns to the `Production.ProductDescription` table.
-   Create the `EXTERNAL MODEL` named `ollama`.
-   Backfill the `embeddings` column by running the update script.

### 2. Configure User Secrets

This project requires the `Microsoft.Extensions.Configuration.UserSecrets` NuGet package.

```bash
dotnet add package Microsoft.Extensions.Configuration.UserSecrets

From your project's root directory in the terminal, initialize and set your database connection string:

# Step 1: Initialize user secrets for the project
dotnet user-secrets init

# Step 2: Set the connection string (replace with your actual string)
dotnet user-secrets set "ConnectionStrings:MainDatabase" "Server=your_server;Database=AdventureWorks2022;Integrated Security=True;TrustServerCertificate=True;"

3. Run the Application

Once the database and user secrets are configured, you can run the application directly:

dotnet run

The application will start and prompt you to ask questions about AdventureWorks products.

--- AdventureWorks AI Assistant (AI.SQLVector) ---

Ask a question about a product (or type 'exit' to quit):

