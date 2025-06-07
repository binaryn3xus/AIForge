using System.Data;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Data.SqlClient;
using Microsoft.Extensions.Configuration;

public class Program
{
    // This will be loaded from user secrets in the Main method.
    private static string? sqlConnectionString;

    // The IP address of your Ollama server.
    private static readonly string ollamaApiBase = "http://10.0.30.6:11434";

    // The names of the Ollama models you want to use.
    private static readonly string embeddingModel = "nomic-embed-text";
    private static readonly string generationModel = "llama3";

    // HttpClient is used to send requests to the Ollama API.
    private static readonly HttpClient httpClient = new() { BaseAddress = new Uri(ollamaApiBase) };

    public static async Task Main()
    {
        // --- STEP 1: LOAD CONFIGURATION FROM USER SECRETS ---
        var configuration = new ConfigurationBuilder()
            .AddUserSecrets<Program>()
            .Build();

        // Corrected the configuration key to use "ConnectionStrings" (plural)
        sqlConnectionString = configuration["ConnectionStrings:MainDatabase"];

        if (string.IsNullOrWhiteSpace(sqlConnectionString))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("Error: Connection string 'ConnectionStrings:MainDatabase' not found in user secrets.");
            Console.WriteLine("Please run 'dotnet user-secrets set \"ConnectionStrings:MainDatabase\" \"<your_connection_string>\"' to configure it.");
            Console.ResetColor();
            return;
        }

        Console.WriteLine("--- AdventureWorks AI Assistant (AI.SQLVector) ---");
        while (true)
        {
            Console.Write("\nAsk a question about a product (or type 'exit' to quit): ");
            string userQuestion = Console.ReadLine() ?? "";

            if (userQuestion.Equals("exit", StringComparison.OrdinalIgnoreCase))
            {
                break;
            }

            if (string.IsNullOrWhiteSpace(userQuestion)) continue;

            try
            {
                // 1. Use the user's question to find relevant text chunks from the database.
                Console.WriteLine("\n> 1. Searching database for relevant context...");
                string context = await FetchContextFromDatabaseAsync(userQuestion); // Pass the raw question string.

                if (string.IsNullOrWhiteSpace(context))
                {
                    Console.WriteLine("\n> I couldn't find any relevant product descriptions to answer your question.");
                    continue;
                }
                Console.WriteLine("> Found relevant context!");

                // 2. Send the original question and the retrieved context to a language model to generate a final answer.
                Console.WriteLine("> 2. Generating AI answer...");
                Console.WriteLine("\n--- AI Answer ---");
                await GenerateCompletionAsync(userQuestion, context);
                Console.WriteLine("\n-----------------");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\nAn error occurred: {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Searches the SQL Server database to find the most relevant text chunks based on the user's question.
    /// It now generates the vector embedding inside the SQL query itself.
    /// </summary>
    /// <param name="userQuestion">The raw text of the user's question.</param>
    /// <returns>A string containing the most relevant text chunks found in the database.</returns>
    private static async Task<string> FetchContextFromDatabaseAsync(string userQuestion)
    {
        var contextBuilder = new StringBuilder();
        await using var connection = new SqlConnection(sqlConnectionString);
        await connection.OpenAsync();

        // **CORRECTED QUERY:** The JOIN path from ProductDescription to ProductCategory
        // now correctly goes through the Product and ProductSubcategory tables.
        string sqlQuery = @"
            DECLARE @search_vector VECTOR(768) = AI_GENERATE_EMBEDDINGS(@question USE MODEL ollama);
            
            SELECT TOP 5
                pd.chunk
            FROM
                Production.ProductDescription AS pd
            -- Join to the linking table that connects descriptions to models and cultures
            INNER JOIN Production.ProductModelProductDescriptionCulture AS pmpdc
                ON pd.ProductDescriptionID = pmpdc.ProductDescriptionID
            -- Join to the ProductModel table
            INNER JOIN Production.ProductModel AS pm
                ON pmpdc.ProductModelID = pm.ProductModelID
            -- Join from ProductModel to the actual Product table
            INNER JOIN Production.Product AS p
                ON pm.ProductModelID = p.ProductModelID
            -- Join from Product to the Subcategory table
            INNER JOIN Production.ProductSubcategory AS psc
                ON p.ProductSubcategoryID = psc.ProductSubcategoryID
            -- Finally, join from Subcategory to Category
            INNER JOIN Production.ProductCategory AS pc
                ON psc.ProductCategoryID = pc.ProductCategoryID
            WHERE
                -- Now we can filter on the category name
                pc.Name LIKE '%Bikes%'
                AND pmpdc.CultureID = 'en'
            ORDER BY
                VECTOR_DISTANCE('cosine', @search_vector, pd.embeddings) DESC;
        ";

        var command = new SqlCommand(sqlQuery, connection);

        // Pass the user's raw question as the parameter.
        command.Parameters.Add(new SqlParameter("@question", SqlDbType.NVarChar, -1) { Value = userQuestion });

        await using var reader = await command.ExecuteReaderAsync();
        while (await reader.ReadAsync())
        {
            // We use the 'chunk' column as remembered.
            contextBuilder.AppendLine($"- {reader["chunk"]}");
        }

        return contextBuilder.ToString();
    }

    /// <summary>
    /// Sends the user's question and the retrieved context to an Ollama chat model to generate a final answer.
    /// The response is streamed to the console to create a "live typing" effect.
    /// </summary>
    /// <param name="question">The original user question.</param>
    /// <param name="context">The context retrieved from the database.</param>
    private static async Task GenerateCompletionAsync(string question, string context)
    {
        // This prompt engineering step is crucial. It instructs the AI on how to behave.
        string prompt = "You are an assistant for the AdventureWorks bicycle company.\n" +
                        "Based ONLY on the context below, answer the user's question.\n" +
                        "If the context does not contain the answer, say 'I do not have enough information to answer that question'.\n\n" +
                        $"--- Context ---\n{context}\n\n" +
                        $"--- User's Question ---\n{question}\n\n" +
                        "--- Answer ---";

        var requestData = new { model = generationModel, prompt = prompt, stream = true }; // stream = true for live typing
        var request = new HttpRequestMessage(HttpMethod.Post, "/api/generate")
        {
            Content = JsonContent.Create(requestData)
        };

        // Use SendAsync with HttpCompletionOption.ResponseHeadersRead for streaming
        using var response = await httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        response.EnsureSuccessStatusCode();

        await using var stream = await response.Content.ReadAsStreamAsync();
        using var reader = new StreamReader(stream);

        // Read the stream line by line until it's finished
        while (!reader.EndOfStream)
        {
            var line = await reader.ReadLineAsync();
            if (string.IsNullOrWhiteSpace(line)) continue;

            // Deserialize each line of the stream and write the content to the console
            var responseData = JsonSerializer.Deserialize<OllamaGenerationResponse>(line);
            Console.Write(responseData?.Response ?? "");
        }
    }
}

// Helper classes to deserialize Ollama's JSON responses.
public class OllamaEmbeddingResponse
{
    [JsonPropertyName("embedding")]
    public float[]? Embedding { get; set; }
}

public class OllamaGenerationResponse
{
    [JsonPropertyName("response")]
    public string? Response { get; set; }
}