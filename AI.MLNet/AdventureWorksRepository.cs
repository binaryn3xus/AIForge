using Microsoft.Data.SqlClient;

namespace AI.MLNet;

public static class AdventureWorksRepository // Made the class static as it will only contain static methods
{
    // Modified to accept the connection string as a parameter
    public static List<SalesDataInput> LoadSalesDataFromSql(string connectionString)
    {
        var salesEntries = new List<SalesDataInput>();

        // The query remains the same (without the OrderDate error)
        string queryString = @"
                SELECT TOP 5000 ProductID, UnitPrice, OrderQty
                FROM Sales.SalesOrderDetail
                WHERE OrderQty > 0 AND UnitPrice > 0;
            ";

        if (string.IsNullOrEmpty(connectionString))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("FATAL: Connection string is not provided to LoadSalesDataFromSql. Cannot load data.");
            Console.ResetColor();
            return salesEntries; // Return empty list
        }

        try
        {
            using (SqlConnection connection = new SqlConnection(connectionString))
            {
                SqlCommand command = new SqlCommand(queryString, connection);
                connection.Open();
                // Console.WriteLine("Successfully connected to SQL Server (from AdventureWorksRepository)."); // Optional: Modify logging
                using (SqlDataReader reader = command.ExecuteReader())
                {
                    while (reader.Read())
                    {
                        var entry = new SalesDataInput
                        {
                            ProductID = Convert.ToSingle(reader["ProductID"]),
                            UnitPrice = Convert.ToSingle(reader["UnitPrice"]),
                            OrderQty = Convert.ToSingle(reader["OrderQty"])
                        };
                        salesEntries.Add(entry);
                    }
                }
            }
        }
        catch (SqlException sqlEx)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"SQL Execution Error (from AdventureWorksRepository): {sqlEx.Message}");
            Console.WriteLine("Troubleshooting tips:");
            Console.WriteLine("- Is the SQL Server service running?");
            Console.WriteLine("- Is the server name in your connection string correct (check User Secrets)?");
            Console.WriteLine("- Are the database name ('AdventureWorks2022'), User ID, and Password correct?");
            Console.WriteLine("- Does the specified user ('sa' in this case) have permissions to access the 'AdventureWorks2022' database and the 'Sales.SalesOrderDetail' table?");
            Console.WriteLine("- Is your network connection to the SQL Server okay (firewalls, etc.)?");
            Console.ResetColor();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"General Error in LoadSalesDataFromSql (from AdventureWorksRepository): {ex.Message}");
            Console.ResetColor();
        }
        return salesEntries;
    }

    public static List<ProductOrderInfo> LoadProductOrderDates(string connectionString, int recordLimit = 20000) // Added a limit
    {
        var productOrderEntries = new List<ProductOrderInfo>();

        // Query to join SalesOrderDetail with SalesOrderHeader to get ProductID and OrderDate
        string queryString = $@"
        SELECT TOP ({recordLimit}) sod.ProductID, soh.OrderDate
        FROM Sales.SalesOrderDetail AS sod
        INNER JOIN Sales.SalesOrderHeader AS soh ON sod.SalesOrderID = soh.SalesOrderID
        ORDER BY soh.OrderDate DESC; -- Get more recent orders first for the sample
    ";

        if (string.IsNullOrEmpty(connectionString))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("FATAL: Connection string is not provided to LoadProductOrderDates. Cannot load data.");
            Console.ResetColor();
            return productOrderEntries;
        }

        try
        {
            using (SqlConnection connection = new SqlConnection(connectionString))
            {
                SqlCommand command = new SqlCommand(queryString, connection);
                connection.Open();
                using (SqlDataReader reader = command.ExecuteReader())
                {
                    while (reader.Read())
                    {
                        var entry = new ProductOrderInfo
                        {
                            ProductID = Convert.ToInt32(reader["ProductID"]),
                            OrderDate = Convert.ToDateTime(reader["OrderDate"])
                        };
                        productOrderEntries.Add(entry);
                    }
                }
            }
        }
        catch (SqlException sqlEx)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"SQL Execution Error (from LoadProductOrderDates): {sqlEx.Message}");
            Console.ResetColor();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"General Error in LoadProductOrderDates: {ex.Message}");
            Console.ResetColor();
        }
        return productOrderEntries;
    }

    public static List<OrderDetailLineItem> LoadOrderLineItems(string connectionString, int recordLimit = 50000)
    {
        var lineItems = new List<OrderDetailLineItem>();
        // We select more records initially because many might be single-item orders
        // or from a small number of large orders.
        // Ordering by SalesOrderID helps if we were to process this as a stream.
        string queryString = $@"
        SELECT TOP ({recordLimit}) SalesOrderID, ProductID
        FROM Sales.SalesOrderDetail
        ORDER BY SalesOrderID, ProductID;
    ";

        if (string.IsNullOrEmpty(connectionString))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("FATAL: Connection string is not provided to LoadOrderLineItems. Cannot load data.");
            Console.ResetColor();
            return lineItems;
        }

        try
        {
            using (SqlConnection connection = new SqlConnection(connectionString))
            {
                SqlCommand command = new SqlCommand(queryString, connection);
                connection.Open();
                using (SqlDataReader reader = command.ExecuteReader())
                {
                    while (reader.Read())
                    {
                        var item = new OrderDetailLineItem
                        {
                            SalesOrderID = Convert.ToInt32(reader["SalesOrderID"]),
                            ProductID = Convert.ToInt32(reader["ProductID"])
                        };
                        lineItems.Add(item);
                    }
                }
            }
        }
        catch (SqlException sqlEx)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"SQL Execution Error (from LoadOrderLineItems): {sqlEx.Message}");
            Console.ResetColor();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"General Error in LoadOrderLineItems: {ex.Message}");
            Console.ResetColor();
        }
        return lineItems;
    }

    public static List<BrevisionData> LoadBrevisionData(string connectionString, out List<BrevisionData> dataToPredict)
    {
        var trainingData = new List<BrevisionData>();
        dataToPredict = new List<BrevisionData>(); // For rows where ReturnLiteralValue is NULL or empty

        // Select relevant columns and filter by ReturnValueType
        // We'll also separate records for training (ReturnLiteralValue is not NULL)
        // from records needing prediction (ReturnLiteralValue IS NULL or empty).
        string queryString = @"
        SELECT
            AttributeName,
            ReturnValueType,
            ReferenceAttributeValueQualifier,
            ReturnLiteralValue
            -- , ReferenceAttributeDataType -- Add if needed later
            -- , ReturnLiteralValueDataType -- Add if needed later
        FROM dbo.Brevision_XRefSolve
        WHERE ReturnValueType = 'Literal Value' AND AttributeName not like '%SalesChannel%';
    ";
        // For a very large table, you might add TOP N here for development,
        // or handle filtering for training/prediction directly in SQL.

        if (string.IsNullOrEmpty(connectionString))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("FATAL: Connection string is not provided to LoadBrevisionData. Cannot load data.");
            Console.ResetColor();
            return trainingData;
        }

        try
        {
            using (SqlConnection connection = new SqlConnection(connectionString))
            {
                SqlCommand command = new SqlCommand(queryString, connection);
                connection.Open();
                using (SqlDataReader reader = command.ExecuteReader())
                {
                    while (reader.Read())
                    {
                        var entry = new BrevisionData
                        {
                            AttributeName = reader["AttributeName"] as string,
                            ReturnValueType = reader["ReturnValueType"] as string,
                            ReferenceAttributeValueQualifier = reader["ReferenceAttributeValueQualifier"] as string,
                            ReturnLiteralValue = reader["ReturnLiteralValue"] as string
                        };

                        // Sanitize inputs slightly, ensure NULLs are handled if reading from DB that allows them for these columns
                        entry.AttributeName ??= string.Empty;
                        entry.ReferenceAttributeValueQualifier ??= string.Empty;


                        if (!string.IsNullOrEmpty(entry.ReturnLiteralValue))
                        {
                            trainingData.Add(entry);
                        }
                        else
                        {
                            // This entry needs its ReturnLiteralValue predicted
                            dataToPredict.Add(entry);
                        }
                    }
                }
            }
        }
        catch (SqlException sqlEx)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"SQL Execution Error (from LoadBrevisionData): {sqlEx.Message}");
            Console.ResetColor();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"General Error in LoadBrevisionData: {ex.Message}");
            Console.ResetColor();
        }
        return trainingData;
    }
}
