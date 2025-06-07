using Microsoft.ML; // For ML.NET Sales Prediction
using Microsoft.ML.Data; // For ML.NET Sales Prediction
using Microsoft.Extensions.Configuration;

namespace AI.MLNet;

class Program
{
    private static string _connectionString;
    private static IConfiguration _configuration;
    private static MLContext _mlContext; // For the prediction model

    static void Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        // --- 1. Setup Configuration (Done once) ---
        _configuration = new ConfigurationBuilder()
            .AddUserSecrets<Program>()
            .Build();

        _connectionString = _configuration.GetConnectionString("AdventureWorksDB");

        if (string.IsNullOrEmpty(_connectionString))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("ERROR: Connection string 'AdventureWorksDB' not found in User Secrets.");
            Console.ResetColor();
            Console.WriteLine("Press any key to exit.");
            Console.ReadKey();
            return;
        }
        Console.WriteLine("Connection string loaded successfully from User Secrets.");

        _mlContext = new MLContext(seed: 0);

        // --- 2. Display Menu and Get User Choice ---
        bool exit = false;
        while (!exit)
        {
            Console.WriteLine("\nAdventureWorks AI & Data Analysis Menu:");
            Console.WriteLine("-----------------------------------------");
            Console.WriteLine("1. Run Sales Quantity Prediction Model");
            Console.WriteLine("2. Run Popular Purchase Day Analysis");
            Console.WriteLine("3. Run Frequently Bought Together Analysis");
            Console.WriteLine("4. Run Brevision Literal Value Prediction"); // <-- NEW OPTION
            Console.WriteLine("5. Exit");                                 // <-- Updated Exit number
            Console.Write("Please enter your choice (1-5): ");

            string choice = Console.ReadLine();

            switch (choice)
            {
                case "1":
                    ExecuteSalesQuantityPrediction();
                    break;
                case "2":
                    ExecutePopularDayAnalysis();
                    break;
                case "3":
                    ExecuteFrequentlyBoughtTogetherAnalysis();
                    break;
                case "4": // <-- NEW CASE
                    ExecuteBrevisionLiteralValuePrediction();
                    break;
                case "5": // <-- Updated Exit case
                    exit = true;
                    Console.WriteLine("Exiting application.");
                    break;
                default:
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine("Invalid choice. Please try again.");
                    Console.ResetColor();
                    break;
            }
        }
        if (!exit)
        {
            Console.WriteLine("\nPress any key to return to the menu...");
            Console.ReadKey();
            Console.Clear();
        }
    }

    // --- Method for Sales Quantity Prediction (your original ML.NET code) ---
    static void ExecuteSalesQuantityPrediction()
    {
        Console.Clear();
        Console.WriteLine("🚀 AdventureWorks AI Sales Quantity Prediction 🚀");

        // --- Load Data for ML Model ---
        Console.WriteLine("\nLoading data for ML Model from AdventureWorks SQL Server...");
        // Using the SalesDataInput model and the corresponding loader
        List<SalesDataInput> salesDataForML = AdventureWorksRepository.LoadSalesDataFromSql(_connectionString); // Assumes this method exists from previous steps

        if (salesDataForML.Count == 0)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("No data loaded for ML model. Check repository method and SQL query.");
            Console.ResetColor();
            return;
        }
        Console.WriteLine($"Loaded {salesDataForML.Count} sales records for ML model.");

        IDataView fullDataView = _mlContext.Data.LoadFromEnumerable(salesDataForML);

        DataOperationsCatalog.TrainTestData dataSplit = _mlContext.Data.TrainTestSplit(fullDataView, testFraction: 0.2, seed: 0);
        IDataView trainingData = dataSplit.TrainSet;
        IDataView testData = dataSplit.TestSet;
        Console.WriteLine("Data successfully split into training and testing sets for ML model.");

        // --- Define and Train ML Pipeline ---
        Console.WriteLine("\nDefining and training ML pipeline...");
        var pipeline = _mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(SalesDataInput.OrderQty))
            .Append(_mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ProductIDEncoded", inputColumnName: nameof(SalesDataInput.ProductID)))
            .Append(_mlContext.Transforms.Concatenate("Features", "ProductIDEncoded", nameof(SalesDataInput.UnitPrice)))
            .Append(_mlContext.Transforms.NormalizeMinMax("Features")) // Important normalization step
            .Append(_mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features"));

        var model = pipeline.Fit(trainingData);
        Console.WriteLine("ML Model training complete. 🎉");

        // --- Evaluate ML Model ---
        Console.WriteLine("\nEvaluating the ML model's performance...");
        var predictionsOnTestData = model.Transform(testData);
        RegressionMetrics metrics = _mlContext.Regression.Evaluate(predictionsOnTestData, labelColumnName: "Label", scoreColumnName: "Score");
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("ML Model Evaluation Metrics:");
        Console.WriteLine("--------------------------------------------------");
        Console.WriteLine($"Mean Absolute Error (MAE):    {metrics.MeanAbsoluteError:F2}");
        Console.WriteLine($"Root Mean Squared Error (RMSE): {metrics.RootMeanSquaredError:F2}");
        Console.WriteLine($"R-squared (Coefficient of Determination): {metrics.RSquared:P2}");
        Console.WriteLine("--------------------------------------------------");
        Console.ResetColor();

        // --- Make Sample Prediction with ML Model ---
        Console.WriteLine("\nMaking a sample prediction with the ML model...");
        var predictionEngine = _mlContext.Model.CreatePredictionEngine<SalesDataInput, SalesPrediction>(model);
        var sampleInput = new SalesDataInput { ProductID = 870, UnitPrice = 5.0f };
        SalesPrediction predictionResult = predictionEngine.Predict(sampleInput);
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"Sample Prediction Details (ML Model):");
        Console.WriteLine("--------------------------------------------------");
        Console.WriteLine($"Product ID: {sampleInput.ProductID}");
        Console.WriteLine($"Unit Price: ${sampleInput.UnitPrice:F2}");
        Console.WriteLine($"Predicted Order Quantity: {predictionResult.PredictedOrderQty:F2} (Approx. {Math.Round(predictionResult.PredictedOrderQty)})");
        Console.WriteLine("--------------------------------------------------");
        Console.ResetColor();
    }

    // --- Method for Popular Purchase Day Analysis ---
    static void ExecutePopularDayAnalysis()
    {
        Console.Clear();
        Console.WriteLine("📊 AdventureWorks - Popular Purchase Day Analysis 📊");

        // --- Load Product Order Data ---
        Console.WriteLine("\nLoading product order data for analysis...");
        // Using the ProductOrderInfo model and the corresponding loader
        List<ProductOrderInfo> productOrders = AdventureWorksRepository.LoadProductOrderDates(_connectionString, 50000); // Assumes this method exists

        if (productOrders.Count == 0)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("No product order data loaded for analysis. Check repository method and SQL query.");
            Console.ResetColor();
            return;
        }
        Console.WriteLine($"Loaded {productOrders.Count} product order transaction records for analysis.");

        // --- Analyze Data ---
        Console.WriteLine("\nAnalyzing purchase day popularity per product...");
        var popularityByProduct = productOrders
            .GroupBy(order => order.ProductID)
            .Select(productGroup => new
            {
                ProductID = productGroup.Key,
                PopularDayInfo = productGroup
                                    .GroupBy(order => order.OrderDate.DayOfWeek)
                                    .Select(dayGroup => new
                                    {
                                        DayOfWeek = dayGroup.Key,
                                        PurchaseCount = dayGroup.Count()
                                    })
                                    .OrderByDescending(dayInfo => dayInfo.PurchaseCount)
                                    .FirstOrDefault()
            })
            .Where(result => result.PopularDayInfo != null)
            .ToList();

        // --- Display Results ---
        Console.WriteLine("\nMost Popular Purchase Day by Product (based on transaction count):");
        Console.WriteLine("--------------------------------------------------------------------");
        var topProductsInSample = productOrders
                            .GroupBy(order => order.ProductID)
                            .Select(g => new { ProductID = g.Key, Count = g.Count() })
                            .OrderByDescending(x => x.Count)
                            .Take(15);

        foreach (var productSampleInfo in topProductsInSample)
        {
            var productPopularity = popularityByProduct.FirstOrDefault(p => p.ProductID == productSampleInfo.ProductID);
            if (productPopularity != null && productPopularity.PopularDayInfo != null)
            {
                Console.WriteLine($"Product ID: {productPopularity.ProductID.ToString().PadRight(5)} - Most Popular Day: {productPopularity.PopularDayInfo.DayOfWeek.ToString().PadRight(10)} (Transactions: {productPopularity.PopularDayInfo.PurchaseCount})");
            }
        }

        var overallPopularDay = productOrders
            .GroupBy(order => order.OrderDate.DayOfWeek)
            .Select(dayGroup => new
            {
                DayOfWeek = dayGroup.Key,
                PurchaseCount = dayGroup.Count()
            })
            .OrderByDescending(dayInfo => dayInfo.PurchaseCount)
            .FirstOrDefault();

        if (overallPopularDay != null)
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("\n--------------------------------------------------------------------");
            Console.WriteLine($"Overall Most Popular Purchase Day (any item): {overallPopularDay.DayOfWeek} (Total Transactions: {overallPopularDay.PurchaseCount})");
            Console.WriteLine("--------------------------------------------------------------------");
            Console.ResetColor();
        }
    }

    static void ExecuteFrequentlyBoughtTogetherAnalysis()
    {
        Console.Clear();
        Console.WriteLine("🛒 AdventureWorks - Frequently Bought Together Analysis 🛒");

        Console.WriteLine("\nLoading order line item data for analysis...");
        List<OrderDetailLineItem> orderLineItems = AdventureWorksRepository.LoadOrderLineItems(_connectionString, 200_000); // Load a decent number of line items

        if (orderLineItems.Count == 0)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("No order line item data loaded. Check repository method and SQL query.");
            Console.ResetColor();
            return;
        }
        Console.WriteLine($"Loaded {orderLineItems.Count} order line items for analysis.");

        Console.WriteLine("\nFinding product pairs frequently bought together...");

        // 1. Group line items by SalesOrderID to get individual orders
        var orders = orderLineItems
            .GroupBy(item => item.SalesOrderID)
            .Select(orderGroup => new
            {
                // SalesOrderID = orderGroup.Key, // Not strictly needed for pair counting
                // Get distinct products in the order, sorted to ensure consistent pair generation (e.g., (A,B) not (B,A))
                Products = orderGroup.Select(item => item.ProductID).Distinct().OrderBy(pid => pid).ToList()
            })
            .Where(order => order.Products.Count >= 2) // Only consider orders with at least two distinct products
            .ToList();

        if (!orders.Any())
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("No orders with 2 or more distinct products found in the loaded data to analyze for pairs.");
            Console.ResetColor();
            return;
        }

        // 2. Generate product pairs from each order and count their frequencies
        var productPairFrequencies = new Dictionary<Tuple<int, int>, int>();

        foreach (var order in orders)
        {
            // Generate all unique pairs for the products in this order
            for (int i = 0; i < order.Products.Count; i++)
            {
                for (int j = i + 1; j < order.Products.Count; j++)
                {
                    // The products list is already sorted, so Products[i] will be less than Products[j]
                    var pair = Tuple.Create(order.Products[i], order.Products[j]);

                    if (productPairFrequencies.ContainsKey(pair))
                    {
                        productPairFrequencies[pair]++;
                    }
                    else
                    {
                        productPairFrequencies[pair] = 1;
                    }
                }
            }
        }

        if (!productPairFrequencies.Any())
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("No product pairs found. This might happen if all multi-item orders had unique product combinations.");
            Console.ResetColor();
            return;
        }

        // 3. Display the top N most frequent pairs
        int topN = 20;
        var topFrequentlyBoughtTogether = productPairFrequencies
            .OrderByDescending(kvp => kvp.Value)
            .Take(topN)
            .ToList();

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"\nTop {topN} Product Pairs Frequently Bought Together:");
        Console.WriteLine("----------------------------------------------------");
        if (topFrequentlyBoughtTogether.Any())
        {
            foreach (var item in topFrequentlyBoughtTogether)
            {
                Console.WriteLine($"Products ({item.Key.Item1}, {item.Key.Item2}) bought together {item.Value} times.");
            }
        }
        else
        {
            Console.WriteLine("No recurring product pairs found in the sample.");
        }
        Console.ResetColor();
    }

    static void ExecuteBrevisionLiteralValuePrediction()
    {
        Console.Clear();
        Console.OutputEncoding = System.Text.Encoding.UTF8; // For emojis/special characters
        Console.WriteLine("🔬 Brevision - Predict Missing ReturnLiteralValue 🔬");

        // Retrieve the new connection string (assuming _configuration is a static field in Program class, initialized in Main)
        string brevisionConnectionString = _configuration.GetConnectionString("ORO");
        if (string.IsNullOrEmpty(brevisionConnectionString))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("ERROR: Connection string 'BrevisionDB' not found in User Secrets.");
            Console.ResetColor();
            return;
        }

        Console.WriteLine("\nLoading Brevision data for training and prediction...");
        // The LoadBrevisionData method now separates training data from items needing prediction
        List<BrevisionData> trainingData = AdventureWorksRepository.LoadBrevisionData(brevisionConnectionString, out List<BrevisionData> itemsToPredict);

        if (trainingData.Count == 0)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("No training data loaded from Brevision DB. Cannot train a model.");
            Console.ResetColor();
            return;
        }
        Console.WriteLine($"Loaded {trainingData.Count} records for training.");
        if (itemsToPredict.Any())
        {
            Console.WriteLine($"Found {itemsToPredict.Count} items needing ReturnLiteralValue prediction.");
        }
        else
        {
            Console.WriteLine("No items found in the loaded data that need ReturnLiteralValue prediction.");
        }

        // ---- ML.NET Pipeline Construction ----
        Console.WriteLine("\nConstructing ML.NET pipeline...");

        // Start with an IEstimator<ITransformer> to allow for flexible appending
        IEstimator<ITransformer> pipeline =
            // 1. Convert 'ReturnLiteralValue' (Label) to a numerical key.
            _mlContext.Transforms.Conversion.MapValueToKey(
                outputColumnName: "Label",
                inputColumnName: nameof(BrevisionData.ReturnLiteralValue));

        // 2. Featurize 'AttributeName' (Categorical feature)
        pipeline = pipeline.Append(_mlContext.Transforms.Categorical.OneHotEncoding(
            outputColumnName: "AttributeNameEncoded",
            inputColumnName: nameof(BrevisionData.AttributeName)));

        // 3. Featurize 'ReferenceAttributeValueQualifier' (Text feature)
        pipeline = pipeline.Append(_mlContext.Transforms.Text.FeaturizeText(
            outputColumnName: "ReferenceAttributeValueQualifierFeaturized",
            inputColumnName: nameof(BrevisionData.ReferenceAttributeValueQualifier)));

        // 4. Combine all generated feature columns into a single 'Features' column.
        pipeline = pipeline.Append(_mlContext.Transforms.Concatenate(
            "Features",
            "AttributeNameEncoded",
            "ReferenceAttributeValueQualifierFeaturized"));

        // 5. (Optional but often Recommended) Normalize Features
        // pipeline = pipeline.Append(_mlContext.Transforms.NormalizeMinMax("Features"));

        // 6. Append the chosen trainer (Algorithm) for Multiclass Classification.
        var learningPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
            labelColumnName: "Label",
            featureColumnName: "Features"));
        // Or try LightGbm for potentially better performance:
        // var learningPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.LightGbm(
        // labelColumnName: "Label",
        // featureColumnName: "Features"));

        // 7. Add a final step to convert the predicted numerical key back to the original text value.
        var fullPredictionPipeline = learningPipeline.Append(_mlContext.Transforms.Conversion.MapKeyToValue(
            outputColumnName: nameof(BrevisionPredictionOutput.PredictedReturnLiteralValue), // Matches property in BrevisionPredictionOutput
            inputColumnName: "PredictedLabel")); // Default output column name for predicted class key

        // ---- Train the Model ----
        Console.WriteLine("\nTraining the prediction model...");
        IDataView trainingDataView = _mlContext.Data.LoadFromEnumerable(trainingData);
        var trainedModel = fullPredictionPipeline.Fit(trainingDataView);
        Console.WriteLine("Model training complete. 🎉");

        // ---- (Optional but Recommended) Evaluate the Model ----
        // To do this properly, you'd split 'trainingData' into a train and test set *before* fitting the pipeline.
        // Example: var trainTestData = _mlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.2);
        //          var trainedModel = fullPredictionPipeline.Fit(trainTestData.TrainSet);
        //          var metrics = _mlContext.MulticlassClassification.Evaluate(trainedModel.Transform(trainTestData.TestSet));
        //          Console.WriteLine($"Macro Accuracy: {metrics.MacroAccuracy:P2}");
        //          Console.WriteLine($"LogLoss: {metrics.LogLoss:F4}");
        // For now, we'll focus on prediction as per your request.

        // ---- Predict for items that are missing 'ReturnLiteralValue' ----
        if (itemsToPredict.Any())
        {
            Console.WriteLine($"\nPredicting missing ReturnLiteralValue for {itemsToPredict.Count} items...");
            IDataView toPredictDataView = _mlContext.Data.LoadFromEnumerable(itemsToPredict);
            IDataView predictions = trainedModel.Transform(toPredictDataView);

            // Extract original data and predicted values
            // We need to use the BrevisionPredictionOutput class here
            var predictionResults = _mlContext.Data.CreateEnumerable<BrevisionData>(predictions, reuseRowObject: false) // Gets original fields passed through
                .Zip(_mlContext.Data.CreateEnumerable<BrevisionPredictionOutput>(predictions, reuseRowObject: false), // Gets the predicted field
                     (original, prediction) => new
                     {
                         OriginalAttributeName = original.AttributeName,
                         OriginalRefQualifier = original.ReferenceAttributeValueQualifier,
                         PredictedValue = prediction.PredictedReturnLiteralValue
                     })
                .ToList();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("\nPrediction Results (for items initially missing ReturnLiteralValue):");
            Console.WriteLine("-------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"{"AttributeName",-30} | {"RefQualifier",-50} | {"Predicted ReturnLiteralValue",-30}");
            Console.WriteLine("-------------------------------------------------------------------------------------------------------------");
            foreach (var result in predictionResults.Take(200)) // Show first 20 predictions
            {
                string attrNameDisplay = (result.OriginalAttributeName?.Length > 28) ? result.OriginalAttributeName.Substring(0, 28) + ".." : result.OriginalAttributeName ?? "";
                string refQualDisplay = (result.OriginalRefQualifier?.Length > 48) ? result.OriginalRefQualifier.Substring(0, 48) + ".." : result.OriginalRefQualifier ?? "";
                string predValueDisplay = (result.PredictedValue?.Length > 28) ? result.PredictedValue.Substring(0, 28) + ".." : result.PredictedValue ?? "";

                Console.WriteLine($"{attrNameDisplay,-30} | {refQualDisplay,-50} | {predValueDisplay,-30}");
            }
            Console.ResetColor();
            if (predictionResults.Count > 200)
            {
                Console.WriteLine($"... and {predictionResults.Count - 20} more predictions.");
            }
        }

        // ---- Allow User to Input a Row for Prediction ----
        Console.WriteLine("\n🧪 Test with a manual input 🧪");
        string continueManualPredict = "y";
        // Create the PredictionEngine using BrevisionData as input and BrevisionPredictionOutput as output
        var predEngine = _mlContext.Model.CreatePredictionEngine<BrevisionData, BrevisionPredictionOutput>(trainedModel);

        while (continueManualPredict.Equals("y", StringComparison.OrdinalIgnoreCase))
        {
            Console.Write("Enter AttributeName: ");
            string attrName = Console.ReadLine() ?? string.Empty;
            Console.Write("Enter ReferenceAttributeValueQualifier: ");
            string refQualifier = Console.ReadLine() ?? string.Empty;

            var manualInput = new BrevisionData
            {
                AttributeName = attrName,
                ReferenceAttributeValueQualifier = refQualifier,
                ReturnValueType = "Literal Value" // Context for this model
                                                  // ReturnLiteralValue is what we want to predict, so it's not set in the input
            };

            var prediction = predEngine.Predict(manualInput);

            Console.ForegroundColor = ConsoleColor.Magenta;
            Console.WriteLine($"\nPredicted ReturnLiteralValue for manual input: {prediction.PredictedReturnLiteralValue}");
            Console.ResetColor();

            Console.Write("\nPredict for another manual input? (y/n): ");
            continueManualPredict = Console.ReadLine();
        }
    }


}
