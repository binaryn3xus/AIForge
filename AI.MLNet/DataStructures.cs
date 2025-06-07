using Microsoft.ML.Data;

namespace AI.MLNet;

public class SalesDataInput
{
    // We'll load ProductID, UnitPrice, and the OrderQty we want to predict.
    // The [LoadColumn] attribute specifies the column index when loading from an IEnumerable.
    [LoadColumn(0)]
    public float ProductID { get; set; } // Using float for ML.NET, even if int in SQL

    [LoadColumn(1)]
    public float UnitPrice { get; set; }

    [LoadColumn(2)] // This is what we want to predict (the "Label")
    public float OrderQty { get; set; }
}

// Prediction output class
public class SalesPrediction
{
    // [ColumnName("Score")] public float PredictedOrderQty;
    // For regression, the default output column name for the predicted value is "Score"
    [ColumnName("Score")]
    public float PredictedOrderQty { get; set; }
}

public class ProductOrderInfo
{
    public int ProductID { get; set; } // Using int as it is in the database
    public DateTime OrderDate { get; set; }
}

public class OrderDetailLineItem
{
    public int SalesOrderID { get; set; }
    public int ProductID { get; set; }
    // public short OrderQty { get; set; } // Optional: if you want to factor in quantities later
}

public class BrevisionData
{
    // From your table structure, these seem most relevant:
    public string AttributeName { get; set; }
    public string ReturnValueType { get; set; }
    public string ReferenceAttributeValueQualifier { get; set; }
    public string ReturnLiteralValue { get; set; } // This is our target/label

    // Optional: Other fields if they might become features later
    // public string ReferenceAttributeDataType { get; set; }
    // public string ReturnLiteralValueDataType { get; set; }
}

public class PredictionOutput
{
    // This will store the original string value of the predicted ReturnLiteralValue
    public string PredictedReturnLiteralValue { get; set; }

    // ML.NET also outputs these by default for multiclass classification if you need them:
    // public uint PredictedLabel { get; set; } // The numeric key of the prediction
    // public float[] Score { get; set; } // Scores for each class
}

public class BrevisionPredictionOutput
{
    // This will store the original string value of the predicted ReturnLiteralValue
    public string PredictedReturnLiteralValue { get; set; }

    // ML.NET also outputs these by default for multiclass classification if you need them:
    // public uint PredictedLabel { get; set; } // The numeric key of the prediction
    // public float[] Score { get; set; } // Scores for each class
}