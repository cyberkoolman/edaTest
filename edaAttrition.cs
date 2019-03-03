using System;
using System.Collections;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.Data.DataView;
using Microsoft.ML.Trainers;
using Microsoft.ML.Learners;
using Microsoft.ML.Transforms.Categorical;
using System.Collections.Generic;

namespace edaAttrition
{
    class Program
    {

        static void Main(string[] args)
        {
            var ml = new MLContext();

            // Not including EmployeeCount, Over18, StandardHours
            var reader = ml.Data.CreateTextLoader(
                columns: new[]
            {
                new TextLoader.Column("Age", DataKind.R4, 0),
                new TextLoader.Column("Attrition", DataKind.Bool, 1),
                new TextLoader.Column("BusinessTravel", DataKind.Text, 2),
                new TextLoader.Column("DailyRate", DataKind.R4, 3),
                new TextLoader.Column("Department", DataKind.Text, 4),
                new TextLoader.Column("DistanceFromHome", DataKind.R4, 5),
                new TextLoader.Column("Education", DataKind.R4, 6),
                new TextLoader.Column("EducationField", DataKind.Text, 7),
                // new TextLoader.Column("EmployeeCount", DataKind.R4, 8),
                new TextLoader.Column("EmployeeNumber", DataKind.R4, 9),
                new TextLoader.Column("EnvironmentSatisfaction", DataKind.R4, 10),
                new TextLoader.Column("Gender", DataKind.Text, 11),
                new TextLoader.Column("HourlyRate", DataKind.R4, 12),
                new TextLoader.Column("JobInvolvement", DataKind.R4, 13),
                new TextLoader.Column("JobLevel", DataKind.R4, 14),
                new TextLoader.Column("JobRole", DataKind.Text, 15),
                new TextLoader.Column("JobSatisfaction", DataKind.R4, 16),
                new TextLoader.Column("MaritalStatus", DataKind.Text, 17),
                new TextLoader.Column("MonthlyIncome", DataKind.R4, 18),
                new TextLoader.Column("MonthlyRate", DataKind.R4, 19),
                new TextLoader.Column("NumCompaniesWorked", DataKind.R4, 20),
                // new TextLoader.Column("Over18", DataKind.Text, 21),
                new TextLoader.Column("OverTime", DataKind.Text, 22),
                new TextLoader.Column("PercentSalaryHike", DataKind.R4, 23),
                new TextLoader.Column("PerformanceRating", DataKind.R4, 24),
                new TextLoader.Column("RelationshipSatisfaction", DataKind.R4, 25),
                // new TextLoader.Column("StandardHours", DataKind.R4, 26),
                new TextLoader.Column("StockOptionLevel", DataKind.R4, 27),
                new TextLoader.Column("TotalWorkingYears", DataKind.R4, 28),
                new TextLoader.Column("TrainingTimesLastYear", DataKind.R4, 29),
                new TextLoader.Column("WorkLifeBalance", DataKind.R4, 30),
                new TextLoader.Column("YearsAtCompany", DataKind.R4, 31),
                new TextLoader.Column("YearsInCurrentRole", DataKind.R4, 32),
                new TextLoader.Column("YearsSinceLastPromotion", DataKind.R4, 33),
                new TextLoader.Column("YearsWithCurrManager", DataKind.R4, 34)
            },
            separatorChar: ',',
            hasHeader: true
            );

            var labelColumn = "Attrition";

            IDataView attritionData = reader.Read("./data/attrition.csv");

            var textFields = attritionData.Schema.AsEnumerable()
                .Select(column => new { column.Name, column.Type })
                .Where(column => (column.Name != labelColumn) && (column.Type.ToString() == "Text"))
                .ToArray();

            var textNames = textFields.AsEnumerable()
                .Select(column => column.Name)
                .ToArray();

            var numericFields = attritionData.Schema.AsEnumerable()
                .Select(column => new { column.Name, column.Type })
                .Where(column => (column.Name != labelColumn) && (column.Type.ToString() != "Text"))
                .ToArray();

            var numericNames = numericFields.AsEnumerable()
                .Select(column => column.Name)
                .ToArray();

            var featureNames = textNames.Concat(numericNames).ToArray();

            // Split to 80/20
            var split = ml.BinaryClassification.TrainTestSplit(attritionData, testFraction: 0.2);
            var trainData = split.trainSet;
            var testData = split.testSet;

            var pipeline = ml.Transforms.Concatenate("Text", textNames)
                .Append(ml.Transforms.Text.FeaturizeText("TextFeatures", "Text"))
                .Append(ml.Transforms.Concatenate("NumericalFeatures", numericNames))
                .Append(ml.Transforms.Concatenate("Features", "NumericalFeatures", "TextFeatures"))
                // Regression requries normalizatoin
                .Append(ml.Transforms.Normalize("Features"))
                // Cache data in memory so that the following trainer will be able to access training examples without
                // reading them from disk multiple times.
                .AppendCacheCheckpoint(ml)                
                .Append(ml.BinaryClassification.Trainers.LogisticRegression(labelColumn: labelColumn, featureColumn: "Features"));

            var model = pipeline.Fit(trainData);

            // Score the model
            var dataWithPredictions = model.Transform(testData);
            var metrics = ml.BinaryClassification.Evaluate(dataWithPredictions, label:labelColumn);

            Console.WriteLine($"Accuracy: {metrics.Accuracy}");
            Console.WriteLine($"AUC: {metrics.Auc}");
            Console.WriteLine($"F1 Score: {metrics.F1Score}");

            Console.WriteLine($"Negative Precision: {metrics.NegativePrecision}");
            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall}");
            Console.WriteLine($"Positive Precision: {metrics.PositivePrecision}");
            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall}");

            // Permutation Feature Importance
            var linearPredictor = model.LastTransformer;

            // Linear models for binary classification are wrapped by a calibrator as a generic predictor
            //  To access it directly, we must extract it out and cast it to the proper class
            var weights = PfiHelper.GetLinearModelWeights(linearPredictor.Model.SubPredictor as LinearBinaryModelParameters);

            // Compute the permutation metrics using the properly normalized data.
            var transformedData = model.Transform(trainData);
            var permutationMetrics = ml.BinaryClassification.PermutationFeatureImportance(
                linearPredictor, transformedData, label: labelColumn, features: "Features", permutationCount: 3);

            // Now let's look at which features are most important to the model overall
            // Get the feature indices sorted by their impact on AUC
            var sortedIndices = permutationMetrics.Select((pfimetrics, index) => new { index, pfimetrics.Auc })
                .OrderByDescending(feature => Math.Abs(feature.Auc.Mean))
                .Select(feature => feature.index);

            Console.WriteLine("Feature\tModel Weight\tChange in AUC\t95% Confidence in the Mean Change in AUC");
            var auc = permutationMetrics.Select(x => x.Auc).ToArray(); // Fetch AUC as an array
            foreach (int i in sortedIndices)
            {
                Console.WriteLine($"{featureNames[i]}\t{weights[i]:0.00}\t{auc[i].Mean:G4}\t{1.96 * auc[i].StandardError:G4}");
            }

        }
    }
}
