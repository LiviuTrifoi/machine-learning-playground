using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace RentEstimator
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("hello machine learning");

            LoadData();
            Console.ReadKey();
        }

        public static void LoadData()
        {
            var mlContext = new MLContext();

            string TrainDataPath = @".\data\house-prices.csv";
            string TestDataPath = @".\data\test-prices.csv";

            var trainDataView = mlContext.Data.LoadFromTextFile<HouseData>(TrainDataPath, hasHeader: true, separatorChar: ',');
            //var testDataView = mlContext.Data.LoadFromTextFile<HouseData>(TestDataPath, hasHeader: true, separatorChar: ',');

            //var pipeline = mlContext.Transforms.SelectColumns("DistanceToMetro", "Price");
            //var transformedData = pipeline.Fit(trainDataView);

            //var x = trainDataView.GetColumn<float>("DistanceToMetro").Select(f => new[] { f }).ToArray();
            //var pipeline = mlContext.Transforms.Concatenate("LatLong", "Latitude", "Longitude");
            //var transformedData = pipeline.Fit(trainDataView).Transform(trainDataView);

            var lats = trainDataView.GetColumn<float>("Latitude").ToArray();
            var longs = trainDataView.GetColumn<float>("Longitude").ToArray();
            var centerDistances = CalcDistance(lats, longs).Select(d => new[] { d }).ToArray();
            var x = trainDataView.GetColumn<float>("DistanceToMetro").Select(f => new[] { f }).ToArray();
            var y = trainDataView.GetColumn<float>("Price").ToArray();

            var result = GradientDescent(centerDistances, y);
            Console.WriteLine("Result found");
        }

        public static float[] CalcDistance(float[] latitudes, float[] longitudes)
        {
            const float TaipeiCenterLat = 25.0339687f;
            const float TaipeiCenterLong = 121.5622835f;

            var distances = new float[latitudes.Length];
            for (int i = 0; i < latitudes.Length; i++)
            {
                var latitude = latitudes[i];
                var longitude = longitudes[i];

                distances[i] = (float) Math.Sqrt(Math.Abs(latitude - TaipeiCenterLat) + Math.Abs(longitude - TaipeiCenterLong));
            }

            return distances;
        }

        public static float H(float[] theta, float[] x)
        {
            float value = theta[0] + theta[1]*(1/x[0]);

            return value;
        }

        public static float Cost(float[] theta, float[][] xSets, float[] y)
        {
            float cost = 0;
            int m = y.Length;
            for (var i = 0; i < m - 1; i++)
            {
                var diff = (H(theta, xSets[i]) - y[i]) ;
                cost += diff * diff;
            }
            cost = cost / (2 * m);

            return cost;
        }

        public static float[] GradientDescent(float[][] xSets, float[] y)
        {
            const float learningRate = 0.3f;
            float t0 = 0;
            float t1 = 1f;
            float prevCost = Int32.MaxValue;
            float currentCost = Cost(new[] { t0, t1 }, xSets, y);
            int maxIterations = 100000;
            do
            {
                prevCost = currentCost;
                var prevTheta = new[] { t0, t1 };
                t0 = t0 - learningRate * Delta(prevTheta, xSets, y);
                t1 = t1 - learningRate * Delta(prevTheta, xSets, y);
                currentCost = Cost(new[] { t0, t1 }, xSets, y);
                maxIterations--;
            } while (currentCost > 1 && maxIterations > 0);

            return new[] { t0, t1 };
        }

        private static float Delta(float[] theta, float[][] xSets, float[] y)
        {
            int m = y.Length;
            float delta = 0f;
            for (int i = 0; i < m; i++)
            {
                delta += H(theta, xSets[i]) - y[i];
            }

            return delta / m;
        }
    }
}
