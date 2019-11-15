using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

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
            string TrainDataPath = @".\data\train.csv";
            var trainDataView = mlContext.Data.LoadFromTextFile<HouseData>(TrainDataPath, hasHeader: true, separatorChar: ',');
            var x = trainDataView.GetColumn<float>(nameof(HouseData.Size)).Select(f => new[] { f }).ToArray();
            var y = trainDataView.GetColumn<float>(nameof(HouseData.Price)).ToArray();
            var result = GradientDescent(x, y);

            Evaluate(mlContext, result);
        }

        public static void Evaluate(MLContext mlContext, float[] theta)
        {
            const string TestDataPath = @".\data\test.csv";
            var trainDataView = mlContext.Data.LoadFromTextFile<HouseData>(TestDataPath, hasHeader: true, separatorChar: ',');
            var x = trainDataView.GetColumn<float>(nameof(HouseData.Size)).Select(f => new[] { f }).ToArray();
            var y = trainDataView.GetColumn<float>(nameof(HouseData.Price)).ToArray();

            var mse = CalcMeanSquareError(theta, x, y);
            Console.WriteLine("Result found. Mean Square Error is: {0}", mse);
            Console.ReadKey();
        }

        public static float CalcMeanSquareError(float[] theta, float[][] xSets, float[] y)
        {
            float cost = 0;
            int m = y.Length;
            for (var i = 0; i < m - 1; i++)
            {
                var diff = Math.Abs(Hypothesis(theta, xSets[i]) - y[i]);
                Console.WriteLine(diff);
                cost += diff;
            }
            cost = cost / m;

            return cost;
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

                distances[i] = (float) Math.Sqrt(Math.Abs(latitude - TaipeiCenterLat) * 111 + Math.Abs(longitude - TaipeiCenterLong) * 111);
            }

            return distances;
        }

        public static float Hypothesis(float[] theta, float[] x)
        {
            //float value = theta[0] + theta[1]*(1/(float)Math.Sqrt(x[0]));
            //var distanceToCenter = x[0];
            //var distanceToMetro = x[1];
            //var numberOfStores = x[2];
            //float value = theta[0] + theta[1]*(1/x[0]) + theta[2]*(1/(float)Math.Sqrt(x[1])) + theta[3] * numberOfStores;
            float value = theta[0] + theta[1] * x[0];

            return value;
        }

        public static float Cost(float[] theta, float[][] xSets, float[] y)
        {
            float cost = 0;
            int m = y.Length;
            for (var i = 0; i < m - 1; i++)
            {
                var diff = (Hypothesis(theta, xSets[i]) - y[i]) ;
                cost += diff * diff;
            }
            cost = cost / (2 * m);

            return cost;
        }

        public static float[] GradientDescent(float[][] xSets, float[] y)
        {
            const float learningRate = 0.1f;
            float t0 = 0f;
            float t1 = 1f;
            float prevCost = Int32.MaxValue;
            float currentCost = Cost(new[] { t0, t1 }, xSets, y);
            int maxIterations = 100000;
            do
            {
                prevCost = currentCost;
                var prevTheta = new[] { t0, t1  };
                t0 = t0 - learningRate * Delta(prevTheta, xSets, y);
                t1 = t1 - learningRate * Delta(prevTheta, xSets, y);
                currentCost = Cost(new[] { t0, t1  }, xSets, y);
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
                delta += Hypothesis(theta, xSets[i]) - y[i];
            }

            return delta / m;
        }
    }
}
