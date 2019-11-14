using System;
using Microsoft.ML;

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
            var testDataView = mlContext.Data.LoadFromTextFile<HouseData>(TestDataPath, hasHeader: true, separatorChar: ',');
        }

        public float H(float[] theta, float[] x)
        {
            float value = theta[0]*x[0] + theta[1]*x[1];

            return value;
        }

        public float Cost(float[] theta, float[][] xSets, float[] y)
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

        public float[] GradientDescent(float[][] xSets, float[] y)
        {
            const float learningRate = 0.3f;
            float t0 = 1;
            float t1 = 2;
            float prevCost = Int32.MaxValue;
            float currentCost = 0;
            do
            {
                var prevTheta = new[] { t0, t1 };
                t0 = t0 - learningRate * Delta(prevTheta, xSets, y);
                t1 = t1 - learningRate * Delta(prevTheta, xSets, y);
                currentCost = Cost(new[] { t0, t1 }, xSets, y);
            } while (currentCost < prevCost);

            return new[] { t0, t1 };
        }

        private float Delta(float[] theta, float[][] xSets, float[] y)
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
