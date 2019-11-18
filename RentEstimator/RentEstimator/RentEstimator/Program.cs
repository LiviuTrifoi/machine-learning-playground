using System;
using System.Collections.Generic;
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
            var dataSet = PrepareData(trainDataView);
            var result = GradientDescent(dataSet.X, dataSet.Y);
            Console.WriteLine("Theta is: ");
            foreach (var item in result)
            {
                Console.Write(item + " ");
            }
            Console.WriteLine();
            Evaluate(mlContext, result);
        }

        public static DataSet PrepareData(IDataView dataView)
        {
            var x1 = Normalize(dataView.GetColumn<float>(nameof(HouseData.Size)).ToArray());
            var x2 = MaxNormalize(dataView.GetColumn<string>(nameof(HouseData.Zone)).Select(z => ConvertZoneToWeight(z)).ToArray());
            var y = Normalize(dataView.GetColumn<float>(nameof(HouseData.Price)).ToArray());
            var x3 = MaxNormalizeAvg(dataView.GetColumn<string>(nameof(HouseData.Neighborhood)).ToArray(), y);
            var x4 = MaxNormalizeAvg(dataView.GetColumn<string>(nameof(HouseData.HouseStyle)).ToArray(), y);
            var x5 = Normalize(dataView.GetColumn<float>(nameof(HouseData.OverallQuality)).ToArray());

            var x = new float[x1.Length][];
            for (int i = 0; i < x1.Length; i++)
            {
                x[i] = new float[] { x1[i], x2[i], x3[i], x4[i], x5[i] };
            }

            return new DataSet()
            {
                X = x,
                Y = y
            };
        }

        public static float[] Normalize(float[] x)
        {
            var xMin = x.Min();
            var xMax = x.Max();
            var newX = new float[x.Length];

            for (int i = 0; i < x.Length; i++)
            {
                newX[i] = (x[i] - xMin) / (xMax - xMin);
            }

            return newX;
        }

        public static float[] MaxNormalize(float[] x)
        {
            var xMax = x.Max();
            var newX = new float[x.Length];

            for (int i = 0; i < x.Length; i++)
            {
                newX[i] = x[i] / xMax;
            }

            return newX;
        }

        public static float[] MaxNormalizeAvg(string[] x, float[] y)
        {
            var groupAvg = new Dictionary<string, SumCount>();
            for (int i = 0; i < x.Length; i++)
            {
                if (!groupAvg.ContainsKey(x[i]))
                {
                    groupAvg[x[i]] = new SumCount();
                }

                groupAvg[x[i]].Sum += y[i];
                groupAvg[x[i]].Count++;
            }

            
            var maxAvg = groupAvg.Max(pair => pair.Value.Avg());
            var newX = new float[x.Length];

            for (int i = 0; i < x.Length; i++)
            {
                newX[i] = groupAvg[x[i]].Avg() / maxAvg;
            }

            return newX;
        }

        public static float ConvertZoneToWeight(string zone)
        {
            var avgPrice = 0;
            switch (zone)
            {
                case "RL":
                    avgPrice = 191481;
                    break;
                case "RM":
                    avgPrice = 125493;
                    break;
                case "FV":
                    avgPrice = 212951;
                    break;
                case "RH":
                    avgPrice = 131558;
                    break;
                default:
                    avgPrice = 74528;
                    break;
            }

            return avgPrice;
        }

        public static void Evaluate(MLContext mlContext, float[] theta)
        {
            const string TestDataPath = @".\data\test.csv";
            var trainDataView = mlContext.Data.LoadFromTextFile<HouseData>(TestDataPath, hasHeader: true, separatorChar: ',');
            var dataSet = PrepareData(trainDataView);
            var mse = CalcMeanSquareError(theta, dataSet.X, dataSet.Y);
            Console.WriteLine("Result found. Mean Square Error is: {0}", mse * 1000000);
            Console.ReadKey();
        }

        public static float CalcMeanSquareError(float[] theta, float[][] xSets, float[] y)
        {
            float cost = 0;
            int m = y.Length;
            for (var i = 0; i < m - 1; i++)
            {
                var diff = Math.Abs(Hypothesis(theta, xSets[i]) - y[i]);
                Console.WriteLine(diff * 1000000);
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
            float livingSize = x[0];
            float zone = x[1];
            float neighbourhood = x[2];
            float buildingType = x[3];
            float overallQuality = x[4];

            float value = theta[0] + theta[1] * livingSize / 2 + theta[2] * zone / 50 + theta[3] * neighbourhood / 100 + theta[4] * buildingType / 30 + theta[5] * overallQuality / 5;

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
            const float learningRate = 0.2f;
            float t0 = 0f;
            float t1 = 1f;
            float t2 = 1f;
            float t3 = 1f;
            float t4 = 1f;
            float t5 = 1f;
            int maxIterations = 50000;

            do
            {
                var prevTheta = new[] { t0, t1, t2, t3, t4, t5 };
                t0 = t0 - learningRate * Delta(prevTheta, xSets, y);
                t1 = t1 - learningRate * Delta(prevTheta, xSets, y);
                t2 = t2 - learningRate * Delta(prevTheta, xSets, y);
                t3 = t3 - learningRate * Delta(prevTheta, xSets, y);
                t4 = t4 - learningRate * Delta(prevTheta, xSets, y);
                t5 = t5 - learningRate * Delta(prevTheta, xSets, y);
                maxIterations--;
            } while (maxIterations > 0);

            return new[] { t0, t1, t2, t3, t4, t5 };
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


        public class DataSet
        {
            public float[][] X { get; set; }
            public float[] Y { get; set; }
        }

        private class SumCount {
            public float Sum;
            public float Count;

            public float Avg()
            {
                return Sum / Count;
            }
        }
    }
}
