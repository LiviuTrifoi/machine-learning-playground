using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RentEstimator
{
    class Program
    {
        static void Main(string[] args)
        {
            LoadData();
            Console.WriteLine("hello world");
            Console.ReadKey();
        }

        public static void LoadData()
        {
            var mlContext = new MLContext();

            string TrainDataPath = @".\data\house-prices.csv";
            string TestDataPath = @".\data\test-prices.csv";

            var trainDataView = mlContext.Data.LoadFromTextFile<HouseData>(TrainDataPath, hasHeader: true, separatorChar: ',');
            var testDataView = mlContext.Data.LoadFromTextFile<HouseData>(TestDataPath, hasHeader: true, separatorChar: ',');

            var preview = trainDataView.Preview(4);
        }
    }
}
