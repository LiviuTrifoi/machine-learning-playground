using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RentEstimator
{
    class HouseData
    {
        [LoadColumn(1)]
        public string TransactionDate;

        [LoadColumn(2)]
        public float HouseAge;

        [LoadColumn(3)]
        public float DistanceToMetro;

        [LoadColumn(4)]
        public int NumberOfNearbyShops;

        [LoadColumn(5)]
        public float Latitude;

        [LoadColumn(6)]
        public float Longitude;

        [LoadColumn(7)]
        public float Price;
    }
}
