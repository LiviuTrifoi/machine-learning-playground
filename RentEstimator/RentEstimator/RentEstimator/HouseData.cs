using Microsoft.ML.Data;

namespace RentEstimator
{
    class HouseData
    {
        [LoadColumn(2)]
        public string Zone;

        [LoadColumn(38)]
        public float BasementSize;

        [LoadColumn(12)]
        public string Neighborhood;

        [LoadColumn(15)]
        public string BldgType;

        [LoadColumn(16)]
        public string HouseStyle;

        [LoadColumn(17)]
        public float OverallQuality;

        [LoadColumn(46)]
        public float Size;

        [LoadColumn(80)]
        public float Price;
    }
}
