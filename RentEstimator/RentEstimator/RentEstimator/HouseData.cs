using Microsoft.ML.Data;

namespace RentEstimator
{
    class HouseData
    {
        [LoadColumn(2)]
        public string Zone;

        [LoadColumn(38)]
        public float BasementSize;

        [LoadColumn(46)]
        public float Size;

        [LoadColumn(80)]
        public float Price;
    }
}
