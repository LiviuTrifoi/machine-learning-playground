using Microsoft.ML.Data;

namespace RentEstimator
{
    class HouseData
    {
        [LoadColumn(46)]
        public float Size;

        [LoadColumn(80)]
        public float Price;
    }
}
