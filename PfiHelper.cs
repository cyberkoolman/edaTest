using System;
using System.Linq;
using Microsoft.Data;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.SamplesUtils;
using Microsoft.ML.Learners;
using Microsoft.ML.Trainers.HalLearners;

namespace edaAttrition
{
    public static class PfiHelper
    {
        public static float[] GetLinearModelWeights(OlsLinearRegressionModelParameters linearModel)
        {
            return linearModel.Weights.ToArray();
        }

        public static float[] GetLinearModelWeights(LinearBinaryModelParameters linearModel)
        {
            return linearModel.Weights.ToArray();
        }
    }
}