#include "operators/matmul.h"
#include "utils/operator_utils.h"
namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        IT_ASSERT(inputs.size() == 2);
        const auto A = inputs[0];
        const auto B = inputs[1];

        const auto &aDims = A->getDims();
        const auto &bDims = B->getDims();
        IT_ASSERT(aDims.size() >= 2);
        IT_ASSERT(bDims.size() >= 2);

        // Batch dims: leading dims except last 2
        Shape aBatch(aDims.begin(), aDims.end() - 2);
        Shape bBatch(bDims.begin(), bDims.end() - 2);
        Shape outBatch = infer_broadcast(aBatch, bBatch);

        // Matrix dims (row-major): A(..., M, K) * B(..., K, N) = C(..., M, N)
        const int aM = transA ? aDims[aDims.size() - 1] : aDims[aDims.size() - 2];
        const int aK = transA ? aDims[aDims.size() - 2] : aDims[aDims.size() - 1];
        const int bK = transB ? bDims[bDims.size() - 1] : bDims[bDims.size() - 2];
        const int bN = transB ? bDims[bDims.size() - 2] : bDims[bDims.size() - 1];
        IT_ASSERT(aK == bK, "Matmul K dimension mismatch");

        m = aM;
        n = bN;
        k = aK;

        Shape out = outBatch;
        out.push_back(m);
        out.push_back(n);
        return {{out}};
    }

} // namespace infini
