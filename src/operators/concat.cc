#include "operators/concat.h"
#include "utils/operator_utils.h"

namespace infini
{
    ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
        : OperatorObj(OpType::Concat, inputs, {output})
    {
        int rank = inputs[0]->getRank();
        dim = get_real_axis(_dim, rank);
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs)
    {
        Shape dims = inputs[0]->getDims();
        // =================================== 作业 ===================================
        // TODO：修改 dims，返回正确的 concat 后的 shape
        // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
        // =================================== 作业 ===================================

        IT_ASSERT(!inputs.empty());
        const int rank = static_cast<int>(inputs[0]->getRank());
        IT_ASSERT(dim >= 0 && dim < rank);

        int sumDim = dims[dim];
        for (size_t i = 1; i < inputs.size(); ++i)
        {
            IT_ASSERT(static_cast<int>(inputs[i]->getRank()) == rank);
            const auto &cur = inputs[i]->getDims();
            for (int r = 0; r < rank; ++r)
            {
                if (r == dim)
                    continue;
                IT_ASSERT(cur[r] == dims[r], "Concat dims mismatch on non-concat axis");
            }
            sumDim += cur[dim];
        }
        dims[dim] = sumDim;

        return {{dims}};
    }

    std::string ConcatObj::toString() const
    {
        std::ostringstream os;
        os << "Concat[" << getGuid() << "]";
        os << "(";
        for (auto input : inputs)
            os << vecToString(input->getDims()) << ",";
        os << "dim=" << dim << ",";
        os << "input=";
        for (auto input : inputs)
            os << input->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }

} // namespace infini
