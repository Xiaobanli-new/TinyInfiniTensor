#include "core/graph.h"
#include "core/blob.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <numeric>
#include <queue>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================

        IT_ASSERT(topo_sort() == true);

        auto isSwapLast2Permute = [](const std::vector<int> &perm) -> bool
        {
            const int r = static_cast<int>(perm.size());
            if (r < 2)
                return false;
            for (int i = 0; i < r - 2; ++i)
                if (perm[i] != i)
                    return false;
            return perm[r - 2] == r - 1 && perm[r - 1] == r - 2;
        };

        auto isInversePermute = [](const std::vector<int> &p1,
                                   const std::vector<int> &p2) -> bool
        {
            if (p1.size() != p2.size())
                return false;
            const int r = static_cast<int>(p1.size());
            std::vector<int> inv(r, -1);
            for (int i = 0; i < r; ++i)
            {
                const int v = p1[i];
                if (v < 0 || v >= r || inv[v] != -1)
                    return false;
                inv[v] = i;
            }
            return inv == p2;
        };

        bool changed = true;
        while (changed)
        {
            changed = false;

            // 规则 1：消除连续 transpose（perm 互逆）
            for (size_t i = 0; i < ops.size(); ++i)
            {
                auto op1 = ops[i];
                if (op1->getOpType() != OpType::Transpose)
                    continue;
                auto t1 = as<TransposeObj>(op1);
                auto out1 = t1->getOutput();
                auto targets = out1->getTargets();
                if (targets.size() != 1)
                    continue;
                auto op2 = targets[0];
                if (!op2 || op2->getOpType() != OpType::Transpose)
                    continue;
                auto t2 = as<TransposeObj>(op2);
                if (t2->getInputs(0) != out1)
                    continue;
                if (!isInversePermute(t1->getPermute(), t2->getPermute()))
                    continue;

                auto in = t1->getInputs(0);
                auto out2 = t2->getOutput();
                for (auto &consumer : out2->getTargets())
                    consumer->replaceInput(out2, in);

                ops.erase(std::remove(ops.begin(), ops.end(), op1), ops.end());
                ops.erase(std::remove(ops.begin(), ops.end(), op2), ops.end());
                changed = true;
                break;
            }
            if (changed)
                continue;

            // 规则 2：将 transpose(交换最后两维) 融合到 matmul 的 transA/transB
            std::unordered_set<OperatorObj *> toRemove;
            for (auto &op : ops)
            {
                if (op->getOpType() != OpType::MatMul)
                    continue;
                auto mm = as<MatmulObj>(op);
                for (int inputIdx = 0; inputIdx < 2; ++inputIdx)
                {
                    auto in = mm->getInputs(inputIdx);
                    auto src = in->getSource();
                    if (!src || src->getOpType() != OpType::Transpose)
                        continue;
                    auto tr = as<TransposeObj>(src);
                    if (tr->getOutput() != in)
                        continue;
                    if (!isSwapLast2Permute(tr->getPermute()))
                        continue;

                    auto trIn = tr->getInputs(0);
                    mm->replaceInput(in, trIn);
                    if (inputIdx == 0)
                        mm->setTransA(!mm->getTransA());
                    else
                        mm->setTransB(!mm->getTransB());
                    toRemove.insert(src.get());
                    changed = true;
                }
            }
            if (!toRemove.empty())
            {
                ops.erase(std::remove_if(ops.begin(), ops.end(),
                                         [&](const Operator &op)
                                         { return toRemove.count(op.get()) != 0; }),
                          ops.end());
            }
        }

        // 清理不再被任何算子引用的张量
        {
            std::unordered_set<TensorObj *> referenced;
            referenced.reserve(tensors.size());
            for (auto &op : ops)
            {
                for (auto &t : op->getInputs())
                    referenced.insert(t.get());
                for (auto &t : op->getOutputs())
                    referenced.insert(t.get());
            }
            TensorVec kept;
            kept.reserve(tensors.size());
            for (auto &t : tensors)
                if (referenced.count(t.get()) != 0)
                    kept.emplace_back(t);
            tensors = std::move(kept);
        }

        // 重新构建 pred/succ 与 tensor source/target
        for (auto &t : tensors)
        {
            t->targets.clear();
            t->source.reset();
        }
        for (auto &op : ops)
        {
            op->predecessors.clear();
            op->successors.clear();
        }
        for (auto &op : ops)
        {
            for (auto &input : op->getInputs())
            {
                if (input)
                {
                    input->addTarget(op);
                    if (auto pred = input->getSource())
                    {
                        pred->addSuccessors(op);
                        op->addPredecessors(pred);
                    }
                }
            }
            for (auto &output : op->getOutputs())
            {
                if (output)
                    output->setSource(op);
            }
        }

        sorted = false;
        IT_ASSERT(topo_sort() == true);
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        std::unordered_map<TensorObj *, int> remainingUses;
        std::unordered_map<TensorObj *, size_t> bytes;
        std::unordered_map<TensorObj *, size_t> offsets;
        remainingUses.reserve(tensors.size());
        bytes.reserve(tensors.size());
        offsets.reserve(tensors.size());

        std::unordered_set<TensorObj *> keepAlive;
        keepAlive.reserve(tensors.size());
        for (auto &t : tensors)
        {
            bytes[t.get()] = t->getBytes();
            remainingUses[t.get()] = static_cast<int>(t->getTargets().size());
            if (t->getTargets().empty())
                keepAlive.insert(t.get());
        }

        auto ensureAlloc = [&](const Tensor &t)
        {
            auto *p = t.get();
            if (offsets.find(p) == offsets.end())
                offsets[p] = allocator.alloc(bytes[p]);
        };

        // 输入张量：dataMalloc 后会 setData
        for (auto &t : getInputs())
            ensureAlloc(t);

        // 遍历 op：分配输出、回收“已完成最后一次使用”的输入
        for (auto &op : ops)
        {
            for (auto &out : op->getOutputs())
                ensureAlloc(out);

            for (auto &in : op->getInputs())
            {
                auto *p = in.get();
                auto it = remainingUses.find(p);
                if (it == remainingUses.end())
                    continue;
                if (it->second > 0)
                    --(it->second);
                if (it->second == 0 && keepAlive.count(p) == 0)
                {
                    auto offIt = offsets.find(p);
                    if (offIt != offsets.end())
                        allocator.free(offIt->second, bytes[p]);
                }
            }
        }

        void *base = allocator.getPtr();
        for (auto &t : tensors)
        {
            auto it = offsets.find(t.get());
            IT_ASSERT(it != offsets.end(), "Tensor not allocated in dataMalloc");
            void *ptr = static_cast<void *>(static_cast<char *>(base) + it->second);
            t->setDataBlob(make_ref<BlobObj>(runtime, ptr));
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini
