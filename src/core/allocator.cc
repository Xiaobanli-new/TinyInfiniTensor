#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================

        auto it = findFreeBlock(size);
        if (it != freeBlocks.end())
        {
            const size_t addr = it->first;
            const size_t blockSize = it->second;
            IT_ASSERT(blockSize >= size);
            freeBlocks.erase(it);

            const size_t remain = blockSize - size;
            if (remain > 0)
            {
                freeBlocks.emplace(addr + size, remain);
            }

            used += size;
            return addr;
        }

        const size_t addr = peak;
        peak += size;
        used += size;
        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        IT_ASSERT(used >= size);
        used -= size;
        addFreeBlock(addr, size);
    }
    std::map<size_t, size_t>::iterator Allocator::findFreeBlock(size_t size)
    {
        // first-fit: 找到第一个 size 足够的空闲块
        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it)
        {
            if (it->second >= size)
                return it;
        }
        return freeBlocks.end();
    }

    void Allocator::addFreeBlock(size_t addr, size_t size)
    {
        // 插入一个 free block，并与左右相邻块合并（coalescing）
        auto it = freeBlocks.lower_bound(addr);

        // 尝试与左侧块合并
        if (it != freeBlocks.begin())
        {
            auto left = std::prev(it);
            const size_t leftAddr = left->first;
            const size_t leftSize = left->second;
            if (leftAddr + leftSize == addr)
            {
                addr = leftAddr;
                size += leftSize;
                freeBlocks.erase(left);
            }
        }

        // 尝试与右侧块合并（重新定位迭代器）
        it = freeBlocks.lower_bound(addr);
        if (it != freeBlocks.end())
        {
            const size_t rightAddr = it->first;
            const size_t rightSize = it->second;
            if (addr + size == rightAddr)
            {
                size += rightSize;
                freeBlocks.erase(it);
            }
        }

        freeBlocks.emplace(addr, size);

        // 若空闲块位于堆顶（addr+size==peak），则可以把 peak 往回收缩。
        // 进一步：如果收缩后的新 peak 仍然与另一个空闲块相邻，也可以继续收缩。
        while (!freeBlocks.empty())
        {
            auto it = freeBlocks.upper_bound(peak);
            if (it == freeBlocks.begin())
                break;
            --it;
            const size_t blockAddr = it->first;
            const size_t blockSize = it->second;
            if (blockAddr + blockSize != peak)
                break;
            peak = blockAddr;
            freeBlocks.erase(it);
        }
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
